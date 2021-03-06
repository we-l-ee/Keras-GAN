from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
from os.path import join
from os import makedirs

from keras.models import save_model
from keras.models import load_model


class BIGAN():
    def __init__(self, config=None):
        if config is not None:
            self.img_rows = config.getint("Model", "rows")
            self.img_cols = config.getint("Model", "cols")
            self.channels = config.getint("Model", "channels")
            self.output_folder = config.get("Model", "output_folder")
            self.load = config.getboolean("Model", "load")
            self.load_folder = join(config.get("Model", "load_folder"), 'models')
            self.latent_dim = config.getint("Model", "latent_dim")
            self.last_epoch = config.getint('Model', "last_epoch") + 1
            self.backup = config.getboolean("Model", 'backup')
            self.backup_interval = config.getint("Model", 'backup_interval')
        else:
            self.img_rows = 28
            self.img_cols = 28
            self.channels = 1
            self.output_folder = "images"
            self.save_path = "model"
            self.load = False
            self.latent_dim = 100
            self.last_epoch = 0

        self.save_folder = join(self.output_folder, "models")
        self.log_folder = join(self.output_folder, "logs")
        self.log_file = join(self.log_folder, "logs.csv")
        makedirs(self.log_folder, exist_ok=True)
        makedirs(self.save_folder, exist_ok=True)

        self.img_dim = self.img_rows * self.img_cols * self.channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        if self.load:
            self.load_model()
            self.last_epoch = 1

        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim,))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                     optimizer=optimizer)

    def build_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim,))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)

    def train(self, data, epochs, batch_size=128, sample_interval=50):
        X, _ = data
        if self.channels == 1:
            X = np.expand_dims(X, axis=3)

        import csv
        with open(self.log_file, 'a') as fout:
            logger = csv.writer(fout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            logger.writerow(["Epoch", "loss", "acc", "G loss"])

            # Rescale -1 to 1
            X = (X.astype(np.float32) - 127.5) / 127.5

            # Adversarial ground truths
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            for epoch in range(self.last_epoch, epochs + self.last_epoch):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Sample noise and generate img
                z = np.random.normal(size=(batch_size, self.latent_dim))
                imgs_ = self.generator.predict(z)

                # Select a random batch of images and encode
                idx = np.random.randint(0, X.shape[0], batch_size)
                imgs = X[idx]
                z_ = self.encoder.predict(imgs)

                # Train the discriminator (img -> z is valid, z -> img is fake)
                d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
                d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (z -> img is valid and img -> z is is invalid)
                g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])

                # Plot the progress
                print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
                logger.writerow([epoch, d_loss[0], 100 * d_loss[1], g_loss[0]])
                fout.flush()

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.sample_interval(epoch)

                if self.backup and epoch % self.backup_interval == 0:
                    self.save_model(ext='_e' + str(epoch))

            self.save_model()

    def generate(self, n):
        z = np.random.normal(size=(n, self.latent_dim))
        return self.generator.predict(z)

    def sample_interval(self, epoch):
        r, c = 5, 5
        z = np.random.normal(size=(r * c, self.latent_dim))
        gen_imgs = self.generator.predict(z)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(join(self.output_folder, "BiGAN_%d.png" % epoch))
        plt.close()

    def load_model(self):
        self.discriminator.load_weights(join(self.load_folder, "discriminator"))
        self.generator.load_weights(join(self.load_folder, "generator"))
        self.encoder.load_weights(join(self.load_folder, "encoder"))

    def save_model(self, ext=''):
        self.generator.save_weights(join(self.save_folder, "generator" + ext))
        self.encoder.save_weights(join(self.save_folder, "encoder" + ext))
        self.discriminator.save_weights(join(self.save_folder, "discriminator" + ext))

# if __name__ == '__main__':
#     bigan = BIGAN()
#     bigan.train(epochs=40000, batch_size=32, sample_interval=400)
