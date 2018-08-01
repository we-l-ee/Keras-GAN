from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

from os.path import join
from os import makedirs

from keras.models import save_model
from keras.models import load_model


class WGAN():
    def __init__(self, config=None):
        if config is not None:
            self.img_rows = config.getint("Model", "rows")
            self.img_cols = config.getint("Model", "cols")
            self.channels = config.getint("Model", "channels")
            self.output_folder = config.get("Model", "output_folder")
            self.save_folder = join(self.output_folder, "models")
            self.load = config.getboolean("Model", "load")
            self.load_folder = config.get("Model", "load_folder")
            self.latent_dim = config.getint("Model", "latent_dim")
        else:
            self.img_rows = 28
            self.img_cols = 28
            self.channels = 1
            self.output_folder = "images"
            self.save_path = "model"
            self.load = False
            self.latent_dim = 100

        makedirs(self.output_folder, exist_ok=True)
        self.log_folder = join(self.output_folder, "logs")
        makedirs(self.log_folder, exist_ok=True)
        self.log_file = join(self.log_folder, "logs.csv")

        self.img_dim = self.img_rows * self.img_cols * self.channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01

        if not self.load:
            optimizer = RMSprop(lr=0.00005)
            # Build and compile the critic
            self.critic = self.build_critic()
            self.critic.compile(loss=self.wasserstein_loss,
                                optimizer=optimizer,
                                metrics=['accuracy'])

            # Build the generator
            self.generator = self.build_generator()
        else: self.load_model()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))

        # model.add(Activation("tanh")) # original
        # should it not be this way?
        model.add(Dense(self.img_dim, activation='tanh'))  # modified

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X, epochs, batch_size=128, sample_interval=50, **kwargs):
        import csv
        with open(self.log_file, 'w') as fout:
            logger = csv.writer(fout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            logger.writerow(["Epoch", "D loss", "G loss"])

            # Rescale -1 to 1
            X = (X.astype(np.float32) - 127.5) / 127.5
            X = np.expand_dims(X, axis=3)

            # Adversarial ground truths
            valid = -np.ones((batch_size, 1))
            fake = np.ones((batch_size, 1))

            for epoch in range(epochs):

                for _ in range(self.n_critic):

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Select a random batch of images
                    idx = np.random.randint(0, X.shape[0], batch_size)
                    imgs = X[idx]

                    # Sample noise as generator input
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                    # Generate a batch of new images
                    gen_imgs = self.generator.predict(noise)

                    # Train the critic
                    d_loss_real = self.critic.train_on_batch(imgs, valid)
                    d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                    # Clip critic weights
                    for l in self.critic.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                # ---------------------
                #  Train Generator
                # ---------------------

                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
                logger.writerow([epoch, 1 - d_loss[0], 1 - g_loss[0]])
                fout.flush()

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.sample_images(epoch)
            save_model()

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(join(self.output_folder, "DUALGAN_%d.png" % epoch))
        plt.close()

    def load_model(self):
        optimizer = RMSprop(lr=0.00005)
        # Build and compile the critic
        self.critic = load_model(join(self.load_folder,"critic"))
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = load_model(join(self.load_folder,"generator"))

    def save_model(self):
        save_model(self.critic, join(self.save_folder,"critic"))

        save_model(self.generator, join(self.save_folder,"generator"))

if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=4000, batch_size=32, sample_interval=50)
