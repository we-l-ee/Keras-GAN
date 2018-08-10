from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

from os.path import join
from os import makedirs

# if there is shape error set the dimension of latent dim to given error dim.
class INFOGAN():
    def __init__(self, config=None):
        if config is not None:
            self.img_rows = config.getint("Model", "rows")
            self.img_cols = config.getint("Model", "cols")
            self.channels = config.getint("Model", "channels")
            self.output_folder = config.get("Model", "output_folder")
            self.load = config.getboolean("Model", "load")
            self.load_folder = join(config.get("Model", "load_folder"), "models")
            self.latent_dim = config.getint("Model", "latent_dim")
            self.last_epoch = config.getint('Model', "last_epoch")+1
            self.backup = config.getboolean("Model", 'backup')
            self.backup_interval = config.getint("Model", 'backup_interval')
            self.num_classes = config.getint("Model", 'num_classes')

        else:
            self.img_rows = 28
            self.img_cols = 28
            self.channels = 1
            self.output_folder = "images"
            self.save_path = "model"
            self.load = False
            self.latent_dim = 72
            self.last_epoch = 1
            self.num_classes = 10

        self.save_folder = join(self.output_folder, "models")
        self.log_folder = join(self.output_folder, "logs")
        self.log_file = join(self.log_folder, "logs.csv")
        makedirs(self.log_folder, exist_ok=True)
        makedirs(self.save_folder, exist_ok=True)
        self.img_rows = self.img_rows // 4 * 4; self.img_cols = self.img_cols // 4 * 4
        self.img_dim = self.img_rows * self.img_cols * self.channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', self.mutual_info_loss]

        # Build and the discriminator and recognition network
        self.discriminator, self.auxilliary = self.build_disk_and_q_net()


        # Build the generator
        self.generator = self.build_generator()

        if self.load:
            self.load_model()
        else:
            self.last_epoch = 1

        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the recognition network Q
        self.auxilliary.compile(loss=[self.mutual_info_loss],
                                optimizer=optimizer,
                                metrics=['accuracy'])

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        valid = self.discriminator(img)
        # The recognition network produces the label
        target_label = self.auxilliary(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(gen_input, [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * self.img_rows // 4 * self.img_cols // 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows // 4, self.img_cols // 4, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        gen_input = Input(shape=(self.latent_dim,))
        img = model(gen_input)


        return Model(gen_input, img)

    def build_disk_and_q_net(self):

        img = Input(shape=self.img_shape)

        # Shared layers between discriminator and recognition network
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())

        model.summary()

        img_embedding = model(img)

        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)

        # Recognition
        q_net = Dense(128, activation='relu')(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        # Return discriminator and recognition network
        return Model(img, validity), Model(img, label)

    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 62))
        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

        return sampled_noise, sampled_labels

    def train(self, data, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X, y_train = data

        # Rescale -1 to 1
        X = (X.astype(np.float32) - 127.5) / 127.5
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        import csv
        with open(self.log_file, 'a') as fout:
            logger = csv.writer(fout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if self.last_epoch != 1: logger.writerow(["Epoch", "D loss", "acc", "Q loss", "G loss"])

            for epoch in range(self.last_epoch, epochs + self.last_epoch):

                # ---------------------
                #  Train Discriminator    def train(self, X, epochs, batch_size=128, sample_interval=50, **kwargs):
                # ---------------------

                # Select a random half batch of images
                idx = np.random.randint(0, X.shape[0], batch_size)
                imgs = X[idx]

                # Sample noise and categorical labels
                sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
                gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)

                # Generate a half batch of new images
                gen_imgs = self.generator.predict(gen_input)

                # Train on real and generated data
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

                # Avg. loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator and Q-network
                # ---------------------

                g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])

                # Plot the progress
                print("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss[1], g_loss[2]))
                logger.writerow([epoch, d_loss[0], 100 * d_loss[1], g_loss[1], g_loss[2]])
                fout.flush()

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.sample_interval(epoch)

                if self.backup and epoch % self.backup_interval == 0:
                    self.save_model(ext='_e' + str(epoch))

            self.save_model()

    def sample_interval(self, epoch):
        r, c = 10, 10

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(c)

            label = to_categorical(np.full(fill_value=i, shape=(r, 1)), num_classes=self.num_classes)
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5
            for j in range(r):
                axs[j, i].imshow(gen_imgs[j, :, :, 0], cmap='gray')
                axs[j, i].axis('off')
        fig.savefig(join(self.output_folder, "InfoGAN_%d.png" % epoch))
        plt.close()

    def load_model(self):

        self.discriminator.load_weights(join(self.load_folder, "discriminator"))

        self.generator.load_weights(join(self.load_folder, "generator"))
        self.auxilliary.load_weights(join(self.load_folder, "auxilliary"))

    def save_model(self, ext=''):
        self.generator.save_weights(join(self.save_folder, "generator" + ext))
        self.auxilliary.save_weights(join(self.save_folder, "auxilliary" + ext))
        self.discriminator.save_weights(join(self.save_folder, "discriminator" + ext))


if __name__ == '__main__':
    infogan = INFOGAN()
    infogan.train(epochs=50000, batch_size=128, sample_interval=50)
