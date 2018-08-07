from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys

import numpy as np
import os

from os.path import join
from os import makedirs

from keras.models import save_model
from keras.models import load_model
from math import ceil
import cv2


class Pix2Pix():
    def __init__(self, config=None):
        '''
        Should be used with one channel only
        :param config:
        '''
        if config is not None:
            self.img_rows = config.getint("Model", "rows")
            self.img_cols = config.getint("Model", "cols")
            self.channels = config.getint("Model", "channels")
            self.output_folder = config.get("Model", "output_folder")
            self.load = config.getboolean("Model", "load")
            self.load_folder = join(config.get("Model", "load_folder"), "models")
            self.last_epoch = config.getint('Model', "last_epoch") + 1
            self.backup = config.getboolean("Model", 'backup')
            self.backup_interval = config.getint("Model", 'backup_interval')
            lr = config.getfloat("Model", "lr")

        else:
            self.img_rows = 256
            self.img_cols = 256
            self.channels = 3
            self.output_folder = "images"
            self.save_path = "model"
            self.load = False
            self.last_epoch = 1
            self.backup = False

        self.save_folder = join(self.output_folder, "models")
        self.log_folder = join(self.output_folder, "logs")
        self.log_file = join(self.log_folder, "logs.csv")
        makedirs(self.save_folder, exist_ok=True)
        makedirs(self.log_folder, exist_ok=True)

        self.img_dim = self.img_rows * self.img_cols * self.channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        self.discriminator.summary()
        self.generator.summary()

        if self.load: self.load_model()

        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)
        d7 = conv2d(d6, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)


    def train(self, data, epochs, batch_size=1, sample_interval=50):
        X, _ = data
        import csv
        with open(self.log_file, 'a') as fout:
            logger = csv.writer(fout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if self.last_epoch != 1: logger.writerow(["Epoch", "Batch", "D loss", "Accuracy", "G loss"])

            start_time = datetime.datetime.now()
            Xc = []
            for img in X:
                if self.channels == 1:
                    Xc.append(cv2.Canny(cv2.blur(img, (5, 5)), 10, 15))
                else:
                    Xc.append(cv2.cvtColor(
                        cv2.Canny(cv2.blur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5)), 10, 15), cv2.COLOR_GRAY2BGR))
            Xc = np.array(Xc)
            if self.channels == 1:
                X = np.expand_dims(X, axis=3)
                Xc = np.expand_dims(Xc, axis=3)

            # print(Xc.shape)

            X = (X.astype(np.float32) - 127.5) / 127.5
            Xc = (Xc.astype(np.float32) - 127.5) / 127.5

            num_batches = ceil(len(X) / batch_size)
            # Adversarial loss ground truths
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            indicies = np.arange(0, len(X))
            for epoch in range(self.last_epoch, epochs + self.last_epoch):
                for batch_i in range(1, num_batches+1):
                    ind = np.random.choice(indicies, size=batch_size)

                    imgs_A, imgs_B = X[ind], Xc[ind]
                    # print(imgs_A.shape, imgs_B.shape)
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Condition on B and generate a translated version
                    fake_A = self.generator.predict(imgs_B)

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                    elapsed_time = datetime.datetime.now() - start_time
                    # Plot the progress
                    print("[Epoch %d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch,
                                                                                                    batch_i,
                                                                                                    d_loss[0],
                                                                                                    100 * d_loss[1],
                                                                                                    g_loss[0],
                                                                                                    elapsed_time))

                    logger.writerow([epoch, batch_i, d_loss[0], 100 * d_loss[1], g_loss[0]])
                    fout.flush()

                    # If at save interval => save generated image samples
                    if ((epoch*num_batches)+batch_i) % sample_interval == 0:
                        self.sample_images(epoch, batch_i, imgs_A[:3], imgs_B[:3])

                    if self.backup and epoch % self.backup_interval == 0:
                        self.save_model(ext='_e' + str(epoch))

            self.save_model()

    def feed(self, img):
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = img + (np.random.rand(img.shape) - 0.5)
        img[img < 0] = 0;
        img[img > 1] = 1
        self.__feed = self.generator.predict(img.reshape((1,) + self.img_shape))

    def generate(self, n):

        gen_imgs = []
        for _ in range(n):
            self.__feed = self.generator(self.__feed.reshape((1,) + self.img_shape))
            gen_imgs.append(self.__feed.copy())
        return gen_imgs

    def sample_images(self, epoch, batch_i, imgs_A, imgs_B):
        r, c = 3, 3

        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(join(self.output_folder, "pix2pix_e%d-b%d.png" % (epoch, batch_i)))
        plt.close()

    def load_model(self):

        self.discriminator.load_weights(join(self.load_folder, "discriminator"))

        self.generator.load_weights(join(self.load_folder, "generator"))

    def save_model(self, ext=''):
        self.generator.save_weights(join(self.save_folder, "generator" + ext))
        self.discriminator.save_weights(join(self.save_folder, "discriminator" + ext))


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=200)
