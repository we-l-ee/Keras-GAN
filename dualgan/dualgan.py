from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np
from os.path import join
from os import makedirs

from keras.models import save_model
from keras.models import load_model

class DUALGAN():
    def __init__(self, config=None):
        if config is not None:
            self.img_rows = config.getint("Model", "rows")
            self.img_cols = config.getint("Model", "cols")
            self.channels = config.getint("Model", "channels")
            self.output_folder = config.get("Model", "output_folder")
            self.load = config.getboolean("Model", "load")
            self.load_folder = join(config.get("Model", "load_folder"),"models")
            self.last_epoch = config.getint('Model', "last_epoch")+1
            self.backup = config.getboolean("Model", 'backup')
            self.backup_interval = config.getint("Model", 'backup_interval')
        else:
            self.img_rows = 28
            self.img_cols = 28
            self.channels = 1
            self.output_folder = "images"
            self.save_path = "model"
            self.load=False
            self.last_epoch = 1

        self.log_folder = join(self.output_folder, "logs")
        self.save_folder = join(self.output_folder, "models")
        self.log_file = join(self.log_folder, "logs.csv")

        makedirs(self.log_folder, exist_ok=True)
        makedirs(self.save_folder, exist_ok=True)

        self.img_dim = self.img_rows * self.img_cols * self.channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)


        # Build and compile the discriminators
        optimizer = Adam(0.0002, 0.5)
        # optimizer = RMSprop()
        self.D_A = self.build_discriminator()


        self.D_B = self.build_discriminator()


        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.G_AB = self.build_generator()
        self.G_BA = self.build_generator()

        if self.load: self.load_model()

        self.D_A.compile(loss=self.wasserstein_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        self.D_B.compile(loss=self.wasserstein_loss,
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # For the combined model we will only train the generators
        self.D_A.trainable = False
        self.D_B.trainable = False


        # The generator takes images from their respective domains as inputs
        imgs_A = Input(shape=(self.img_dim,))
        imgs_B = Input(shape=(self.img_dim,))

        # Generators translates the images to the opposite domain
        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        # The discriminators determines validity of translated images
        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        # Generators translate the images back to their original domain
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)



        # The combined model  (stacked generators and discriminators)
        self.combined = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B])


        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'],
                              optimizer=optimizer,
                              loss_weights=[1, 1, 100, 100])

        self.combined.summary()

    def build_generator(self):

        X = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(256, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(self.img_dim, activation='tanh'))

        X_translated = model(X)

        return Model(X, X_translated)

    def build_discriminator(self):

        img = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(512, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1))

        validity = model(img)

        return Model(img, validity)

    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    # 12 Generator? for each 30 degree?
    def train(self, data, epochs, batch_size=128, sample_interval=50):
        X, _ = data

        import csv
        with open(self.log_file, 'a') as fout:
            logger = csv.writer(fout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if self.last_epoch != 1: logger.writerow(["Epoch", "D1 loss", "D2 loss", "G loss"])

            # Rescale -1 to 1
            X = (X.astype(np.float32) - 127.5) / 127.5

            # Domain A and B (rotated)
            X_A = X[:int(X.shape[0] / 2)]
            X_B = scipy.ndimage.interpolation.rotate(X[int(X.shape[0] / 2):], 90, axes=(1, 2))

            X_A = X_A.reshape(X_A.shape[0], self.img_dim)
            X_B = X_B.reshape(X_B.shape[0], self.img_dim)

            clip_value = 0.01
            n_critic = 4

            # Adversarial ground truths
            valid = -np.ones((batch_size, 1))
            fake = np.ones((batch_size, 1))

            for epoch in range(self.last_epoch, epochs+self.last_epoch):

                # Train the discriminator for n_critic iterations
                for _ in range(n_critic):

                    # ----------------------
                    #  Train Discriminators
                    # ----------------------

                    # Sample generator inputs
                    imgs_A = self.sample_generator_input(X_A, batch_size)
                    imgs_B = self.sample_generator_input(X_B, batch_size)

                    # Translate images to their opposite domain
                    fake_B = self.G_AB.predict(imgs_A)
                    fake_A = self.G_BA.predict(imgs_B)

                    # Train the discriminators
                    D_A_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                    D_A_loss_fake = self.D_A.train_on_batch(fake_A, fake)

                    D_B_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                    D_B_loss_fake = self.D_B.train_on_batch(fake_B, fake)

                    D_A_loss = 0.5 * np.add(D_A_loss_real, D_A_loss_fake)
                    D_B_loss = 0.5 * np.add(D_B_loss_real, D_B_loss_fake)

                    # Clip discriminator weights
                    for d in [self.D_A, self.D_B]:
                        for l in d.layers:
                            weights = l.get_weights()
                            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                            l.set_weights(weights)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

                # Plot the progress
                print("%d [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                      % (epoch, D_A_loss[0], D_B_loss[0], g_loss[0]))
                logger.writerow([epoch, D_A_loss[0], D_B_loss[0], g_loss[0]])
                fout.flush()

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.save_imgs(epoch, X_A, X_B)

                if self.backup and epoch % self.backup_interval == 0:
                    self.save_model(ext='_e'+str(epoch))

            self.save_model()

    def feed(self, img):
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = img.reshape(self.img_dim)
        img = img + (np.random.rand(self.img_dim)-0.5)
        img[img<0] = 0; img[img>1]=1
        img = img.reshape(1, self.img_dim)
        self.__feed = self.G_AB.predict(img)

    def generate(self, n):

        gen_imgs=[]
        for _ in range(n):
            self.__feed = (np.random.rand(self.img_dim) - 0.5) * 2
            # img = img.reshape(1, self.img_dim)
            self.__feed = self.G_AB.predict(self.__feed.reshape(1, self.img_dim))
            gen_imgs.append(self.__feed.copy().reshape(self.img_shape))
        return gen_imgs

    def save_imgs(self, epoch, X_A, X_B):
        r, c = 4, 4

        # Sample generator inputs
        imgs_A = self.sample_generator_input(X_A, c)
        imgs_B = self.sample_generator_input(X_B, c)

        # Images translated to their opposite domain
        fake_B = self.G_AB.predict(imgs_A)
        fake_A = self.G_BA.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A])
        gen_imgs = gen_imgs.reshape((r, c, self.img_rows, self.img_cols, self.channels))

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[i, j, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(join(self.output_folder, "DUALGAN_%d.png" % epoch))
        plt.close()



    def load_model(self):

        self.D_A.load_weights(join(self.load_folder,"D_A"))

        self.D_B.load_weights(join(self.load_folder,"D_B"))

        self.G_AB.load_weights(join(self.load_folder,"G_AB"))
        self.G_BA.load_weights(join(self.load_folder,"G_BA"))

    def save_model(self, ext=''):
        self.D_A.save_weights(join(self.save_folder,"D_A"+ext))

        self.D_B.save_weights(join(self.save_folder,"D_B"+ext))

        self.G_AB.save_weights(join(self.save_folder,"G_AB"+ext))
        self.G_BA.save_weights(join(self.save_folder,"G_BA"+ext))

# if __name__ == '__main__':
#     gan = DUALGAN()
#     gan.train(epochs=30000, batch_size=32, sample_interval=200)
