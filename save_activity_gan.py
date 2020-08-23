from __future__ import print_function, division

import keras.backend as K
import keras
# from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.models import load_model


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
from six.moves import cPickle
from collections import namedtuple
import os
import argparse
import datetime
import utils
from loggingreporter import LoggingReporter


class GAN():
    def __init__(self, args):
        self.latent_dim = 100
        self.activation = args.activation
        self.rawdata_dir = 'rawdata/' + self.activation + '_' + '1024-512-256-1'
        self.img_dir = "images_" + self.activation

        optimizer = Adam(args.lr, args.beta_1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        self.discriminator.trainable = False

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        ganInput = Input(shape=(self.latent_dim,))
        img = self.generator(ganInput)
        ganOutput = self.discriminator(img)

        self.gan = Model(ganInput, ganOutput)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Load the dataset
        self.trn, self.tst = utils.get_data(self.activation, args.isTrain, args.fake_size)
        print("tst shape:", self.tst.X.shape)

        ## Initialize LoggingReporter
        self.log = LoggingReporter(self.trn, self.tst, self.rawdata_dir, self.img_dir, do_save_func=self.do_report)

    def build_generator(self):

        if self.activation == 'tanh' or self.activation == 'relu':
            ###### Use tanh or relu #######
            model = Sequential()

            model.add(Dense(256, input_dim=self.latent_dim, activation=self.activation))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(512, activation=self.activation))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(1024, activation=self.activation))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(784, activation='tanh'))

        elif self.activation == 'leakyrelu':
            ###### Use LeakyReLU #######
            model = Sequential()

            model.add(Dense(256, input_dim=self.latent_dim))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(1024))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(784, activation='tanh'))

        model.summary()
        return model

    def build_discriminator(self):

        if self.activation == 'tanh' or self.activation == 'relu':
            ############## tanh or relu ##############
            input_layer = Input(shape=(28*28*1,))
            layer1 = Dense(1024, activation=self.activation)(input_layer)
            layer2 = Dense(512, activation=self.activation)(layer1)
            layer3 = Dense(256, activation=self.activation)(layer2)
            output_layer = Dense(1,activation='sigmoid')(layer3)
            model = Model(inputs=input_layer, outputs=output_layer)

        elif self.activation == 'leakyrelu':
            ############# leakyrelu #############
            input_layer = Input(shape=(28*28*1,))
            layer1 = Dense(1024)(input_layer)
            act1 = LeakyReLU(alpha=0.2)(layer1)
            layer2 = Dense(512)(act1)
            act2 = LeakyReLU(alpha=0.2)(layer2)
            layer3 = Dense(256)(act2)
            act3 = LeakyReLU(alpha=0.2)(layer3)
            output_layer = Dense(1,activation='sigmoid')(act3)
            model = Model(inputs=input_layer, outputs=output_layer)

        model.summary()
        return model

    def do_report(self,epoch):
        # Only log activity for some epochs.  Mainly this is to make things run faster.

        if epoch < 500:    # Then every 10th
            return (epoch % 10 == 0)
        elif epoch < 2000:    # Then every 5th
            return (epoch % 5 == 0)
        elif epoch < 4000:    # Then every 10th
            return (epoch % 10 == 0)
        else:                # Then every 100th
            return (epoch % 100 == 0)


    def train(self, epochs, batch_size=128, isTrain=False):
        # Adversarial ground truths
        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))

        for epoch in range(epochs):
            self.log.on_epoch_begin(self.discriminator, epoch)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, self.trn.X.shape[0], batch_size)
            imgs = self.trn.X[idx] #imgs.shape = (128,28,28,1)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            self.log.on_batch_begin(self.discriminator, batch_size)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.gan.train_on_batch(noise, valid)


            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if isTrain:
                if epoch == (epochs-1): #only save model at last epoch
                    utils.save_model(self.generator,self.discriminator, self.activation)

                # If at save interval => save generated image samples
                if self.do_report is not None and self.do_report(epoch):
                    utils.sample_images(self.generator, self.img_dir, epoch)

            else: # if not training, save activity for MI calculation
                self.log.on_epoch_end(self.discriminator, d_loss_real[0], d_loss_fake[0], g_loss, epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Information Bottleneck GAN')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30000,
                        help='number of epochs to train (default: 30000)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--beta_1', type=float, default=0.5,
                        help='beta_1 for ADAM (default: 0.5)')
    parser.add_argument('--activation', type=str, choices = ['tanh', 'relu', 'leakyrelu'], required = True,
                        help='options: tanh, relu, leakyrelu')
    parser.add_argument('--isTrain', type=int, choices = [0,1], required=True,
                        help='0 to get each layer activities, 1 to obtain Generator to construct fake+real dataset')
    parser.add_argument('--fake_size', type=int, default=5000,
                        help='proportion of fake images. Must be <=10000  (default:5000)')

    args = parser.parse_args()

    if args.fake_size > 10000:
        parser.error("fake_size should be <= 10000")

    print("Start:", datetime.datetime.now())
    ###################################################
    gan = GAN(args)
    gan.log.on_train_begin(gan.discriminator)
    gan.train(epochs=args.epochs, batch_size=args.batch_size, isTrain=args.isTrain)
    ###################################################
    print("End:", datetime.datetime.now())
