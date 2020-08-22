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
        # optimizer = SGD(lr=0.01, momentum=0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        # self.generator.compile(loss='binary_crossentropy',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])


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
        self.log = LoggingReporter(self.trn, self.tst, self.discriminator,
                                    self.rawdata_dir, self.img_dir, do_save_func=self.do_report)

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

    # ## loggingreporter ##
    # def on_train_begin(self, model):
    #     if not os.path.exists(self.rawdata_dir):
    #         print("Making directory", self.rawdata_dir)
    #         os.makedirs(self.rawdata_dir)
    #
    #     if not os.path.exists(self.img_dir):
    #         print("Making directory", self.img_dir)
    #         os.makedirs(self.img_dir)
    #
    #     # Indexes of the layers which we keep track of. Basically, this will be any layer
    #     # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
    #     self.layerixs = []
    #
    #     # Functions return activity of each layer
    #     self.layerfuncs = []
    #
    #     # Functions return weights of each layer
    #     self.layerweights = []
    #     for lndx, l in enumerate(model.layers):
    #         if hasattr(l, 'kernel'):
    #             self.layerixs.append(lndx)
    #             self.layerfuncs.append(K.function(model.inputs, [l.output,]))
    #             self.layerweights.append(l.kernel)
    #
    #     inputs = [model._feed_inputs,
    #               model._feed_targets,
    #               model._feed_sample_weights,
    #               K.learning_phase()]
    #
    #     # Get gradients of all the relevant layers at once
    #     grads = model.optimizer.get_gradients(model.total_loss, self.layerweights)
    #     self.get_gradients = K.function(inputs=inputs, outputs=grads)
    #
    #     # Get cross-entropy loss
    #     self.get_loss = K.function(inputs=inputs, outputs=[model.total_loss,])

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

    # def on_epoch_begin(self, model, epoch):
    #     if self.do_report is not None and not self.do_report(epoch):
    #         # Don't log this epoch
    #         self._log_gradients = False
    #     else:
    #         # We will log this epoch.  For each batch in this epoch, we will save the gradients (in on_batch_begin)
    #         # We will then compute means and vars of these gradients
    #
    #         self._log_gradients = True
    #         self._batch_weightnorm = []
    #
    #         self._batch_gradients = [ [] for _ in model.layers[1:] ] #model.layers[1:] to skip Input layer
    #
    #         # Indexes of all the training data samples. These are shuffled and read-in in chunks of SGD_BATCHSIZE
    #         ixs = list(range(len(self.trn.X)))
    #         np.random.shuffle(ixs)
    #         self._batch_todo_ixs = ixs
    #
    # def on_batch_begin(self, model, batch_size):
    #     if not self._log_gradients:
    #         # We are not keeping track of batch gradients, so do nothing
    #         return
    #
    #     # Sample a batch
    #     batchsize = batch_size
    #     cur_ixs = self._batch_todo_ixs[:batchsize]
    #     # Advance the indexing, so next on_batch_begin samples a different batch
    #     self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]
    #
    #     # Get gradients for this batch
    #
    #     x, y, weights = model._standardize_user_data(self.trn.X[cur_ixs,:], self.trn.y[cur_ixs])
    #     inputs = [x, y, weights, 1] # 1 indicates training phase
    #     for lndx, g in enumerate(self.get_gradients(inputs)):
    #         # g is gradients for weights of lndx's layer
    #         oneDgrad = np.reshape(g, [-1, 1])                  # Flatten to one dimensional vector
    #         # print(len(self._batch_gradients)) #len=3
    #         # print("Indx:%d" % lndx) #indx=0,1,2,3
    #         # print(g.shape) #indx0:(784,1024), indx1:(1024,512), indx2:(512,256),
    #         # print(oneDgrad.shape) #indx0:(802816,1), indx1:(524288,1), indx2:(131072,1)
    #         self._batch_gradients[lndx].append(oneDgrad)
    #
    # def on_epoch_end(self, model, d_loss_real, d_loss_fake, g_loss, epoch):
    #     if self.do_report is not None and not self.do_report(epoch):
    #         # Don't log this epoch
    #         return
    #     d_loss = 0.5*(d_loss_real+d_loss_fake)
    #     # Get overall performance
    #     data = {
    #         'weights_norm' : [],   # L2 norm of weights
    #         'gradmean'     : [],   # Mean of gradients
    #         'gradstd'      : [],   # Std of gradients
    #         'activity_tst' : []    # Activity in each layer for test set
    #     }
    #
    #     for lndx, layerix in enumerate(self.layerixs):
    #         clayer = model.layers[layerix]
    #
    #         data['weights_norm'].append( np.linalg.norm(K.get_value(clayer.kernel)) )
    #
    #         stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
    #         data['gradmean'    ].append( np.linalg.norm(stackedgrads.mean(axis=1)) )
    #         data['gradstd'     ].append( np.linalg.norm(stackedgrads.std(axis=1)) )
    #
    #         # if self.cfg['FULL_MI']:
    #         #     data['activity_tst'].append(self.layerfuncs[lndx]([self.full.X,])[0])
    #         # else:
    #         #     data['activity_tst'].append(self.layerfuncs[lndx]([self.tst.X,])[0])
    #
    #         data['activity_tst'].append(self.layerfuncs[lndx]([self.tst.X,])[0])
    #
    #     fname = self.rawdata_dir + "/epoch%08d"% epoch
    #     print("Saving", fname)
    #     with open(fname, 'wb') as f:
    #         cPickle.dump({'ACTIVATION':self.activation , 'epoch':epoch, 'data':data,
    #         'd_loss':d_loss, 'd_loss_real':d_loss_real, 'd_loss_fake':d_loss_fake ,'g_loss':g_loss}, f, cPickle.HIGHEST_PROTOCOL)


    def train(self, epochs, batch_size=128, isTrain=False):
        # Adversarial ground truths
        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))

        for epoch in range(epochs):
            ## loggingreporter ##
            self.log.on_epoch_begin(epoch)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, self.trn.X.shape[0], batch_size)
            imgs = self.trn.X[idx] #imgs.shape = (128,28,28,1)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            ## loggingreporter ##
            self.log.on_batch_begin(batch_size)
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
                    self.save_model(self.generator,self.discriminator)

                # If at save interval => save generated image samples
                if self.do_report is not None and self.do_report(epoch):
                    self.sample_images(epoch)

            else: # if not training, save activity for MI calculation
                self.log.on_epoch_end(d_loss_real[0], d_loss_fake[0], g_loss, epoch)

    def save_model(self, generator, discriminator):
        generator.save("models/generator_"+self.activation +".h5")
        discriminator.save("models/discriminator_"+self.activation+".h5")

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = np.reshape(gen_imgs, [-1, 28,28,1])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.img_dir + "/%d.png" % epoch)
        plt.close()


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
    parser.add_argument('--activation', type=str, default='tanh', choices = ['tanh', 'relu', 'leakyrelu'],
                        help='options: tanh, relu, leakyrelu (default:tanh)')
    parser.add_argument('--isTrain', type=int, choices = [0,1], required=True,
                        help='0 to get each layer activities, 1 to obtain Generator to construct fake+real dataset (default:0)')
    parser.add_argument('--fake_size', type=int, default=5000,
                        help='proportion of fake images. Must be <=10000  (default:5000)')

    args = parser.parse_args()

    if args.fake_size > 10000:
        parser.error("fake_size should be <= 10000")

    print("Start:", datetime.datetime.now())
    ###################################################
    gan = GAN(args)
    gan.log.on_train_begin()
    gan.train(epochs=args.epochs, batch_size=args.batch_size, isTrain=args.isTrain)
    ###################################################
    print("End:", datetime.datetime.now())
