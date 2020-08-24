from __future__ import print_function
import keras
import keras.backend as K
import numpy as np

from six.moves import cPickle
import os

import utils

class LoggingReporter():
    def __init__(self, trn, tst, rawdata_dir, do_save_func=None):
        self.trn = trn  # Train data
        self.tst = tst  # Test data
        self.do_save_func = do_save_func
        self.rawdata_dir = rawdata_dir

    def on_train_begin(self, model):
        # Indexes of the layers which we keep track of. Basically, this will be any layer
        # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
        self.layerixs = []

        # Functions return activity of each layer
        self.layerfuncs = []

        # Functions return weights of each layer
        self.layerweights = []
        for lndx, l in enumerate(model.layers):
            if hasattr(l, 'kernel'):
                self.layerixs.append(lndx)
                self.layerfuncs.append(K.function(model.inputs, [l.output,]))
                self.layerweights.append(l.kernel)

        inputs = [model._feed_inputs,
                  model._feed_targets,
                  model._feed_sample_weights,
                  K.learning_phase()]

        # Get gradients of all the relevant layers at once
        grads = model.optimizer.get_gradients(model.total_loss, self.layerweights)
        self.get_gradients = K.function(inputs=inputs, outputs=grads)

        # Get cross-entropy loss
        self.get_loss = K.function(inputs=inputs, outputs=[model.total_loss,])

    def on_epoch_begin(self, model, epoch):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            self._log_gradients = False
        else:
            # We will log this epoch.  For each batch in this epoch, we will save the gradients (in on_batch_begin)
            # We will then compute means and vars of these gradients

            self._log_gradients = True
            self._batch_weightnorm = []

            self._batch_gradients = [ [] for _ in model.layers[1:] ] #model.layers[1:] to skip Input layer

            # Indexes of all the training data samples. These are shuffled and read-in in chunks of SGD_BATCHSIZE
            ixs = list(range(len(self.trn.X)))
            np.random.shuffle(ixs)
            self._batch_todo_ixs = ixs

    def on_batch_begin(self, model, batch_size):
        if not self._log_gradients:
            # We are not keeping track of batch gradients, so do nothing
            return

        # Sample a batch
        batchsize = batch_size
        cur_ixs = self._batch_todo_ixs[:batchsize]
        # Advance the indexing, so next on_batch_begin samples a different batch
        self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]

        # Get gradients for this batch

        x, y, weights = model._standardize_user_data(self.trn.X[cur_ixs,:], self.trn.y[cur_ixs])
        inputs = [x, y, weights, 1] # 1 indicates training phase
        for lndx, g in enumerate(self.get_gradients(inputs)):
            # g is gradients for weights of lndx's layer
            oneDgrad = np.reshape(g, [-1, 1])                  # Flatten to one dimensional vector
            # print(len(self._batch_gradients)) #len=3
            # print("Indx:%d" % lndx) #indx=0,1,2,3
            # print(g.shape) #indx0:(784,1024), indx1:(1024,512), indx2:(512,256),
            # print(oneDgrad.shape) #indx0:(802816,1), indx1:(524288,1), indx2:(131072,1)
            self._batch_gradients[lndx].append(oneDgrad)


    def on_epoch_end(self, model, d_loss_real, d_loss_fake, g_loss, epoch):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            return
        d_loss = 0.5*(d_loss_real+d_loss_fake)
        # Get overall performance
        data = {
            'weights_norm' : [],   # L2 norm of weights
            'gradmean'     : [],   # Mean of gradients
            'gradstd'      : [],   # Std of gradients
            'activity_tst' : []    # Activity in each layer for test set
        }

        for lndx, layerix in enumerate(self.layerixs):
            clayer = model.layers[layerix]

            data['weights_norm'].append( np.linalg.norm(K.get_value(clayer.kernel)) )

            stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
            data['gradmean'    ].append( np.linalg.norm(stackedgrads.mean(axis=1)) )
            data['gradstd'     ].append( np.linalg.norm(stackedgrads.std(axis=1)) )
            data['activity_tst'].append(self.layerfuncs[lndx]([self.tst.X,])[0])

        fname = self.rawdata_dir + "/epoch%08d"% epoch
        print("Saving", fname)
        with open(fname, 'wb') as f:
            cPickle.dump({'epoch':epoch, 'data':data,
            'd_loss':d_loss, 'd_loss_real':d_loss_real, 'd_loss_fake':d_loss_fake ,'g_loss':g_loss}, f, cPickle.HIGHEST_PROTOCOL)
