from __future__ import print_function
# %load_ext autoreload
# %autoreload 2
import os
import argparse
import datetime
from six.moves import cPickle
from collections import defaultdict, OrderedDict
import numpy as np
import keras.backend as K
import keras
from keras.datasets import mnist
from keras.models import load_model

from information_plot_ibsgd import kde, simplebinmi
from information_plot_ibsgd.plots import Plots
import utils

# %matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
sns.set_style('darkgrid')

def computeMI(tst, measures, loss, PLOT_LAYERS, ARCH, MAX_EPOCHS):
    # Functions to return upper and lower bounds on entropy of layer activity
    noise_variance = 1e-1                    # Added Gaussian noise variance
    Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder
    entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
    entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

    # nats to bits conversion factor
    nats2bits = 1.0/np.log(2)

    # Save indexes of tests data for each of the output classes
    saved_labelixs = {}
    for i in range(2):
        saved_labelixs[i] = tst.y == i

    labelprobs = np.mean(tst.Y, axis=0)
    for activation in measures.keys():
        cur_dir = 'rawdata/' + activation + '_' + ARCH
        if not os.path.exists(cur_dir):
            print("Directory %s not found" % cur_dir)
            continue

        # Load files saved during each epoch, and compute MI measures of the activity in that epoch
        print('*** Doing %s ***' % cur_dir)
        for epochfile in sorted(os.listdir(cur_dir)):
            if not epochfile.startswith('epoch'):
                continue

            fname = cur_dir + "/" + epochfile
            with open(fname, 'rb') as f:
                d = cPickle.load(f)

            epoch = d['epoch']
            loss['d_loss'].append(d['d_loss'])
            loss['g_loss'].append(d['g_loss'])
            loss['d_loss_real'].append(d['d_loss_real'])
            loss['d_loss_fake'].append(d['d_loss_fake'])
            if epoch in measures[activation]: # Skip this epoch if its already been processed
                continue                      # this is a trick to allow us to rerun this cell multiple times)

            if epoch > MAX_EPOCHS:
                continue

            print("Doing", fname)

            num_layers = len(d['data']['activity_tst'])
            if PLOT_LAYERS is None:
                PLOT_LAYERS = []
                for lndx in range(num_layers):
                    PLOT_LAYERS.append(lndx)

            cepochdata = defaultdict(list)
            for lndx in range(num_layers):
                activity = d['data']['activity_tst'][lndx]
                # activity = np.reshape(activity,[-1,32*32*3]) #(93750, 3072)

                # Layer activity given input. This is simply the entropy of the Gaussian noise
                hM_given_X = kde.kde_condentropy(activity, noise_variance)


                """ Upper """
                # Compute marginal entropies
                h_upper = entropy_func_upper([activity,])[0]
                # Compute conditional entropies of layer activity given output
                hM_given_Y_upper=0.
                for i in range(2):
                    # print(i)
                    # print(len(activity)) #10000
                    # print(len(activity[0])) #1024
                    # print(len(saved_labelixs[0])) #10000
                    hcond_upper = entropy_func_upper([activity[saved_labelixs[i],:],])[0]
                    hM_given_Y_upper += labelprobs[i] * hcond_upper

                cepochdata['MI_XM_upper'].append( nats2bits * (h_upper - hM_given_X) )
                cepochdata['MI_YM_upper'].append( nats2bits * (h_upper - hM_given_Y_upper) )
                cepochdata['H_M_upper'  ].append( nats2bits * h_upper )
                pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])

                """ Lower """
                h_lower = entropy_func_lower([activity,])[0]
                hM_given_Y_lower=0.
                for i in range(2):
                    hcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
                    hM_given_Y_lower += labelprobs[i] * hcond_lower

                cepochdata['MI_XM_lower'].append( nats2bits * (h_lower - hM_given_X) )
                cepochdata['MI_YM_lower'].append( nats2bits * (h_lower - hM_given_Y_lower) )
                cepochdata['H_M_lower'  ].append( nats2bits * h_lower )
                pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

                """ Bin """
                # binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, 0.5)
                binxm,  binym= simplebinmi.bin_calc_information(tst.X, activity, saved_labelixs, num_of_bins=30)
                cepochdata['MI_XM_bin'].append( nats2bits * binxm )
                cepochdata['MI_YM_bin'].append( nats2bits * binym )
                pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])

                print('- Layer %d %s' % (lndx, pstr) )

            measures[activation][epoch] = cepochdata
    return measures, loss, PLOT_LAYERS

def show_sampled_image(data, save_plot_dir):
    class_names = ['0','1']
    img_x = np.reshape(data.X,[-1,28,28])
    img_y = data.y
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        i+=4985
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_x[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[img_y[i]])
    plt.show()
    fig_name = "sampled_real_fake_test.PNG"
    plt.savefig(os.path.join(save_plot_dir, fig_name))

def main():
    parser = argparse.ArgumentParser(description='Calculate Mutual Information GAN')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--activation', type=str, choices = ['tanh', 'relu', 'leakyrelu'], required = True,
                        help='options: tanh, relu, leakyrelu')
    parser.add_argument('--fake_size', type=int, default=5000,
                        help='proportion of fake images. Must be <=10000  (default:5000)')

    args = parser.parse_args()
    activ = args.activation

    save_plot_dir = os.path.join(os.getcwd(), "plots/mnist/", activ)
    if not os.path.exists(save_plot_dir):
        print("Making directory", save_plot_dir)
        os.makedirs(save_plot_dir)

    _, tst = utils.get_data(activ, False, args.fake_size)
    print("tst.X.shape:", tst.X.shape)
    show_sampled_image(tst, save_plot_dir)  # to verify the mixed dataset is correct

    MAX_EPOCHS = 30000      # Max number of epoch for which to compute mutual information measure

    # Directories from which to load saved layer activity
    # ARCH = '1024-20-20-20'
    ARCH = '1024-512-256-1'

    PLOT_LAYERS    = None     # Which layers to plot.  If None, all saved layers are plotted

    # Data structure used to store results
    measures = OrderedDict()
    measures[activ] = {}
    loss = defaultdict(list)

    """ Compute MI measures"""
    measures, loss, PLOT_LAYERS = computeMI(tst, measures, loss, PLOT_LAYERS, ARCH, MAX_EPOCHS)


    """Init Plot """
    plots = Plots(measures, loss, PLOT_LAYERS, save_plot_dir, ARCH)
    ### Plot Summary ###
    plots.plot_summary()
    ### Plot Discriminator & Generator Error ###
    plots.plot_error()
    ### Plot Comparison of loss and Mutual Information calculated via lower bound ###
    plots.plot_compare_error_MI()
    # """ Plot SNR graphs"""
    # plots.plot_snr()
    ### Plot Infoplane Visualization ###
    plots.plot_infoplane(MAX_EPOCHS, infoplane_measure='bin') #infoplane_measure= 'bin' or 'upper'


if __name__=='__main__':

    print("Start:", datetime.datetime.now())
    main()
    print("End:", datetime.datetime.now())
