from __future__ import print_function
# %load_ext autoreload
# %autoreload 2
import os
import datetime
# if not os.path.exists('/plots/'):
#   print("creating directory...")
#   os.mkdir('/plots/')

from six.moves import cPickle
from collections import defaultdict, OrderedDict, namedtuple
import numpy as np
import keras.backend as K
import keras
from keras.datasets import mnist
from keras.models import load_model

import kde
import simplebinmi

import utils

# %matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from notify_run import Notify

import seaborn as sns
sns.set_style('darkgrid')

def fake_datasets():
    fake_size = 5000
    generator = load_model("/cluster/home/it_stu150/lzc/IB_GAN_ZC/models/generator_"+ activ + '_' + optim + ".h5")
    # generator = load_model("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\屠老师实验室\\Codes\\IB_GAN_ZC\\models\\generator.h5")
    noise = np.random.normal(0, 1, (fake_size,100))
    gen_imgs = generator.predict(noise)

    return gen_imgs

def get_data():
    fake_size = 5000
    # Returns two namedtuples, with MNIST training and testing data
    #   trn.X is training data
    #   trn.y is trainiing class, with numbers from 0 to 9
    #   trn.Y is training class, but coded as a 10-dim vector with one entry set to 1
    # similarly for tst
    # nb_classes = 10
    (X_train, _), (X_test, _) = mnist.load_data(path="/cluster/home/it_stu150/lzc/IB_GAN_ZC/datasets/mnist.npz")
    # (X_train, _), (X_test, _) = mnist.load_data(path="C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\屠老师实验室\\Codes\\IB_GAN_ZC\\datasets\\mnist.npz")
    X_train = np.reshape(X_train,[-1,28*28*1])
    X_test = np.reshape(X_test,[-1,28*28*1])

    X_train = X_train / 127.5 - 1.
    X_test = X_test / 127.5 - 1.
    # batch_size = 32
    # valid = np.ones((batch_size,))

    # modified the test set for MI calculations
    gen_imgs = fake_datasets()
    fake_y = np.zeros((fake_size,)).astype(int)
    y_test = np.ones((fake_size,)).astype(int)
    fake_real_tstX = np.concatenate([X_test[:fake_size], gen_imgs], axis=0)
    fake_real_tsty = np.concatenate([y_test, fake_y], axis=0)

    y_train = np.ones((X_train.shape[0],1)).astype(int)
    # y_test = np.ones((X_test.shape[0],)).astype(int)

    Y_train = keras.utils.np_utils.to_categorical(y_train, 2)
    fake_real_tstY  = keras.utils.np_utils.to_categorical(fake_real_tsty,  2)

    Dataset = namedtuple('Dataset',['X','y','Y'])
    trn = Dataset(X_train, y_train, Y_train)
    tst = Dataset(fake_real_tstX, fake_real_tsty, fake_real_tstY)

    # del X_train, X_test, Y_train, Y_test, y_train, y_test
    del X_train, y_train, Y_train, y_test, fake_real_tstY

    return trn,tst

def show_sampled_image(data):
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
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[img_y[i]])
    plt.show()
    plt.savefig("plots/mnist/"+"sampled_real_fake_test_"+ activ + '_' + optim + ".png")

def plot_summary(measures, PLOT_LAYERS, ARCH, DO_LOWER, DO_BINNED):
    print("------------Plotting Summary-------------")
    #PLOT_LAYERS = [0,1,2,3,4] # [1,2,3]
    #PLOT_LAYERS = [0,1,2,3]
    #PLOT_LAYERS = [0,1,2,3]

    ## Start Plotting ##
    plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(5,2)
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue
        print()
        print("actndx: %d" % actndx)
        plt.subplot(gs[0,1]) #actndx = 1
        for lndx, layerid in enumerate(PLOT_LAYERS):
            xmvalsU = np.array([vals[epoch]['H_M_upper'][layerid] for epoch in epochs])
            plt.plot(epochs, xmvalsU, label='Layer %d'%layerid)
            #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
        plt.xscale('log')
        plt.yscale('log')
        # plt.title('tanh'+ ", UPPER")
        plt.title(activ+ ", UPPER")
        plt.ylabel('H(M)')
        plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

        # lower
        if DO_LOWER:
            plt.subplot(gs[0,0]) #actndx = 1
            for lndx, layerid in enumerate(PLOT_LAYERS):
                xmvalsL = np.array([vals[epoch]['H_M_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, xmvalsL, label='Layer %d'%layerid)
                #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.yscale('log')
            # plt.title('tanh'+ ", LOWER")
            plt.title(activ+ ", LOWER")
            plt.ylabel('H(M)')
    #         plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

        # upper
        plt.subplot(gs[1,1])
        for lndx, layerid in enumerate(PLOT_LAYERS):
            #for epoch in epochs:
            #    print('her',epoch, measures[activation][epoch]['MI_XM_upper'])
            xmvalsU = np.array([vals[epoch]['MI_XM_upper'][layerid] for epoch in epochs])
            plt.plot(epochs, xmvalsU, label='Layer %d'%layerid)
            #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
        plt.xscale('log')
        plt.ylabel('I(X;M)')
        plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

        # lower
        if DO_LOWER:
            plt.subplot(gs[1,0])
            for lndx, layerid in enumerate(PLOT_LAYERS):
                xmvalsL = np.array([vals[epoch]['MI_XM_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, xmvalsL, label='Layer %d'%layerid)
                #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.ylabel('I(X;M)')
    #         plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

        # upper
        plt.subplot(gs[2,1])
        for lndx, layerid in enumerate(PLOT_LAYERS):
            ymvalsU = np.array([vals[epoch]['MI_YM_upper'][layerid] for epoch in epochs])
            plt.plot(epochs, ymvalsU, label='Layer %d'%layerid)
        plt.xscale('log')
        plt.ylabel('I(Y;M)')
        plt.xlabel('Epoch')
        plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

        # lower
        if DO_LOWER:
            plt.subplot(gs[2,0])
            for lndx, layerid in enumerate(PLOT_LAYERS):
                ymvalsL = np.array([vals[epoch]['MI_YM_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, ymvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.ylabel('I(Y;M)')
            plt.xlabel('Epoch')
    #         plt.legend(loc='lower left', bbox_to_anchor=(1.1,, 0))

        # bin
        if DO_BINNED:
            plt.subplot(gs[3,:])
            for lndx, layerid in enumerate(PLOT_LAYERS):
                hbinnedvals = np.array([vals[epoch]['MI_XM_bin'][layerid] for epoch in epochs])
                plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
            plt.ylabel("I(X;M)")
            # plt.title('tanh'+ ", BINNED")
            plt.title(activ+ ", BINNED")
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            plt.subplot(gs[4,:])
            for lndx, layerid in enumerate(PLOT_LAYERS):
                hbinnedvals = np.array([vals[epoch]['MI_YM_bin'][layerid] for epoch in epochs])
                plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
            plt.xlabel('Epoch')
            plt.ylabel("I(Y;M)")
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

    #     if actndx == 0:
    #         plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('plots/mnist/' + 'summary_'+ activ +'_'+ optim + "_" + ARCH,bbox_inches='tight')
    print("------------Summary plot done!-------------")

def plot_error(measures, d_loss_list, d_loss_real_list, d_loss_fake_list, g_loss_list, ARCH):

    err_fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(2,1)
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        ### Plot 1, d_loss vs g_loss ###
        plt.subplot(gs[0,0])
        plt.plot(epochs, d_loss_list, label='d_loss')
        plt.plot(epochs, g_loss_list, label='g_loss')
        # plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.ylim([0.5, 1])
        plt.xscale('log')
        plt.legend(loc='upper right')
        plt.title(activ+ ", d_loss vs g_loss")

        ### Plot 2, d_loss_real vs d_loss_fake ###
        plt.subplot(gs[1,0])
        plt.plot(epochs, d_loss_real_list, label='d_loss_real')
        plt.plot(epochs, d_loss_fake_list, label='d_loss_fake')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.ylim([0.5, 1])
        plt.xscale('log')
        plt.legend(loc='upper right')
        plt.title(activ+ ", d_loss_real vs d_loss_fake")
    plt.tight_layout()
    plt.savefig('plots/mnist/' + 'dgloss_'+ activ +"_"+ optim + '_' +ARCH+ '.png')

def plot_compare_error_MI(measures, d_loss_list, g_loss_list, ARCH, PLOT_LAYERS):
    plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(3,1)
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue

        ### Plot MI(X;M) ###
        plt.subplot(gs[0,0])
        for lndx, layerid in enumerate(PLOT_LAYERS):
            xmvalsL = np.array([vals[epoch]['MI_XM_lower'][layerid] for epoch in epochs])
            plt.plot(epochs, xmvalsL, label='Layer %d'%layerid)
            #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
        plt.title(activ+ ", LOWER")
        plt.xscale('log')
        plt.ylabel('I(X;M)')
        # plt.legend(loc='upper right')

        ### Plot MI(Y;M) ###
        plt.subplot(gs[1,0])
        for lndx, layerid in enumerate(PLOT_LAYERS):
            ymvalsL = np.array([vals[epoch]['MI_YM_lower'][layerid] for epoch in epochs])
            plt.plot(epochs, ymvalsL, label='Layer %d'%layerid)
        plt.xscale('log')
        plt.ylabel('I(Y;M)')
        # plt.legend(loc='upper right')
        # plt.xlabel('Epoch')

        ### Plot d_loss vs g_loss ###
        plt.subplot(gs[2,0])
        plt.plot(epochs, d_loss_list, label='d_loss')
        plt.plot(epochs, g_loss_list, label='g_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.ylim([0.5, 1])
        plt.xscale('log')
        plt.legend(loc='upper right')
        plt.title(activ+ ", d_loss vs g_loss")

    plt.tight_layout()
    plt.savefig('plots/mnist/' + 'LOSSvsMI_'+ activ +'_' + optim + "_" + ARCH + '.png')

def plot_snr(measures, PLOT_LAYERS, ARCH):
    plt.figure(figsize=(12,5))

    gs = gridspec.GridSpec(len(measures), len(PLOT_LAYERS))
    # saved_data = {}
    for actndx, activation in enumerate(measures.keys()):
        # cur_dir = 'rawdata/' + DIR_TEMPLATE % activation
        cur_dir = 'rawdata/' + activ +'_'+ARCH + '_' + optim
        if not os.path.exists(cur_dir):
            continue

        epochs = []
        means = []
        stds = []
        wnorms = []
        # trnloss = []
        # tstloss = []
    #     print(cur_dir) #rawdata/tanh_1024-20-20-20
        for epochfile in sorted(os.listdir(cur_dir)):
            if not epochfile.startswith('epoch'):
                continue

            with open(cur_dir + "/"+epochfile, 'rb') as f:
                try:
                    d = cPickle.load(f)
                except:
                    print('Error loading ', epochfile)
                    continue
            epoch = d['epoch']
            epochs.append(epoch)
            wnorms.append(d['data']['weights_norm'])
            means.append(d['data']['gradmean'])
            stds.append(d['data']['gradstd'])
            # trnloss.append(d['loss']['trn'])
            # tstloss.append(d['loss']['tst'])
    #     print(len(trnloss)) ##144
        # wnorms, means, stds, trnloss, tstloss = map(np.array, [wnorms, means, stds, trnloss, tstloss])
        wnorms, means, stds = map(np.array, [wnorms, means, stds])
        # saved_data[activ] = {'epochs':epochs, 'wnorms':wnorms, 'means': means, 'stds': stds}


        for lndx,layerid in enumerate(PLOT_LAYERS):
            plt.subplot(gs[actndx, lndx])
            plt.plot(epochs, means[:,layerid], 'b', label="Mean")
            plt.plot(epochs, stds[:,layerid], 'orange', label="Std")
            plt.plot(epochs, means[:,layerid]/stds[:,layerid], 'red', label="SNR")
            plt.plot(epochs, wnorms[:,layerid], 'g', label="||W||")

            plt.title('%s - Layer %d'%(activ, layerid))
            plt.xlabel('Epoch')
            plt.gca().set_xscale("log", nonposx='clip')
            plt.gca().set_yscale("log", nonposy='clip')


    plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.2))
    plt.tight_layout()
    plt.savefig('plots/mnist/' +  'snr_'+ activ +'_' + optim +'_' + ARCH, bbox_inches='tight')


# def plot_compare_error_MI2(measures, d_loss_list, g_loss_list, ARCH, PLOT_LAYERS):
#     plt.figure(figsize=(8,8))
#     gs = gridspec.GridSpec(2,1)
#     for actndx, (activation, vals) in enumerate(measures.items()):
#         epochs = sorted(vals.keys())
#         if not len(epochs):
#             continue
#
#         ### Plot MI(X;M) ###
#         plt.subplot(gs[0,0])
#         for lndx, layerid in enumerate(PLOT_LAYERS):
#             xmvalsL = np.array([vals[epoch]['MI_XM_lower'][layerid] for epoch in epochs])
#             plt.plot(epochs, xmvalsL, label='Layer %d'%layerid)
#             #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
#         plt.plot(epochs, d_loss_list, label='d_loss')
#         plt.plot(epochs, g_loss_list, label='g_loss')
#         plt.xscale('log')
#         plt.ylabel('I(X;M)')
#         plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))
#
#         ### Plot MI(Y;M) ###
#         plt.subplot(gs[1,0])
#         for lndx, layerid in enumerate(PLOT_LAYERS):
#             ymvalsL = np.array([vals[epoch]['MI_YM_lower'][layerid] for epoch in epochs])
#             plt.plot(epochs, ymvalsL, label='Layer %d'%layerid)
#         plt.plot(epochs, d_loss_list, label='d_loss')
#         plt.plot(epochs, g_loss_list, label='g_loss')
#         plt.xscale('log')
#         plt.ylabel('I(Y;M)')
#         plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))
#         plt.xlabel('Epoch')
#
#
#     plt.tight_layout()
#     plt.savefig('plots/mnist/' + 'overwrite_'+ activ +"_"+ARCH+ '.png')


def main():
    trn,tst = get_data()
    show_sampled_image(tst)  # to verify the mixed dataset is correct


    # Which measure to plot
    # infoplane_measure = 'upper'
    infoplane_measure = 'bin'

    DO_SAVE        = True    # Whether to save plots or just show them
    DO_LOWER       = True    # (infoplane_measure == 'lower')   # Whether to compute lower bounds also
    DO_BINNED      = True    #(infoplane_measure == 'bin')     # Whether to compute MI estimates based on binning

    MAX_EPOCHS = 30000      # Max number of epoch for which to compute mutual information measure
    # MAX_EPOCHS = 1000
    COLORBAR_MAX_EPOCHS = 30000

    # Directories from which to load saved layer activity
    # ARCH = '1024-20-20-20'
    ARCH = '1024-512-256-1'
    #ARCH = '20-20-20-20-20-20'
    #ARCH = '32-28-24-20-16-12'
    #ARCH = '32-28-24-20-16-12-8-8'
    DIR_TEMPLATE = '%%s_%s'%ARCH
    # print(DIR_TEMPLATE) output: %s_1024-20-20-20

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

    PLOT_LAYERS    = None     # Which layers to plot.  If None, all saved layers are plotted

    # Data structure used to store results
    measures = OrderedDict()
    # measures['relu'] = {}
    # measures['tanh'] = {}
    measures[activ] = {}

    """# Compute MI measures"""
    d_loss_list = []
    g_loss_list = []
    d_loss_real_list = []
    d_loss_fake_list = []
    for activation in measures.keys():
        cur_dir = 'rawdata/' + activ + '_' + ARCH +'_'+ optim
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
            d_loss_list.append(d['d_loss'])
            g_loss_list.append(d['g_loss'])
            d_loss_real_list.append(d['d_loss_real'])
            d_loss_fake_list.append(d['d_loss_fake'])
            if epoch in measures[activation]: # Skip this epoch if its already been processed
                continue                      # this is a trick to allow us to rerun this cell multiple times)

            if epoch > MAX_EPOCHS:
                continue

            print("Doing", fname)

            num_layers = len(d['data']['activity_tst'])

            if PLOT_LAYERS is None:
                PLOT_LAYERS = []
                for lndx in range(num_layers):
                    #if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
                    PLOT_LAYERS.append(lndx)

            cepochdata = defaultdict(list)
            for lndx in range(num_layers):
                activity = d['data']['activity_tst'][lndx]
                # activity = np.reshape(activity,[-1,32*32*3]) #(93750, 3072)
                # print(activity.shape)
                # Compute marginal entropies
                h_upper = entropy_func_upper([activity,])[0]
                if DO_LOWER:
                    h_lower = entropy_func_lower([activity,])[0]

                # Layer activity given input. This is simply the entropy of the Gaussian noise
                hM_given_X = kde.kde_condentropy(activity, noise_variance)

                # Compute conditional entropies of layer activity given output
                hM_given_Y_upper=0.
                for i in range(1):
                    # print(i)
                    # print(len(activity)) #10000
                    # print(len(activity[0])) #1024
                    # print(len(saved_labelixs[0])) #10000
                    hcond_upper = entropy_func_upper([activity[saved_labelixs[i],:],])[0]
                    hM_given_Y_upper += labelprobs[i] * hcond_upper

                if DO_LOWER:
                    hM_given_Y_lower=0.
                    for i in range(2):
                        hcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
                        hM_given_Y_lower += labelprobs[i] * hcond_lower

                cepochdata['MI_XM_upper'].append( nats2bits * (h_upper - hM_given_X) )
                cepochdata['MI_YM_upper'].append( nats2bits * (h_upper - hM_given_Y_upper) )
                cepochdata['H_M_upper'  ].append( nats2bits * h_upper )

                pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
                if DO_LOWER:  # Compute lower bounds
                    cepochdata['MI_XM_lower'].append( nats2bits * (h_lower - hM_given_X) )
                    cepochdata['MI_YM_lower'].append( nats2bits * (h_lower - hM_given_Y_lower) )
                    cepochdata['H_M_lower'  ].append( nats2bits * h_lower )
                    pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

                if DO_BINNED: # Compute binner estimates
                    # binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, 0.5)
                    binxm,  binym= simplebinmi.bin_calc_information(tst.X, activity, 10, saved_labelixs)
                    cepochdata['MI_XM_bin'].append( nats2bits * binxm )
                    cepochdata['MI_YM_bin'].append( nats2bits * binym )
                    pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])

                print('- Layer %d %s' % (lndx, pstr) )

            measures[activation][epoch] = cepochdata

    """ Plot Summary"""
    plot_summary(measures, PLOT_LAYERS, ARCH, DO_LOWER, DO_BINNED)
    """ Plot Discriminator & Generator Error"""
    plot_error(measures, d_loss_list, d_loss_real_list, d_loss_fake_list, g_loss_list, ARCH)
    """ Plot Comparison of loss and Mutual Information calculated via lower bound"""
    plot_compare_error_MI(measures, d_loss_list, g_loss_list, ARCH, PLOT_LAYERS)
    # plot_compare_error_MI2(measures, d_loss_list, g_loss_list, ARCH, PLOT_LAYERS)

    # """ Plot SNR graphs"""
    # plot_snr(measures, PLOT_LAYERS, ARCH)

    """ Plot Infoplane Visualization"""

    max_epoch = max( (max(vals.keys()) if len(vals) else 0) for vals in measures.values())
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
    sm._A = []

    fig=plt.figure(figsize=(10,5))
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue
        plt.subplot(1,2,actndx+1)
        for epoch in epochs:
            c = sm.to_rgba(epoch)
            xmvals = np.array(vals[epoch]['MI_XM_'+infoplane_measure])[PLOT_LAYERS]
            ymvals = np.array(vals[epoch]['MI_YM_'+infoplane_measure])[PLOT_LAYERS]

            plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
            plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

        # plt.ylim([0, 240])
        plt.ylim([0, 1.5])
        plt.xlim([0, 14])
        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        # plt.title('tanh, MI of discriminator')
        plt.title(activ+', MI of discriminator')

    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()

    if DO_SAVE:
        # plt.savefig('plots/mnist/' + 'tanh' + 'discriminator'+ARCH,bbox_inches='tight')
        plt.savefig('plots/mnist/' + activ+"_"+optim+"_" + 'discriminator'+ARCH,bbox_inches='tight')

if __name__=='__main__':
    # notify = Notify()
    # try:
    #     notify.send('Start plotting ibsgd method!')
    # except:
    #     print("Noti failed")
    # notify.send('Start plotting ibsgd method!')
    print("Start:", datetime.datetime.now())
    ###################################################
    activ = 'tanh'
    # activ = 'leakyrelu'
    # activ = 'relu'
    # optim = 'adam'
    optim = 'sgd'
    main()
    ##################################################
    print("End:", datetime.datetime.now())
    # notify.send('Finish plotting ibsgd method!')
    # try:
    #     notify.send('Finish plotting ibsgd method!')
    # except:
    #     print("Noti failed")
