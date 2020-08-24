import os
import numpy as np
from six.moves import cPickle
# %matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
sns.set_style('darkgrid')

class Plots():
    def __init__(self, measures, loss, PLOT_LAYERS, save_plot_dir, ARCH):
        self.measures = measures
        self.loss = loss
        self.PLOT_LAYERS = PLOT_LAYERS
        self.ARCH = ARCH
        self.save_plot_dir = save_plot_dir

    def plot_summary(self):

        plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(5,2)
        for actndx, (activation, vals) in enumerate(self.measures.items()):
            epochs = sorted(vals.keys())
            if not len(epochs):
                continue
            plt.subplot(gs[0,1])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                xmvalsU = np.array([vals[epoch]['H_M_upper'][layerid] for epoch in epochs])
                plt.plot(epochs, xmvalsU, label='Layer %d'%layerid)
                #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.yscale('log')
            # plt.title('tanh'+ ", UPPER")
            plt.title(activation+ ", UPPER")
            plt.ylabel('H(M)')
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            # lower
            plt.subplot(gs[0,0]) #actndx = 1
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                xmvalsL = np.array([vals[epoch]['H_M_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, xmvalsL, label='Layer %d'%layerid)
                #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.yscale('log')
            # plt.title('tanh'+ ", LOWER")
            plt.title(activation+ ", LOWER")
            plt.ylabel('H(M)')
    #         plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            # upper
            plt.subplot(gs[1,1])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                xmvalsU = np.array([vals[epoch]['MI_XM_upper'][layerid] for epoch in epochs])
                plt.plot(epochs, xmvalsU, label='Layer %d'%layerid)
                #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.ylabel('I(X;M)')
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            # lower
            plt.subplot(gs[1,0])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                xmvalsL = np.array([vals[epoch]['MI_XM_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, xmvalsL, label='Layer %d'%layerid)
                #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.ylabel('I(X;M)')
    #         plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            # upper
            plt.subplot(gs[2,1])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                ymvalsU = np.array([vals[epoch]['MI_YM_upper'][layerid] for epoch in epochs])
                plt.plot(epochs, ymvalsU, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.ylabel('I(Y;M)')
            plt.xlabel('Epoch')
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            # lower
            plt.subplot(gs[2,0])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                ymvalsL = np.array([vals[epoch]['MI_YM_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, ymvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.ylabel('I(Y;M)')
            plt.xlabel('Epoch')
    #         plt.legend(loc='lower left', bbox_to_anchor=(1.1,, 0))

            # bin
            plt.subplot(gs[3,:])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                hbinnedvals = np.array([vals[epoch]['MI_XM_bin'][layerid] for epoch in epochs])
                plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
            plt.ylabel("I(X;M)")
            plt.title(activation+ ", BINNED")
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            plt.subplot(gs[4,:])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                hbinnedvals = np.array([vals[epoch]['MI_YM_bin'][layerid] for epoch in epochs])
                plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
            plt.xlabel('Epoch')
            plt.ylabel("I(Y;M)")
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0))

            plt.tight_layout()
            fig_name = 'summary_' + self.ARCH + ".PNG"
            plt.savefig(os.path.join(self.save_plot_dir, fig_name),bbox_inches='tight')

    def plot_error(self):

        err_fig = plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(2,1)
        for actndx, (activation, vals) in enumerate(self.measures.items()):
            epochs = sorted(vals.keys())
            ### Plot 1, d_loss vs g_loss ###
            plt.subplot(gs[0,0])
            plt.plot(epochs, self.loss['d_loss'], label='d_loss')
            plt.plot(epochs, self.loss['g_loss'], label='g_loss')
            # plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # plt.ylim([0.5, 1])
            plt.xscale('log')
            plt.legend(loc='upper right')
            plt.title(activation+ ", d_loss vs g_loss")

            ### Plot 2, d_loss_real vs d_loss_fake ###
            plt.subplot(gs[1,0])
            plt.plot(epochs, self.loss['d_loss_real'], label='d_loss_real')
            plt.plot(epochs, self.loss['d_loss_fake'], label='d_loss_fake')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # plt.ylim([0.5, 1])
            plt.xscale('log')
            plt.legend(loc='upper right')
            plt.title(activation+ ", d_loss_real vs d_loss_fake")
            plt.tight_layout()
            fig_name = 'dgloss_' +self.ARCH+ '.png'
            plt.savefig(os.path.join(self.save_plot_dir, fig_name),bbox_inches='tight')

    def plot_compare_error_MI(self):
        plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(3,1)
        for actndx, (activation, vals) in enumerate(self.measures.items()):
            epochs = sorted(vals.keys())
            if not len(epochs):
                continue

            ### Plot MI(X;M) ###
            plt.subplot(gs[0,0])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                xmvalsL = np.array([vals[epoch]['MI_XM_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, xmvalsL, label='Layer %d'%layerid)
                #plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
            plt.title(activation+ ", LOWER")
            plt.xscale('log')
            plt.ylabel('I(X;M)')
            # plt.legend(loc='upper right')

            ### Plot MI(Y;M) ###
            plt.subplot(gs[1,0])
            for lndx, layerid in enumerate(self.PLOT_LAYERS):
                ymvalsL = np.array([vals[epoch]['MI_YM_lower'][layerid] for epoch in epochs])
                plt.plot(epochs, ymvalsL, label='Layer %d'%layerid)
            plt.xscale('log')
            plt.ylabel('I(Y;M)')
            # plt.legend(loc='upper right')
            # plt.xlabel('Epoch')

            ### Plot d_loss vs g_loss ###
            plt.subplot(gs[2,0])
            plt.plot(epochs, self.loss['d_loss'], label='d_loss')
            plt.plot(epochs, self.loss['g_loss'], label='g_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # plt.ylim([0.5, 1])
            plt.xscale('log')
            plt.legend(loc='upper right')
            plt.title(activation+ ", d_loss vs g_loss")

            plt.tight_layout()
            fig_name = 'LOSSvsMI_' + self.ARCH + '.png'
            plt.savefig(os.path.join(self.save_plot_dir, fig_name),bbox_inches='tight')

    def plot_snr(self):
        plt.figure(figsize=(12,5))

        gs = gridspec.GridSpec(len(self.measures), len(self.PLOT_LAYERS))
        for actndx, activation in enumerate(self.measures.keys()):
            cur_dir = 'rawdata/' + activation +'_'+self.ARCH
            if not os.path.exists(cur_dir):
                continue

            epochs = []
            means = []
            stds = []
            wnorms = []
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

            wnorms, means, stds = map(np.array, [wnorms, means, stds])

            for lndx,layerid in enumerate(self.PLOT_LAYERS):
                plt.subplot(gs[actndx, lndx])
                plt.plot(epochs, means[:,layerid], 'b', label="Mean")
                plt.plot(epochs, stds[:,layerid], 'orange', label="Std")
                plt.plot(epochs, means[:,layerid]/stds[:,layerid], 'red', label="SNR")
                plt.plot(epochs, wnorms[:,layerid], 'g', label="||W||")

                plt.title('%s - Layer %d'%(activation, layerid))
                plt.xlabel('Epoch')
                plt.gca().set_xscale("log", nonposx='clip')
                plt.gca().set_yscale("log", nonposy='clip')


            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.2))
            plt.tight_layout()
            fig_name = 'snr_' + self.ARCH + ".png"
            plt.savefig(os.path.join(self.save_plot_dir, fig_name),bbox_inches='tight')

    def plot_infoplane(self, MAX_EPOCHS, infoplane_measure='bin'):
        max_epoch = max( (max(vals.keys()) if len(vals) else 0) for vals in self.measures.values())
        sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=MAX_EPOCHS))
        sm._A = []

        fig=plt.figure(figsize=(6,5))
        for actndx, (activation, vals) in enumerate(self.measures.items()):
            epochs = sorted(vals.keys())
            if not len(epochs):
                continue
            # plt.subplot(1,2,actndx+1)
            for epoch in epochs:
                c = sm.to_rgba(epoch)
                xmvals = np.array(vals[epoch]['MI_XM_'+infoplane_measure])[self.PLOT_LAYERS]
                ymvals = np.array(vals[epoch]['MI_YM_'+infoplane_measure])[self.PLOT_LAYERS]

                plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
                plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in self.PLOT_LAYERS], edgecolor='none', zorder=2)

            # plt.ylim([0, 240])
            plt.ylim([0, 1.5])
            plt.xlim([0, 14])
            plt.xlabel('I(X;M)')
            plt.ylabel('I(Y;M)')
            plt.title(activation+', MI of discriminator')

            cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
            plt.colorbar(sm, label='Epoch', cax=cbaxes)
            plt.tight_layout()

            fig_name = "infoplane_discriminator_" + self.ARCH + ".png"
            plt.savefig(os.path.join(self.save_plot_dir, fig_name),bbox_inches='tight')
