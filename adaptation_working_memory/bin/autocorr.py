import itertools

import numpy as np
import numpy.random as rd


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def autocorr_plot_neurons(data_path, FLAGS, n_start, n_neurons, data_file=None, plot=True,
                          max_neurons=-1, sample_starts=[200, 1600, 2600], label='TEST', dirlabel=''):
    """
    Estimating "intrinsic timescales" of neurons as described in Methods of [1].
    [1] Wasmuht, D.F., Spaak, E., Buschman, T.J., Miller, E.K. and Stokes, M.G., 2018.
    Intrinsic neuronal dynamics predict distinct functional roles during working memory.
    Nature communications, 9(1), p.3499.
    :param data_path:
    :param FLAGS:
    :param n_start: starting index of neurons to consider (use to skip non-adaptive neurons)
    :param n_neurons: number of neurons to evaluate
    :param data_file:
    :param plot:
    :param max_neurons:
    :param sample_start: starting time-step indexes to be considered for analysis (following 500ms used)
    :param label:
    :return:
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr
    import pickle
    import json
    import math
    import os

    # create directory to store the results
    dirname = 'autocorr' + dirlabel
    if not os.path.exists(os.path.join(data_path, dirname)):
        os.makedirs(os.path.join(data_path, dirname))

    plot_data = 'plot_trajectory_data.pickle' if data_file is None else data_file
    data = pickle.load(open(os.path.join(data_path, plot_data), 'rb'))
    bin_size = 50  # described in [1]
    sample_size = 500  # described in [1]
    if sample_starts is None:
        sample_starts = [r for r in range(0, data['z'].shape[1] - sample_size, 50)]
    assert sample_size % bin_size == 0
    n_bins = int(sample_size / bin_size)  # == 10
    spikes = data['z'][:, :, n_start:n_start+n_neurons]  # batch x time x neurons
    inferred_taus = []
    inferred_As = []
    inferred_Bs = []
    n_idxs = []
    no_spike_count = 0
    no_corr_count = 0
    large_tau_count = 0
    max_neurons = max_neurons if max_neurons > 0 else n_neurons
    for n_idx in range(min(n_neurons, max_neurons)):  # loop over adaptive neurons
        corrs_neuron = []
        for sample_start in sample_starts:
            spk_count = spikes[:, sample_start:sample_start+sample_size, n_idx]
            bin_spk_count = bin_ndarray(spk_count, (spikes.shape[0], n_bins), operation='sum')  # batch x n_bins

            if np.count_nonzero(spk_count) == 0:  # FIXME: check if this is correct
                # print('skipping dead neuron', n_start+n_idx, "mean of spikes over batches = ", np.mean(spk_count))
                no_spike_count += 1
                continue  # skip dead neurons
            # calculate correlations across bins
            lags_idxs = [l for l in range(1, n_bins)]  # idy for [l for l in range(50, 500, 50)]
            corrs_neuron = []  # black dots
            for corrs_idx, lag_idx in enumerate(lags_idxs):
                corr_per_lag = []
                for i in range(n_bins - lag_idx):
                    pearson_corr_coeff, _ = pearsonr(bin_spk_count[:, i], bin_spk_count[:, i + lag_idx])
                    # print(i, i+lag_idx, pearson_corr_coeff)
                    if math.isnan(pearson_corr_coeff) or math.isinf(pearson_corr_coeff):
                        continue
                    corr_per_lag.append(pearson_corr_coeff)  # add black dot
                if len(corrs_neuron) <= corrs_idx:
                    corrs_neuron.append(corr_per_lag)
                else:
                    corrs_neuron[corrs_idx] = corrs_neuron[corrs_idx] + corr_per_lag

        if len(corrs_neuron) == 0:
            continue
        avg_corrs_neuron = [np.mean(c) for c in corrs_neuron]  # red dots
        if np.array([math.isnan(c) for c in avg_corrs_neuron]).any():
            no_corr_count += 1
            # print('skipping neuron with nan correlation')  # this means at some lag there were no black dots
            continue  # skip neurons with nan correlation
        # print("---------- neuron ", n_idx)
        # for l in range(len(avg_corrs_neuron)):
        #     print(l*bin_size, avg_corrs_neuron[l])  # print red dots

        # fit curve to red dots
        def func(x, A, B, tau):
            return A * (np.exp(-x / tau) + B)
        xdata = np.arange(len(avg_corrs_neuron))
        try:
            popt, pcov = curve_fit(func, xdata, avg_corrs_neuron)
            A, B, tau = tuple(popt)
            tau = tau * bin_size  # convert from bins to ms
            # if 'tauas' in FLAGS.__dict__.keys():
            #     if type(FLAGS.tauas) is list and tau > max(FLAGS.tauas):
            #         continue
            inferred_As.append(A)
            inferred_Bs.append(B)
            inferred_taus.append(tau)
            n_idxs.append(n_start+n_idx)

            if plot:
                plt.cla()
                plt.clf()
                for l in range(len(corrs_neuron)):
                    for i in range(len(corrs_neuron[l])):
                        plt.plot(xdata[l], corrs_neuron[l][i], 'ko', markersize=2)  # grid line at zero
                plt.plot(xdata, np.zeros_like(xdata), 'k--', alpha=0.2)  # grid line at zero
                plt.plot(xdata, avg_corrs_neuron, 'ro', markersize=4)  # avg red dots
                plt.plot(xdata, func(xdata, *popt), 'k-', label='fitted curve')
                plt.title("Fitted exponential curve tau = {:.0f}ms".format(tau))
                plt.xticks(xdata, [str(50 * (i+1)) for i in xdata])
                plt.xlabel("time lag (ms)")
                plt.ylabel("autocorrelation")
                plt.tight_layout()
                # plt.show()
                plt_path = os.path.join(data_path, dirname, label + 'autocorr_{}_tau{:.0f}.pdf'.format(n_idx, tau))
                plt.savefig(plt_path, format='pdf')
        except RuntimeError as e:
            print("Skipping")
            print(e)

    print("SKIPPED: {} dead neurons (no spikes); {} neurons with nan correlation; {} neurons with too long tau"
          .format(no_spike_count, no_corr_count, large_tau_count))
    if len(n_idxs) == 0:
        print("FAIL: no neurons survived the filtering")
        return
    else:
        print("Number of OK neurons processed =", len(n_idxs))
    resuls = {
        'tau': inferred_taus,
        'A': inferred_As,
        'B': inferred_Bs,
        'neuron_idx': n_idxs,
    }

    tmp = np.array(inferred_taus)
    print("num inferred taus total = ",tmp.shape)
    print("inferred taus = ",tmp)
    print("num inferred taus < 10 = ",np.where(tmp < 10))
    if plot:
        # Plot histogram of intrinsic timescales
        # hist, bins, _ = plt.hist(inferred_taus, bins=20, normed=True)
        plt.cla()
        plt.clf()
        # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        logbins = np.logspace(np.log10(10), np.log10(1000), 25)
        hist, bins, _ = plt.hist(inferred_taus, bins=logbins, normed=False, facecolor='green', alpha=0.5,
                                 edgecolor='black', linewidth=0.5)
        plt.xscale("log")  # , nonposx='clip')
        plt.xlim([10, 10 ** 3])
        locs, labels = plt.yticks()
        plt.yticks(locs, ["{:.1f}".format(100 * i/len(inferred_taus)) for i in locs])
        # plt.yticks([i for i in range(0, int(len(inferred_taus) * 0.16), int(len(inferred_taus) * 0.04))], ['0', '4', '8', '12'])
        plt.ylabel("percentage of cells")
        # plt.ylabel("num. of cells")
        plt.xlabel("intrinsic time constant (ms)")
        plt_path = os.path.join(data_path, dirname, label + 'histogram.pdf')
        plt.savefig(plt_path, format='pdf')

    try:  # if all tau_a-s are stored in flags, we can relate them to the intrinsic time constants (inferred_taus)
        resuls['taua'] = FLAGS.tauas
        if type(FLAGS.tauas) == int:
            defined_tauas = np.array([0 for _ in range(FLAGS.n_regular)] +
                                     [FLAGS.tauas for _ in range(FLAGS.n_adaptive)])
        else:
            defined_tauas = np.array(FLAGS.tauas)
        defined_tauas = defined_tauas[n_idxs]  # take only entries for the neurons that have inferred tau
        if plot and len(n_idxs) > 5:
            print("All tau_a-s available in FLAGS; going to plot relation to intrinsic timescales")
            plt.cla()
            plt.clf()
            plt.plot(defined_tauas, inferred_taus, 'ko')
            plt.xlabel('(defined) adaptation time constant')
            plt.ylabel('(inferred) intrinsic time constant')
            # plt.show()
            plt_path = os.path.join(data_path, dirname, label + 'tau_comp.pdf')
            plt.savefig(plt_path, format='pdf')
    except AttributeError:
        pass  # skip if the recorded data does not contain the tau_a-s
    with open(os.path.join(data_path, dirname, label + 'results.json'), 'w') as fp:
        json.dump(resuls, fp, indent=4)


def autocorr_plot(data_path, data_file=None, plot=True, max_neurons=20, sample_starts=[0]):
    import json
    import os
    if os.path.exists(os.path.join(data_path, 'flags.json')):
        flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    else:
        flags_dict = json.load(open(os.path.join(data_path, 'flag.json')))

    from types import SimpleNamespace
    FLAGS = SimpleNamespace(**flags_dict)
    autocorr_plot_neurons(data_path, FLAGS, n_start=0, n_neurons=FLAGS.n_regular, data_file=data_file,
                          max_neurons=max_neurons, sample_starts=sample_starts, label='R_')
    autocorr_plot_neurons(data_path, FLAGS, n_start=FLAGS.n_regular, n_neurons=FLAGS.n_adaptive, data_file=data_file,
                          max_neurons=max_neurons, sample_starts=sample_starts, label='A_')


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description='Calculate autocorrelation and plot à la Stokes 2018.')
    parser.add_argument('path', help='Path to directory that contains flags and plot data.')
    parser.add_argument('--plot', help='Filename of pickle file containing data for plotting.',
                        default='plot_trajectory_data.pickle')
    parser.add_argument("--meanhist", help="string to match when doing mean histogram over many runs")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("Given path (" + args.path + ") does not exist")
        exit(1)
    if not os.path.isdir(args.path):
        print("Given path (" + args.path + ") is not a directory")
        exit(1)

    if args.meanhist:
        # n_adaptive = 0
        # n_regular = 0
        taus_per_sim = []
        for subdir, dirs, files in os.walk(args.path):
            if args.meanhist not in subdir:
                continue
            print(subdir)
            for f in files:
                # if "flags.json" in f:
                #     fpath = os.path.join(subdir, f)
                #     d = json.load(open(fpath))
                #     if n_adaptive == 0:
                #         n_adaptive = d['n_adaptive']
                #     else:
                #         assert n_adaptive == d['n_adaptive'], "FLAGS.n_adaptive do not match in " + subdir
                #     if n_regular == 0:
                #         n_regular = d['n_regular']
                #     else:
                #         assert n_regular == d['n_regular'], "FLAGS.n_regular do not match in " + subdir
                #     print("n_adaptive", n_adaptive, "n_regular", n_regular)
                if "TESTresults.json" in f:
                    fpath = os.path.join(subdir, f)
                    d = json.load(open(fpath))
                    try:
                        taus_per_sim.append(d['tau'])
                        print(subdir, d['tau'])
                    except:
                        print(subdir, "ERROR")
        merged_taus = list(itertools.chain.from_iterable(taus_per_sim))

        import matplotlib.pyplot as plt
        plt.cla()
        plt.clf()
        # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        logbins = np.logspace(np.log10(10), np.log10(1000), 25)
        hist, bins, _ = plt.hist(merged_taus, bins=logbins, normed=False, facecolor='green', alpha=0.5,
                                 edgecolor='black', linewidth=0.5)
        plt.xscale("log")  # , nonposx='clip')
        plt.xlim([10, 10 ** 3])
        locs, labels = plt.yticks()
        plt.yticks(locs, ["{:.1f}".format(100 * i / len(merged_taus)) for i in locs])
        plt.ylabel("percentage of cells")
        # plt.ylabel("num. of cells")
        plt.xlabel("intrinsic time constant (ms)")
        plt_path = os.path.join(args.path, 'autocorr_merged_hist_{}'.format(args.meanhist))
        plt.savefig(plt_path+'.pdf', format='pdf')
        plt.savefig(plt_path+'.png', format='png')
    else:
        print("Attempting to load model from " + args.path)

        if os.path.exists(os.path.join(args.path, 'flags.json')):
            flags_dict = json.load(open(os.path.join(args.path, 'flags.json')))
        else:
            flags_dict = json.load(open(os.path.join(args.path, 'flag.json')))

        from types import SimpleNamespace
        FLAGS = SimpleNamespace(**flags_dict)
        autocorr_plot_neurons(args.path, FLAGS, n_start=0, n_neurons=FLAGS.n_regular+FLAGS.n_adaptive,
                              data_file=args.plot, dirlabel='_ALL_1')
        autocorr_plot_neurons(args.path, FLAGS, n_start=0, n_neurons=FLAGS.n_regular,
                              data_file=args.plot, dirlabel='_LIF_1')
        autocorr_plot_neurons(args.path, FLAGS, n_start=FLAGS.n_regular, n_neurons=FLAGS.n_adaptive,
                              data_file=args.plot, dirlabel='_ALIF_1')
