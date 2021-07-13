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
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    import argparse
    import pickle
    import json
    import os

    parser = argparse.ArgumentParser(description='Try to predict labels based on network firing activity using SVMs.')
    parser.add_argument('path', help='Path to directory that contains flags and plot data.')
    parser.add_argument('--plot', default='plot_trajectory_data.pickle',
                        help='Filename of pickle file containing data for plotting.')
    args = parser.parse_args()

    print("Attempting to load model from " + args.path)

    if os.path.exists(os.path.join(args.path, 'flags.json')):
        flags_dict = json.load(open(os.path.join(args.path, 'flags.json')))
    else:
        flags_dict = json.load(open(os.path.join(args.path, 'flag.json')))

    from types import SimpleNamespace
    FLAGS = SimpleNamespace(**flags_dict)

    data = pickle.load(open(os.path.join(args.path, args.plot), 'rb'))
    if not FLAGS.analog_in:
        raise ValueError("Spiking input analysis not implemented")
    raw_input = data['input_spikes']  # also for analog input the key is 'input_spikes'
    # print(FLAGS)
    # print(raw_input.shape)  # batch, time, channels (128, 1000, 60)
    shp = raw_input.shape
    ch_in = np.mean(np.reshape(raw_input, (shp[0], -1, FLAGS.tau_char, shp[2])), axis=2)  # avg per char step
    shp = ch_in.shape
    ch_in = np.mean(np.reshape(ch_in, (shp[0], shp[1], -1, FLAGS.n_per_channel)), axis=3)  # avg per channel
    # print(ch_in.shape)
    n_group = FLAGS.n_charac  # size of a group in input channels. groups: store-recall, input, inv-input
    assert ch_in.shape[2] == 3 * n_group,\
        "ch_in.shape[2]" + str(ch_in.shape[2]) + " does not contain 3 groups of " + str(n_group)
    store = np.mean(ch_in[:, :, :n_group//2], axis=2)[..., np.newaxis]  # first half of first group
    recall = np.mean(ch_in[:, :, n_group//2:n_group], axis=2)[..., np.newaxis]  # second half of first group
    norm_input = ch_in[:, :, n_group:2*n_group]
    # print("store", store.shape)
    # print("recall", recall.shape)
    # print("norm_input", norm_input.shape)
    plot_input = np.dstack([store, recall, norm_input])

    # plt.cla()
    # plt.clf()
    # plt.imshow(plot_input[1].T)
    # plt.tight_layout()
    # plt.show()

    z_output = data['z']  # batch, time, neurons
    shp = z_output.shape
    z_avg = np.mean(np.reshape(z_output, (shp[0], -1, FLAGS.tau_char, shp[2])), axis=2)  # batch, FLAGS.seq_len, neurons

    def to_label(input_data):
        # return index of the first non-zero element (one-hot -> integers)
        return np.nonzero(input_data)[0][0]

    def get_delay_data():
        x_all = []
        y_all = []
        x = []
        y = []
        x_test = []
        y_test = []
        store_idxs = np.nonzero(store)  # list of batch idxs, list of time idxs
        recall_idxs = np.nonzero(recall)  # list of batch idxs, list of time idxs
        for b, s, r in zip(store_idxs[0], store_idxs[1], recall_idxs[1]):
            # print("batch", b, "store", s, "recall", r)
            if np.random.random() < 0.1:
                x_test.append(z_avg[b, r-1])
                y_test.append(to_label(norm_input[b, s]))
            else:
                x.append(z_avg[b, r-1])
                y.append(to_label(norm_input[b, s]))
            x_all.append(z_avg[b, r-1])
            y_all.append(to_label(norm_input[b, s]))
        return x, y, x_test, y_test, x_all, y_all

    def get_recall_data():
        x_all = []
        y_all = []
        x = []
        y = []
        x_test = []
        y_test = []
        store_idxs = np.nonzero(store)  # list of batch idxs, list of time idxs
        recall_idxs = np.nonzero(recall)  # list of batch idxs, list of time idxs
        for b, s, r in zip(store_idxs[0], store_idxs[1], recall_idxs[1]):
            # print("batch", b, "store", s, "recall", r)
            if np.random.random() < 0.1:
                x_test.append(z_avg[b, r])
                y_test.append(to_label(norm_input[b, s]))
            else:
                x.append(z_avg[b, r])
                y.append(to_label(norm_input[b, s]))
            x_all.append(z_avg[b, r])
            y_all.append(to_label(norm_input[b, s]))
        return x, y, x_test, y_test, x_all, y_all

    x_delay, y_delay, x_delay_test, y_delay_test, x_delay_all, y_delay_all = get_delay_data()
    x_recall, y_recall, x_recall_test, y_recall_test, x_recall_all, y_recall_all = get_recall_data()

    for x, y, dname in [(x_delay_all, y_delay_all, "DELAY"), (x_recall_all, y_recall_all, "RECALL")]:
        parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.1, 1, 10, 100, 1000, 10000]}
        svc = svm.SVC(gamma="auto")
        clf = GridSearchCV(svc, parameters, cv=5, refit=True, iid=True)
        clf.fit(x, y)
        print("MODEL trained during ", dname)
        print("score:", clf.best_score_)
        print("params:", clf.best_params_)

"""
salaj@figipc64 /calc/salaj/repos/LSNN_560 $ python3 bin/activity_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/2019_11_27_17_07_48_ALIF_seqlen10_seqdelay4_in60_R0_A300_lr0.01_tauchar100_commentDEBUG_POPENC_EntrLoss/
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_11_27_17_07_48_ALIF_seqlen10_seqdelay4_in60_R0_A300_lr0.01_tauchar100_commentDEBUG_POPENC_EntrLoss/
MODEL trained during  DELAY
score: 0.1357142857142857
params: {'kernel': 'linear', 'C': 10}
MODEL trained during  RECALL
score: 0.9785714285714285
params: {'kernel': 'linear', 'C': 0.1}


salaj@figipc64 /calc/salaj/repos/LSNN_560 $ python3 bin/activity_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/2019_11_27_16_33_05_ALIF_seqlen10_seqdelay4_in60_R300_A300_lr0.01_tauchar100_commentDEBUG_POPENC_EntrLoss/
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_11_27_16_33_05_ALIF_seqlen10_seqdelay4_in60_R300_A300_lr0.01_tauchar100_commentDEBUG_POPENC_EntrLoss/
MODEL trained during  DELAY
score: 0.14285714285714285
params: {'C': 100, 'kernel': 'linear'}
MODEL trained during  RECALL
score: 0.9071428571428571
params: {'C': 10, 'kernel': 'linear'}


WITH cv=3:
salaj@figipc64 /calc/salaj/repos/LSNN_560 $ python3 bin/activity_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/2019_11_28_09_34_34_ALIF_seqlen10_seqdelay4_in150_R0_A300_lr0.01_tauchar100_commentDEBUG_POPENC_EntrLoss/
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_11_28_09_34_34_ALIF_seqlen10_seqdelay4_in150_R0_A300_lr0.01_tauchar100_commentDEBUG_POPENC_EntrLoss/
/home/salaj/.local/lib/python3.5/site-packages/sklearn/model_selection/_split.py:657: Warning: The least populated class in y has only 2 members, which is too few. The minimum number of members in any class cannot be less than n_splits=3.
  % (min_groups, self.n_splits)), Warning)
MODEL trained during  DELAY
score: 0.08450704225352113
params: {'kernel': 'linear', 'C': 0.1}
/home/salaj/.local/lib/python3.5/site-packages/sklearn/model_selection/_split.py:657: Warning: The least populated class in y has only 2 members, which is too few. The minimum number of members in any class cannot be less than n_splits=3.
  % (min_groups, self.n_splits)), Warning)
MODEL trained during  RECALL
score: 0.9929577464788732
params: {'kernel': 'linear', 'C': 10}


"""
