import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import argparse
import pickle
import json
import os


def decode_memory_with_SVC(dir_path, plot_filename):
    if os.path.exists(os.path.join(dir_path, 'flags.json')):
        flags_dict = json.load(open(os.path.join(dir_path, 'flags.json')))
    else:
        flags_dict = json.load(open(os.path.join(dir_path, 'flag.json')))

    from types import SimpleNamespace
    FLAGS = SimpleNamespace(**flags_dict)

    data = pickle.load(open(os.path.join(dir_path, plot_filename), 'rb'))

    raw_input = data['input_spikes']  # also for analog input the key is 'input_spikes'
    # print(FLAGS)
    # print(raw_input.shape)  # batch, time, channels (128, 1000, 60)
    shp = raw_input.shape
    ch_in = np.mean(np.reshape(raw_input, (shp[0], -1, FLAGS.tau_char, shp[2])), axis=2)  # avg per char step
    shp = ch_in.shape
    ch_in = np.mean(np.reshape(ch_in, (shp[0], shp[1], -1, FLAGS.n_per_channel)), axis=3)  # avg per channel
    ch_in = ch_in > 0.0  # convert to binary
    # print(ch_in.shape)
    n_group = FLAGS.n_charac  # size of a group in input channels. groups: store-recall, input, inv-input
    assert ch_in.shape[2] == 2 * n_group + 2 * 2,\
        "ch_in.shape[2]" + str(ch_in.shape[2]) + " does not contain 3 groups of " + str(n_group)
    store = np.mean(ch_in[:, :, :2], axis=2)[..., np.newaxis]  # first half of first group
    recall = np.mean(ch_in[:, :, 2:4], axis=2)[..., np.newaxis]  # second half of first group
    norm_input = ch_in[:, :, 4:4+n_group]
    # print("store", store.shape)
    # print("recall", recall.shape)
    # print("norm_input", norm_input.shape)
    plot_input = np.dstack([store, recall, norm_input])

    # plt.cla()
    # plt.clf()
    # plt.imshow(plot_input[1].T)
    # plt.tight_layout()
    # plt.show()
    # exit()

    z_output = data['z']  # batch, time, neurons
    shp = z_output.shape
    z_avg = np.mean(np.reshape(z_output, (shp[0], -1, FLAGS.tau_char, shp[2])), axis=2)  # batch, FLAGS.seq_len, neurons

    def to_label(input_data):
        return ''.join(input_data.astype(int).astype(str))

    def get_data():
        x_store = []
        x_delay = []
        x_recall = []
        # y_store = []
        y_recall = []
        store_idxs = np.nonzero(store)  # list of batch idxs, list of time idxs
        recall_idxs = np.nonzero(recall)  # list of batch idxs, list of time idxs
        for b, s, r in zip(store_idxs[0], store_idxs[1], recall_idxs[1]):
            x_recall.append(z_avg[b, r])
            x_delay.append(z_avg[b, r - 1])
            x_store.append(z_avg[b, s])
            # y_store.append(to_label(norm_input[b, s]))
            y_recall.append(to_label(norm_input[b, s]))
        return x_store, x_delay, x_recall, y_recall#, y_store

    x_store_all, x_delay_all, x_recall_all, y_recall_all = get_data()

    results = {}
    for x, y, dname in [(x_store_all, y_recall_all, "STORE"),
                        (x_delay_all, y_recall_all, "DELAY"),
                        (x_recall_all, y_recall_all, "RECALL")]:
        parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.1, 1, 10, 100, 1000]}
        svc = svm.SVC(gamma="auto")
        clf = GridSearchCV(svc, parameters, cv=5, refit=True, iid=True)
        clf.fit(x, y)
        print("MODEL trained during ", dname, "score:", clf.best_score_)
        recall_score = clf.score(x_recall_all, y_recall_all)
        store_score = clf.score(x_store_all, y_recall_all)
        print("scored on recall data:", recall_score, "scored on store data:", store_score)
        # print("params:", clf.best_params_)
        results[dname] = {'period': dname, 'accuracy': clf.best_score_, 'params': clf.best_params_}
    json.dump(results, open(os.path.join(dir_path, 'SVM_information_analysis.json'), 'w'),
              indent=4, sort_keys=True)
    return results


if __name__ == "__main__":
    """
    To compute decoding accuracy of a single run, use path argument like so:
    PYTHONPATH=. python3 bin/SVM_binstr_information_analysis.py --path path/to/simulation/result
    Otherwise use --dir argument to compute for every simulation in directory filtered with --filter
    """
    parser = argparse.ArgumentParser(description='Try to predict labels based on network firing activity using SVMs.')
    parser.add_argument('--path', default=None, help='Path to directory that contains flags and plot data.')
    parser.add_argument('--filter', default="_ExtSR_", help='When using --dir filter simulations by this string.')
    parser.add_argument('--dir', default='results/tutorial_extended_storerecall_with_LSNN',
                        help='Path to directory that contains flags and plot data.')
    parser.add_argument('--plot', default='plot_trajectory_data.pickle',
                        help='Filename of pickle file containing data for plotting.')
    args = parser.parse_args()

    if args.path is not None:
        print("Attempting to load model from " + args.path)
        res = decode_memory_with_SVC(dir_path=args.path, plot_filename=args.plot)
    else:
        # dirs = [name for name in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, name))]
        # dirs = ['2019_12_05_10_17_41_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG2_GPU_0',
        #         '2019_12_05_17_13_18_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG2_GPU_1',
        #         '2019_12_05_10_18_23_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG2_0',
        #         '2019_12_05_17_18_55_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG2_0']
        dirs = ['2019_12_08_03_51_50_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG3_3',
                '2019_12_07_00_12_39_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG3_1',
                '2019_12_08_13_28_08_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG3_4',
                '2019_12_07_09_58_07_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG3_2',
                '2019_12_06_10_27_42_FastALIF_seqlen10_seqdelay4_in88_R0_A500_lr0.01_tauchar200_comment560_ExtSR_FastLONG3_0']
        delay_accs = []
        recall_accs = []
        for dir in dirs:
            if args.filter not in dir:
                continue
            print("Processing:", dir)
            try:
                res = decode_memory_with_SVC(dir_path=os.path.join(args.dir, dir), plot_filename=args.plot)
            except Exception as e:
                print(e)
                continue
            delay_accs.append(res['DELAY']['accuracy'])
            recall_accs.append(res['RECALL']['accuracy'])
        print("delay_accs", delay_accs)
        print("recall_accs", recall_accs)
        print("ACCURACY during delay of {} simulations: {} +- {}".format(
            len(delay_accs), np.mean(delay_accs), np.std(delay_accs)))
        print("ACCURACY during recall of {} simulations: {} +- {}".format(
            len(recall_accs), np.mean(recall_accs), np.std(recall_accs)))


"""
salaj@figipc64 /calc/salaj/repos/LSNN_560 $ python3 bin/SVM_binstr_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/2019_11_30_12_28_51_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_0/
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_11_30_12_28_51_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_0/
MODEL trained during  DELAY
score: 0.22280701754385965
params: {'C': 1, 'kernel': 'linear'}
MODEL trained during  RECALL
score: 1.0
params: {'C': 0.1, 'kernel': 'linear'}

(gui) salaj@node21 /calc/salaj/repos/LSNN_560 $ PYTHONPATH=. python3 bin/SVM_binstr_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/20
19_12_02_20_34_53_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_2/                                                               
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_12_02_20_34_53_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_2/
MODEL trained during  DELAY
score: 0.22800718132854578
params: {'C': 0.1, 'kernel': 'linear'}
MODEL trained during  RECALL
score: 1.0
params: {'C': 0.1, 'kernel': 'linear'}

(gui) salaj@node21 /calc/salaj/repos/LSNN_560 $ PYTHONPATH=. python3 bin/SVM_binstr_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/2019_11_30_12_19_06_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_0/                                                               
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_11_30_12_19_06_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_0/
MODEL trained during  DELAY
score: 0.22241992882562278
params: {'C': 1, 'kernel': 'linear'}
MODEL trained during  RECALL
score: 1.0
params: {'C': 0.1, 'kernel': 'linear'}

(gui) salaj@node21 /calc/salaj/repos/LSNN_560 $ PYTHONPATH=. python3 bin/SVM_binstr_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/2019_11_30_12_23_06_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_0/                                                               
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_11_30_12_23_06_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_0/
MODEL trained during  DELAY
score: 0.23024054982817868
params: {'C': 1, 'kernel': 'linear'}
MODEL trained during  RECALL
score: 1.0
params: {'C': 0.1, 'kernel': 'linear'}

(gui) salaj@node21 /calc/salaj/repos/LSNN_560 $ PYTHONPATH=. python3 bin/SVM_binstr_information_analysis.py results/tutorial_extended_storerecall_with_LSNN/2019_12_01_05_55_12_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_1/                                                               
Attempting to load model from results/tutorial_extended_storerecall_with_LSNN/2019_12_01_05_55_12_ALIF_seqlen10_seqdelay4_in120_R0_A1000_lr0.01_tauchar100_comment560_ExtSR_1/
MODEL trained during  DELAY
score: 0.2198952879581152
params: {'C': 1, 'kernel': 'linear'}
MODEL trained during  RECALL
score: 1.0
params: {'C': 0.1, 'kernel': 'linear'}

"""
