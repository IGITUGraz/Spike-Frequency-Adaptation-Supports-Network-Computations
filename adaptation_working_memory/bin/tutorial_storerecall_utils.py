import numpy as np
import numpy.random as rd
import tensorflow as tf
from matplotlib import collections as mc, patches
from lsnn.guillaume_toolbox.matplotlib_extension import strip_right_top_axis, raster_plot, hide_bottom_axis


# Variations of sequence with specific delay for plotting
def gen_custom_delay_batch(seq_len, seq_delay, batch_size):
    assert type(seq_delay) is int
    assert 2 + 1 + seq_delay + 1 < seq_len

    def gen_custom_delay_input(seq_len, seq_delay):
        seq_delay = 1 + np.random.choice(seq_len - 2) if seq_delay == 0 else seq_delay
        return [np.random.choice([0, 1]) for _ in range(2)] + \
               [2] + [np.random.choice([0, 1]) for _ in range(seq_delay)] + [3] + \
               [np.random.choice([0, 1]) for _ in range(seq_len - (seq_delay + 4))]

    return np.array([gen_custom_delay_input(seq_len, seq_delay) for i in range(batch_size)])


def error_rate(z, num_Y, num_X, n_character):
    # Find the recall index
    n_recall_symbol = n_character + 1
    shp = tf.shape(num_X)

    # Translate the one hot into ints
    char_predict = tf.argmax(z, axis=2)
    char_true = num_Y
    char_input = num_X

    # error rate 1) Wrong characters
    char_correct = tf.cast(tf.equal(char_predict, char_true), tf.float32)
    character_errors = tf.reduce_mean(1 - char_correct)

    # error rate 2) wrong recalls
    recall_mask = tf.equal(char_input, n_recall_symbol)
    recalls_predict = tf.boolean_mask(char_predict, recall_mask)
    recalls_true = tf.boolean_mask(char_true, recall_mask)

    recall_correct = tf.equal(recalls_predict, recalls_true)
    recall_errors = tf.reduce_mean(tf.cast(tf.logical_not(recall_correct), tf.float32))

    # Get wrong samples
    sentence_id = tf.tile(tf.expand_dims(tf.range(shp[0]), axis=1), (1, shp[1]))
    recall_sentence_id = tf.boolean_mask(sentence_id, recall_mask)
    false_sentence_id_list = tf.boolean_mask(recall_sentence_id, tf.logical_not(recall_correct))

    return character_errors, recall_errors, false_sentence_id_list


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern,list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes


def validity_test(seq, recall_char, store_char):
    is_valid = True

    # At least a store, a digit and a recall
    if np.max(seq == recall_char) == 0 or np.max(seq == store_char) == 0 or np.max(seq < store_char) == 0:
        is_valid = False

    # First store before first recall
    t_first_recall = np.argmax(seq == recall_char)
    t_first_store = np.argmax(seq == store_char)
    if t_first_recall < t_first_store:
        is_valid = False

    # Last recall after last store
    t_last_recall = - np.argmax(seq[::-1] == recall_char)
    t_last_store = - np.argmax(seq[::-1] == store_char)
    if t_last_recall < t_last_store:
        is_valid = False

    # Always a digit after a store
    t_store_list = np.where(seq == store_char)[0]
    for t_store in t_store_list:
        if t_store == seq.size - 1 or seq[t_store + 1] in [recall_char, store_char]:
            is_valid = False
            break

    # Between two recall there is a store
    t_recall_list = np.where(seq == recall_char)[0]
    for k, t_recall in enumerate(t_recall_list[:-1]):
        next_t_recall = t_recall_list[k + 1]

        is_store_between = np.logical_and(t_recall < t_store_list, t_store_list < next_t_recall)
        if not (is_store_between.any()):
            is_valid = False

    # Between two store there is a recall
    for k, t_store in enumerate(t_store_list[:-1]):
        next_t_store = t_store_list[k + 1]

        is_recall_between = np.logical_and(t_store < t_recall_list, t_recall_list < next_t_store)
        if not (is_recall_between.any()):
            is_valid = False
    return is_valid


def generate_input_with_prob(batch_size, length, recall_char, store_char, prob_bit_to_store,
                             prob_bit_to_recall):
    input_nums = np.zeros((batch_size, length), dtype=int)

    for b in range(batch_size):
        last_signal = recall_char

        # init a sequence
        is_valid = False
        seq = rd.choice([0, 1], size=length)

        while not is_valid:
            seq = rd.choice([0, 1], size=length)
            for t in range(length):
                # If the last symbol is a recall we wait for a store
                if last_signal == recall_char and rd.rand() < prob_bit_to_store:
                    seq[t] = store_char
                    last_signal = store_char

                # Otherwise we wait for a recall
                elif last_signal == store_char and rd.rand() < prob_bit_to_recall:
                    seq[t] = recall_char
                    last_signal = recall_char

            is_valid = validity_test(seq, recall_char, store_char)

        input_nums[b, :] = seq

    return input_nums


def generate_data(batch_size, length, n_character, prob_bit_to_store=1. / 3, prob_bit_to_recall=1. / 5, input_nums=None,
                  with_prob=True, delay=None):

    store_char = n_character
    recall_char = n_character + 1

    # Generate the input data
    if input_nums is None:
        if with_prob and prob_bit_to_store < 1. and prob_bit_to_recall < 1.:
            input_nums = generate_input_with_prob(batch_size, length, recall_char, store_char,
                                                  prob_bit_to_store, prob_bit_to_recall)
        else:
            raise ValueError("Only use input generated with probabilities")

    input_nums = np.array(input_nums)

    # generate the output
    target_nums = input_nums.copy()
    inds_recall = np.where(input_nums == recall_char)
    for k_trial, k_t in zip(inds_recall[0], inds_recall[1]):
        assert k_t > 0, 'A recall is put at the beginning to avoid this'
        store_list = np.where(input_nums[k_trial, :k_t] == store_char)[0]
        previous_store_t = store_list[-1]
        target_nums[k_trial, k_t] = input_nums[k_trial, previous_store_t + 1]

    memory_nums = np.ones_like(input_nums) * store_char
    for k_trial in range(batch_size):
        t_store_list = np.where(input_nums[k_trial, :] == store_char)[0]
        for t_store in np.sort(t_store_list):
            if t_store < length - 1:
                memory_nums[k_trial, t_store:] = input_nums[k_trial, t_store + 1]

    return input_nums, target_nums, memory_nums


def generate_mikolov_data(batch_size, length, n_character, with_prob, prob_bit_to_recall,
                          prob_bit_to_store, override_input=None, delay=None):
    if n_character > 2:
        raise NotImplementedError("Not implemented for n_character != 2")
    total_character = n_character + 2
    recall_character = total_character - 1
    store_character = recall_character - 1
    i_1 = np.zeros((batch_size, length), dtype=float)
    i_2 = np.zeros((batch_size, length), dtype=float)
    store = np.zeros((batch_size, length), dtype=float)
    recall = np.zeros((batch_size, length), dtype=float)
    channels = [i_1, i_2, store, recall]
    input_nums, target_nums, memory_nums = generate_data(batch_size, length, n_character,
                                                         with_prob=with_prob, prob_bit_to_recall=prob_bit_to_recall,
                                                         prob_bit_to_store=prob_bit_to_store, input_nums=override_input,
                                                         delay=delay)
    for c in range(total_character):
        channels[c] = np.isin(input_nums, [c]).astype(int)

    for b in range(batch_size):
        for i in range(length):
            if channels[store_character][b,i] == 1:
                # copy next input to concurrent step with store
                channels[0][b,i] = channels[0][b,i+1]
                channels[1][b,i] = channels[1][b,i+1]
                # sometimes inverse the next input
                if rd.uniform() < 0.5:
                    channels[0][b, i+1] = 1 - channels[0][b, i + 1]
                    channels[1][b, i+1] = 1 - channels[1][b, i + 1]
    return channels, target_nums, memory_nums, input_nums


def generate_storerecall_data(batch_size, sentence_length, n_character, n_charac_duration, n_neuron, f0=200 / 1000,
                                 with_prob=True, prob_signals=1 / 5, override_input=None, delay=None):
    channels, target_nums, memory_nums, input_nums = generate_mikolov_data(
        batch_size, sentence_length, n_character, with_prob=with_prob, prob_bit_to_recall=prob_signals,
        prob_bit_to_store=prob_signals, override_input=override_input, delay=delay)

    total_character = n_character + 2  # number of input gates
    recall_character = total_character - 1
    store_character = recall_character - 1

    neuron_split = np.array_split(np.arange(n_neuron), total_character)
    lstm_in_rates = np.zeros((batch_size, sentence_length*n_charac_duration, n_neuron))
    in_gates = channels
    for b in range(batch_size):
        for c in range(sentence_length):
            for in_gate_i, n_group in enumerate(neuron_split):
                lstm_in_rates[b, c*n_charac_duration:(c+1)*n_charac_duration, n_group] = in_gates[in_gate_i][b][c] * f0

    spikes = generate_poisson_noise_np(lstm_in_rates)
    target_sequence = np.repeat(target_nums, repeats=n_charac_duration, axis=1)
    # Generate the recall mask
    is_recall_table = np.zeros((total_character, n_charac_duration), dtype=bool)
    is_recall_table[recall_character, :] = True
    is_recall = np.concatenate([is_recall_table[input_nums][:, k] for k in range(sentence_length)], axis=1)

    return spikes, is_recall, target_sequence, None, input_nums, target_nums


def update_plot(plt, ax_list, FLAGS, plot_result_values, batch=0, n_max_neuron_per_raster=100):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    ylabel_x = -0.11
    ylabel_y = 0.5
    fs = 10
    plt.rcParams.update({'font.size': fs})

    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    z = plot_result_values['z']
    raster_data = \
        zip(range(3), [plot_result_values['input_spikes'], z, z], ['input X', 'R', 'A']) if FLAGS.n_regular > 0 else \
        zip(range(2), [plot_result_values['input_spikes'], z], ['input X', 'A'])

    for k_data, data, d_name in raster_data:
        ax = ax_list[k_data]
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        hide_bottom_axis(ax)

        if np.size(data) > 0:
            data = data[batch]
            if d_name is 'R':
                data = data[:, :FLAGS.n_regular]
            elif d_name is 'A':
                data = data[:, FLAGS.n_regular:]
            n_max = min(data.shape[1], n_max_neuron_per_raster)
            cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
            data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
            raster_plot(ax, data, linewidth=0.15)
            ax.set_ylabel(d_name, fontsize=fs)
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
            ax.set_yticklabels(['1', str(data.shape[-1])])

            if k_data == 0:
                ax.set_yticklabels([])
                n_channel = data.shape[1] // (FLAGS.n_charac + 2)  # divide #in_neurons with #in_channels
                ax.add_patch(  # Value 0 row
                    patches.Rectangle((0, 0), data.shape[0], n_channel, facecolor="red", alpha=0.15))
                ax.add_patch(  # Value 1 row
                    patches.Rectangle((0, n_channel), data.shape[0], n_channel, facecolor="blue", alpha=0.15))
                ax.add_patch(  # Store row
                    patches.Rectangle((0, 2 * n_channel), data.shape[0], n_channel, facecolor="yellow", alpha=0.15))
                ax.add_patch(  # Recall row
                    patches.Rectangle((0, 3 * n_channel), data.shape[0], n_channel, facecolor="green", alpha=0.15))

                top_margin = 0.08
                left_margin = -0.085
                ax.text(left_margin, 1. - top_margin, 'recall', transform=ax.transAxes, fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.75 - top_margin, 'store', transform=ax.transAxes, fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.5 - top_margin, 'value 1', transform=ax.transAxes, fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.25 - top_margin, 'value 0', transform=ax.transAxes, fontsize=7, verticalalignment='top')

    ax = ax_list[-2]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('thresholds of A', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    sub_data = plot_result_values['b_con'][batch]
    vars = np.var(sub_data, axis=0)
    cell_with_max_var = np.argsort(vars)[::-1]
    presentation_steps = np.arange(sub_data.shape[0])
    ax.plot(sub_data[:, cell_with_max_var], color='r', alpha=0.4, linewidth=1)
    ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
             np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]
    hide_bottom_axis(ax)

    # plot targets
    ax = ax_list[-1]
    mask = plot_result_values['recall_charac_mask'][batch]
    data = plot_result_values['target_nums'][batch]
    data[np.invert(mask)] = -1
    lines = []
    ind_nt = np.argwhere(data != -1)
    for idx in ind_nt.tolist():
        i = idx[0]
        lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    lc_t = mc.LineCollection(lines, colors='green', linewidths=2, label='target')
    ax.add_collection(lc_t)  # plot target segments

    # plot output per tau_char
    data = plot_result_values['out_plot_char_step'][batch]
    data = np.array([(d[1] - d[0] + 1) / 2 for d in data])
    data[np.invert(mask)] = -1
    lines = []
    ind_nt = np.argwhere(data != -1)
    for idx in ind_nt.tolist():
        i = idx[0]
        lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    lc_o = mc.LineCollection(lines, colors='blue', linewidths=2, label='avg. output')
    ax.add_collection(lc_o)  # plot target segments

    # plot softmax of psp-s per dt for more intuitive monitoring
    # ploting only for second class since this is more intuitive to follow (first class is just a mirror)
    output2 = plot_result_values['out_plot'][batch, :, 1]
    presentation_steps = np.arange(output2.shape[0])
    ax.set_yticks([0, 0.5, 1])
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('output Y', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    line_output2, = ax.plot(presentation_steps, output2, color='purple', label='output', alpha=0.7)
    ax.axis([0, presentation_steps[-1] + 1, -0.1, 1.1])
    ax.legend(handles=[lc_t, lc_o, line_output2], loc='lower center', fontsize=7,
              bbox_to_anchor=(0.5, -0.1), ncol=3)

    ax.set_xlabel('time in ms', fontsize=fs)
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(1)


def update_stp_plot(plt, ax_list, FLAGS, plot_result_values, batch=0, n_max_neuron_per_raster=100):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    ylabel_x = -0.11
    ylabel_y = 0.5
    fs = 10
    plt.rcParams.update({'font.size': fs})

    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    z = plot_result_values['z']
    raster_data = \
        zip(range(3), [plot_result_values['input_spikes'], z, z], ['input X', 'R', 'A']) if FLAGS.n_regular > 0 else \
        zip(range(2), [plot_result_values['input_spikes'], z], ['input X', 'A'])

    for k_data, data, d_name in raster_data:
        ax = ax_list[k_data]
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        hide_bottom_axis(ax)

        if np.size(data) > 0:
            data = data[batch]
            if d_name is 'R':
                data = data[:, :FLAGS.n_regular]
            elif d_name is 'A':
                data = data[:, FLAGS.n_regular:]
            n_max = min(data.shape[1], n_max_neuron_per_raster)
            cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
            data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
            raster_plot(ax, data, linewidth=0.15)
            ax.set_ylabel(d_name, fontsize=fs)
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
            ax.set_yticklabels(['1', str(data.shape[-1])])

            if k_data == 0:
                ax.set_yticklabels([])
                n_channel = data.shape[1] // (FLAGS.n_charac + 2)  # divide #in_neurons with #in_channels
                ax.add_patch(  # Value 0 row
                    patches.Rectangle((0, 0), data.shape[0], n_channel, facecolor="red", alpha=0.15))
                ax.add_patch(  # Value 1 row
                    patches.Rectangle((0, n_channel), data.shape[0], n_channel, facecolor="blue", alpha=0.15))
                ax.add_patch(  # Store row
                    patches.Rectangle((0, 2 * n_channel), data.shape[0], n_channel, facecolor="yellow", alpha=0.15))
                ax.add_patch(  # Recall row
                    patches.Rectangle((0, 3 * n_channel), data.shape[0], n_channel, facecolor="green", alpha=0.15))

                top_margin = 0.08
                left_margin = -0.085
                ax.text(left_margin, 1. - top_margin, 'recall', transform=ax.transAxes, fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.75 - top_margin, 'store', transform=ax.transAxes, fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.5 - top_margin, 'value 1', transform=ax.transAxes, fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.25 - top_margin, 'value 0', transform=ax.transAxes, fontsize=7, verticalalignment='top')

    ax = ax_list[-2]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('STP u, x', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    u_data = plot_result_values['stp_u'][batch]
    x_data = plot_result_values['stp_x'][batch]
    vars = np.var(u_data, axis=0)
    cell_with_max_var = np.argsort(vars)[::-1]
    presentation_steps = np.arange(u_data.shape[0])
    ax.plot(u_data[:, cell_with_max_var], color='b', alpha=0.4, linewidth=1, label="u")
    ax.plot(x_data[:, cell_with_max_var], color='r', alpha=0.4, linewidth=1, label="x")
    # [xmin, xmax, ymin, ymax]
    # ymin = np.min([np.min(u_data[:, cell_with_max_var]), np.min(x_data[:, cell_with_max_var])])
    # ymax = np.min([np.max(u_data[:, cell_with_max_var]), np.max(x_data[:, cell_with_max_var])])
    # ax.axis([0, presentation_steps[-1], ymin, ymax])
    ax.axis([0, presentation_steps[-1], 0., 1.])
    # ax.legend()
    hide_bottom_axis(ax)

    # plot targets
    ax = ax_list[-1]
    mask = plot_result_values['recall_charac_mask'][batch]
    data = plot_result_values['target_nums'][batch]
    data[np.invert(mask)] = -1
    lines = []
    ind_nt = np.argwhere(data != -1)
    for idx in ind_nt.tolist():
        i = idx[0]
        lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    lc_t = mc.LineCollection(lines, colors='green', linewidths=2, label='target')
    ax.add_collection(lc_t)  # plot target segments

    # plot output per tau_char
    data = plot_result_values['out_plot_char_step'][batch]
    data = np.array([(d[1] - d[0] + 1) / 2 for d in data])
    data[np.invert(mask)] = -1
    lines = []
    ind_nt = np.argwhere(data != -1)
    for idx in ind_nt.tolist():
        i = idx[0]
        lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    lc_o = mc.LineCollection(lines, colors='blue', linewidths=2, label='avg. output')
    ax.add_collection(lc_o)  # plot target segments

    # plot softmax of psp-s per dt for more intuitive monitoring
    # ploting only for second class since this is more intuitive to follow (first class is just a mirror)
    output2 = plot_result_values['out_plot'][batch, :, 1]
    presentation_steps = np.arange(output2.shape[0])
    ax.set_yticks([0, 0.5, 1])
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('output Y', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    line_output2, = ax.plot(presentation_steps, output2, color='purple', label='output', alpha=0.7)
    ax.axis([0, presentation_steps[-1] + 1, -0.1, 1.1])
    ax.legend(handles=[lc_t, lc_o, line_output2], loc='lower center', fontsize=7,
              bbox_to_anchor=(0.5, -0.1), ncol=3)

    ax.set_xlabel('time in ms', fontsize=fs)
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(1)


def offline_plot(data_path, custom_plot=True):
    import matplotlib.pyplot as plt
    import datetime
    import pickle
    import json
    import os

    flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    from types import SimpleNamespace
    flags = SimpleNamespace(**flags_dict)

    plot_data = 'plot_custom_trajectory_data.pickle' if custom_plot else 'plot_trajectory_data.pickle'
    plot_result_values = pickle.load(open(os.path.join(data_path, plot_data), 'rb'))

    plt.ion()
    nrows = 5 if flags.n_regular > 0 else 4
    height = 7.5 if flags.n_regular > 0 else 6
    fig, ax_list = plt.subplots(nrows=nrows, figsize=(6, height), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    for b in range(flags.batch_test):
        update_plot(plt, ax_list, flags, plot_result_values, batch=b, n_max_neuron_per_raster=100)
        start_time = datetime.datetime.now()
        fig.savefig(os.path.join(data_path, 'figure_test' + str(b) + '_' + start_time.strftime("%H%M") + '.pdf'),
                    format='pdf')


def avg_firingrates_during_delay(data_path):
    """
    Calculate average firing rates during delays of custom plot.
    Data is conditioned on the current memory content
    [0 (after storing 0), 1 (after storing 1), blank (after recall)]
    Motivation: check if firing rate is higher during delay if the memory is filled.
    :param data_path:
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime
    import pickle
    import json
    import os

    flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    from types import SimpleNamespace
    flags = SimpleNamespace(**flags_dict)

    plot_result_values = pickle.load(open(os.path.join(data_path, 'plot_custom_trajectory_data.pickle'), 'rb'))
    firing_rates = {'0': [], '1': [], 'blank': []}
    for b in range(flags.batch_test):  # plot_result_values['input_nums'].shape[0]:
        symbolic_input = plot_result_values['input_nums'][b]
        z = plot_result_values['z'][b]
        step = flags.tau_char
        # index 1 and 12 determine the content of memory during the first and third delay periods
        # recalls at 7 and 19
        # relevant periods 1-7 first memory, 8-12 blank, 13-19 second memory
        firing_rates[str(symbolic_input[1])].append(np.mean(z[1*step:7*step]))
        firing_rates['blank'].append(np.mean(z[8*step:12*step]))
        firing_rates[str(symbolic_input[12])].append(np.mean(z[13*step:19*step]))
    for k in firing_rates.keys():
        firing_rates[k] = np.mean(firing_rates[k]) * 1000
        print("AVG Firing rate for memory content ({}) = {:.2g}".format(k, firing_rates[k]))
    return firing_rates
