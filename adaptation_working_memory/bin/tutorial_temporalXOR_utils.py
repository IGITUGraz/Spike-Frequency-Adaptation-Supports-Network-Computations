import numpy as np
import numpy.random as rd
from scipy.stats import norm

from bin.tutorial_extended_storerecall_utils import plot_spikes
from lsnn.guillaume_toolbox.matplotlib_extension import strip_right_top_axis, raster_plot, hide_bottom_axis


def generate_xor_input(batch_size, length, expected_delay=100, pulse_duration=30, out_duration=60):
    input_nums = np.zeros((batch_size, length), dtype=float)
    gocue_nums = np.zeros((batch_size, length), dtype=float)
    targets = np.ones((batch_size), dtype=int) * 2
    # target_nums = np.zeros((batch_size, length), dtype=int)  # 2 classes
    target_nums = np.ones((batch_size, length), dtype=int) * 2
    target_mask_nums = np.zeros((batch_size, length), dtype=int)
    delay_prob = 10/expected_delay

    def prob_calc_delay():
        d = 10
        while True:
            if delay_prob > rd.uniform():
                break
            else:
                d += 10
        d = min(d, int(expected_delay))
        return d

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def pulse_trace():
        return normalize(norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), pulse_duration)))

    for b in range(batch_size):
        null_target = 0.2 > rd.uniform()
        pulse_delay = expected_delay if b == 0 else prob_calc_delay()
        seq = np.zeros(length)
        p1 = rd.choice([-1, 1])
        p2 = rd.choice([-1, 1])
        pulse_1 = pulse_trace() * p1
        pulse_2 = pulse_trace() * p2
        seq[0:pulse_duration] += pulse_1
        if not null_target:
            seq[pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_2
        input_nums[b, :] = seq

        gocue_seq = np.zeros(length)
        end_pulses = np.nonzero(seq)[0][-1]  # FIXME: null target moves cue earlier
        gocue_seq[end_pulses+pulse_delay:end_pulses+pulse_delay+pulse_duration] += pulse_trace()
        gocue_nums[b, :] = gocue_seq

        # target = 1 if p1 != p2 else -1
        target = p1 != p2
        start_cue = end_pulses+pulse_delay
        if not null_target:
            targets[b] = target
            target_nums[b, start_cue:start_cue+pulse_duration+out_duration] = target

        target_mask_nums[b, start_cue:start_cue+out_duration] = 1
        # target_mask_nums[b, :] = 1

    network_input = np.stack((input_nums, gocue_nums), axis=-1)  # batch x length x 2
    return network_input, target_nums, target_mask_nums, targets


def generate_xor_spike_input(batch_size, length, expected_delay=100, pulse_duration=30, out_duration=60):
    input_nums0 = np.zeros((batch_size, length), dtype=float)
    input_nums1 = np.zeros((batch_size, length), dtype=float)
    gocue_nums = np.zeros((batch_size, length), dtype=float)
    targets = np.zeros((batch_size), dtype=int)
    target_nums = np.zeros((batch_size, length), dtype=int)
    target_mask_nums = np.zeros((batch_size, length), dtype=int)
    delay_prob = 10/expected_delay

    def prob_calc_delay():
        d = 10
        while True:
            if delay_prob > rd.uniform():
                break
            else:
                d += 10
        d = min(d, int(expected_delay))
        return d

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def pulse_trace():
        return normalize(norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), pulse_duration)))

    for b in range(batch_size):
        pulse_delay = prob_calc_delay()
        p1 = rd.choice([0, 1])
        p2 = rd.choice([0, 1])
        if p1 == 0:
            input_nums0[b, 0:pulse_duration] += pulse_trace()
        else:
            input_nums1[b, 0:pulse_duration] += pulse_trace()

        if p2 == 0:
            input_nums0[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()
        else:
            input_nums1[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()

        seq = input_nums0[b] + input_nums1[b]
        end_pulses = np.nonzero(seq)[0][-1]
        gocue_seq = np.zeros(length)
        gocue_seq[end_pulses+pulse_delay:end_pulses+pulse_delay+pulse_duration] += pulse_trace()
        gocue_nums[b, :] = gocue_seq

        # target = 1 if p1 != p2 else -1
        target = p1 != p2
        targets[b] = target
        start_cue = end_pulses+pulse_delay
        target_nums[b, start_cue:start_cue+pulse_duration+out_duration] = target

        target_mask_nums[b, start_cue:start_cue+out_duration] = 1
        # target_mask_nums[b, :] = 1

    network_input = np.stack((input_nums0, input_nums1, gocue_nums), axis=-1)  # batch x length x 2
    return network_input, target_nums, target_mask_nums, targets


def generate_3class_xor_spike_input(batch_size, length, expected_delay=100, pulse_duration=30, out_duration=60):
    input_nums0 = np.zeros((batch_size, length), dtype=float)
    input_nums1 = np.zeros((batch_size, length), dtype=float)
    gocue_nums = np.zeros((batch_size, length), dtype=float)
    targets = np.zeros((batch_size), dtype=int)
    target_nums = np.ones((batch_size, length), dtype=int) * 2
    target_mask_nums = np.zeros((batch_size, length), dtype=int)
    delay_prob = 10/expected_delay

    def prob_calc_delay():
        d = 10
        while True:
            if delay_prob > rd.uniform():
                break
            else:
                d += 10
        d = min(d, int(expected_delay))
        return d

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def pulse_trace():
        return normalize(norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), pulse_duration)))

    delays = []
    for b in range(batch_size):
        pulse_delay = prob_calc_delay()
        delays.append(pulse_delay)
        p1 = rd.choice([0, 1])
        p2 = rd.choice([0, 1])
        if p1 == 0:
            input_nums0[b, 0:pulse_duration] += pulse_trace()
        else:
            input_nums1[b, 0:pulse_duration] += pulse_trace()

        if p2 == 0:
            input_nums0[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()
        else:
            input_nums1[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()

        seq = input_nums0[b] + input_nums1[b]
        end_pulses = np.nonzero(seq)[0][-1]
        gocue_seq = np.zeros(length)
        gocue_seq[end_pulses+pulse_delay:end_pulses+pulse_delay+pulse_duration] += pulse_trace()
        gocue_nums[b, :] = gocue_seq

        # target = 1 if p1 != p2 else -1
        target = p1 != p2
        targets[b] = target
        start_cue = end_pulses+pulse_delay
        target_nums[b, start_cue:start_cue+pulse_duration+out_duration] = target

        target_mask_nums[b, start_cue:start_cue+out_duration] = 1
        # target_mask_nums[b, :] = 1

    network_input = np.stack((input_nums0, input_nums1, gocue_nums), axis=-1)  # batch x length x 2
    return network_input, target_nums, target_mask_nums, targets


def update_plot(plt, ax_list, FLAGS, plot_result_values, batch=0, end_time=600):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    ylabel_x = -0.08
    ylabel_y = 0.5
    fs = 10
    plt.rcParams.update({'font.size': fs})
    end_time =plot_result_values['input_spikes'].shape[1]

    sub_data = plot_result_values['b_con'][batch, :end_time]
    vars = np.var(sub_data, axis=0)
    cell_with_max_var = np.argsort(vars)[::-1]
    cell_with_max_var = cell_with_max_var[::4]  # half the neurons

    # Clear the axis to draw new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # PLOT Input signals
    ax = ax_list[0]
    data = plot_result_values['input_spikes']
    data = data[batch, :end_time]
    presentation_steps = np.arange(data.shape[0])
    ax.plot(presentation_steps, data[:, 0], color='tab:blue', alpha=0.7)
    ax.axis([0, len(data), -1.1, 1.1])
    ax.set_ylabel('Input', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    ax.set_yticks([-1, 1])
    hide_bottom_axis(ax)

    # PLOT Go-cue
    ax = ax_list[1]
    data = plot_result_values['input_spikes']
    data = data[batch, :end_time]
    presentation_steps = np.arange(data.shape[0])
    ax.plot(presentation_steps, data[:, 1], color='tab:red', alpha=0.7)
    ax.axis([0, len(data), -1.1, 1.1])
    ax.set_ylabel('Go-cue', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    ax.set_yticks([-1, 1])
    hide_bottom_axis(ax)

    # PLOT SPIKES
    # ax = ax_list[2]
    # data = plot_result_values['z']
    # data = data[batch, :end_time,]
    # # raster_plot(ax, data, linewidth=1.)
    # plot_spikes(ax, data.T, linewidth=0.15, max_spike=20000)
    # ax.set_ylabel('LSNN', fontsize=fs)
    # ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    # hide_bottom_axis(ax)
    ax = ax_list[2]
    data = plot_result_values['z']
    data = data[batch, :end_time, cell_with_max_var]
    # raster_plot(ax, data, linewidth=1.)
    plot_spikes(ax, data, linewidth=0.15, max_spike=20000)
    ax.set_ylabel('ALIF', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    hide_bottom_axis(ax)

    # debug plot for psp-s or biases
    plot_param = 'b_con'  # or 'psp'
    ax.set_xticklabels([])
    ax = ax_list[-2]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Thresholds', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    sub_data = plot_result_values['b_con'][batch, :end_time]
    # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses * 3:3]
    presentation_steps = np.arange(sub_data.shape[0])
    ax.plot(sub_data[:, cell_with_max_var], color='r', label='Output', alpha=0.4, linewidth=1)
    ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
                 np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]
    hide_bottom_axis(ax)

    # PLOT OUTPUT AND TARGET
    ax = ax_list[-1]
    mask = plot_result_values['recall_charac_mask'][batch, :end_time]
    raw_out = plot_result_values['out_plot'][batch, :end_time]
    presentation_steps = np.arange(raw_out.shape[0])
    # processed target
    data = plot_result_values['target_nums'][batch, :end_time].astype(float)
    data[data == 2] = 0.5
    line_target, = ax.plot(presentation_steps[:], data[:], color='black', label='target', alpha=0.7)
    # processed output
    output0 = 1. - raw_out[:, 0]
    output1 = raw_out[:, 1]
    combined_output = (output0 + output1)/2
    presentation_steps = np.arange(combined_output.shape[0])
    line_output2, = ax.plot(presentation_steps, combined_output, color='tab:green', label='output', alpha=0.7)
    line_handles = [line_output2, line_target]

    ax.set_yticks([0, 1])
    ax.set_ylabel('Output', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    ax.axis([0, presentation_steps[-1] + 1, -0.3, 1.1])
    ax.legend(handles=line_handles, loc='lower left', fontsize=7, ncol=len(line_handles))

    ax.set_xlabel('time in ms', fontsize=fs)
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(1)


def offline_plot(data_path, custom_plot=False):
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
    nrows = 6
    height = nrows
    fig, ax_list = plt.subplots(nrows=nrows, figsize=(6, height), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    for b in range(flags.batch_test):
        gocue_input = plot_result_values['input_spikes'][b, :, 1]
        target = plot_result_values['target_nums'][b]
        if np.nonzero(gocue_input)[0][0] < 400 and target[target != 2].size > 0:
            continue
        update_plot(plt, ax_list, flags, plot_result_values, batch=b)
        start_time = datetime.datetime.now()
        fig.savefig(os.path.join(data_path, 'figure_test' + str(b) + '_' + start_time.strftime("%H%M") + '.pdf'),
                    format='pdf')
