from bin.tutorial_storerecall_utils import generate_poisson_noise_np
from bin.tutorial_temporalXOR_utils import generate_xor_input, generate_xor_spike_input
from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik
# import matplotlib
# matplotlib.use('Agg')

import datetime
import os
import socket
from time import time

import matplotlib.pyplot as plt
from matplotlib import collections as mc, patches
import numpy as np
import numpy.random as rd
import tensorflow as tf
from lsnn.guillaume_toolbox.file_saver_dumper_no_h5py import save_file
from lsnn.guillaume_toolbox.matplotlib_extension import strip_right_top_axis, raster_plot
from tutorial_storerecall_utils import generate_storerecall_data, error_rate, gen_custom_delay_batch

from lsnn.guillaume_toolbox.tensorflow_utils import tf_downsample
from lsnn.spiking_models import tf_cell_to_savable_dict, placeholder_container_for_rnn_state,\
    feed_dict_with_placeholder_container, exp_convolve, ALIF
from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
##
tf.app.flags.DEFINE_integer('batch_train', 128, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('batch_val', 128, 'batch size of the validation set')
tf.app.flags.DEFINE_integer('batch_test', 128, 'batch size of the testing set')
tf.app.flags.DEFINE_integer('n_charac', 2, 'number of characters in the recall task')
tf.app.flags.DEFINE_integer('n_in', 3, 'number of input units.')
tf.app.flags.DEFINE_integer('n_regular', 80, 'number of recurrent units.')
tf.app.flags.DEFINE_integer('n_adaptive', 80, 'number of controller units')
tf.app.flags.DEFINE_integer('f0', 50, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('reg_max_rate', 100, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 200, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 10, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 3, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('seq_len', 16, 'Number of character steps')
tf.app.flags.DEFINE_integer('seq_delay', 1, 'Expected delay in character steps. Must be <= seq_len - 2')
tf.app.flags.DEFINE_integer('tau_char', 50, 'Duration of symbols')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('lr_decay_every', 40, 'Decay every')
tf.app.flags.DEFINE_integer('print_every', 20, 'Decay every')
tf.app.flags.DEFINE_integer('xor_delay', 100, 'Expected delay in character steps. Must be <= seq_len - 2')
##
tf.app.flags.DEFINE_float('stop_crit', 0.05, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_float('beta', 1.7, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 200, 'Mikolov model alpha - threshold decay')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in LSNN and output neurons')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 1e-2, 'regularization coefficient')
tf.app.flags.DEFINE_float('rewiring_connectivity', -1, 'possible usage of rewiring with ALIF and LIF (0.1 is default)')
tf.app.flags.DEFINE_float('readout_rewiring_connectivity', -1, '')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring')
tf.app.flags.DEFINE_float('rewiring_temperature', 0, '')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, '')
tf.app.flags.DEFINE_float('stochastic_factor', -1, '')
tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', .01, 'threshold at which the LSNN neurons spike')
##
tf.app.flags.DEFINE_bool('tau_a_spread', False, 'Mikolov model spread of alpha - threshold decay')
tf.app.flags.DEFINE_bool('save_data', True, 'Save the data (training, test, network, trajectory for plotting)')
tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('monitor_plot', False, 'Perform plots during training')
tf.app.flags.DEFINE_bool('interactive_plot', False, 'Perform plots')
tf.app.flags.DEFINE_bool('device_placement', False, '')
tf.app.flags.DEFINE_bool('verbose', True, '')
tf.app.flags.DEFINE_bool('neuron_sign', True, '')
tf.app.flags.DEFINE_bool('adaptive_reg', False, '')

# Run asserts to check seq_delay and seq_len relation is ok
_ = gen_custom_delay_batch(FLAGS.seq_len, FLAGS.seq_delay, 1)

# Fix the random seed if given as an argument
if FLAGS.seed >= 0:
    seed = FLAGS.seed
else:
    seed = rd.randint(10 ** 6)
rd.seed(seed)
tf.set_random_seed(seed)

# Experiment parameters
dt = 1.
repeat_batch_test = 10
print_every = FLAGS.print_every

# Frequencies
input_f0 = FLAGS.f0 / 1000  # in kHz in coherence with the usgae of ms for time
regularization_f0 = FLAGS.reg_rate / 1000
regularization_f0_max = FLAGS.reg_max_rate / 1000

# Network parameters
tau_v = FLAGS.tau_out
thr = FLAGS.thr

decay = np.exp(-dt / FLAGS.tau_out)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
# Symbol number
n_charac = FLAGS.n_charac  # Number of digit symbols
n_input_symbols = n_charac + 2  # Total number of symbols including recall and store
n_output_symbols = n_charac  # Number of output symbols
recall_symbol = n_input_symbols - 1  # ID of the recall symbol
store_symbol = n_input_symbols - 2  # ID of the store symbol

# Neuron population sizes
input_neuron_split = np.array_split(np.arange(FLAGS.n_in), n_input_symbols)

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(0.75 * FLAGS.n_in)
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    np.random.shuffle(in_neuron_sign)

    n_excitatory = int(0.75 * (FLAGS.n_regular + FLAGS.n_adaptive))
    n_inhibitory = FLAGS.n_regular + FLAGS.n_adaptive - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not FLAGS.neuron_sign: print('WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Generate the cell
beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v, n_delay=FLAGS.n_delay,
            n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=FLAGS.tau_a, beta=beta, thr=thr,
            rewiring_connectivity=FLAGS.rewiring_connectivity,
            in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
            dampening_factor=FLAGS.dampening_factor,
            )

cell_name = type(cell).__name__
print('\n -------------- \n' + cell_name + '\n -------------- \n')
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_reference = '{}_{}_seqlen{}_seqdelay{}_in{}_R{}_A{}_lr{}_tauchar{}_comment{}'.format(
    time_stamp, cell_name, FLAGS.seq_len, FLAGS.seq_delay, FLAGS.n_in, FLAGS.n_regular, FLAGS.n_adaptive,
    FLAGS.learning_rate, FLAGS.tau_char, FLAGS.comment)
file_reference = file_reference + '_taua' + str(FLAGS.tau_a) + '_beta' + str(FLAGS.beta)
print('FILE REFERENCE: ' + file_reference)

# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
target_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                             name='TargetNums')  # Lists of target characters of the recall task
recall_mask = tf.placeholder(dtype=tf.bool, shape=(None, None),
                             name='RecallMask')  # Binary tensor that points to the time of presentation of a recall

# Other placeholder that are useful for computing accuracy and debuggin
target_sequence = tf.placeholder(dtype=tf.int64, shape=(None),
                                 name='TargetSequence')  # The target characters with time expansion
batch_size_holder = tf.placeholder(dtype=tf.int32, name='BatchSize')  # Int that contains the batch size
init_state_holder = placeholder_container_for_rnn_state(cell.state_size, dtype=tf.float32, batch_size=None)
# recall_charac_mask = tf.equal(input_nums, recall_symbol, name='RecallCharacMask')
recall_charac_mask = recall_mask


def get_data_dict(batch_size, pulse_delay=FLAGS.xor_delay):
    input_data, target_data, target_mask, targets = generate_xor_spike_input(
        batch_size=batch_size, length=FLAGS.seq_len * FLAGS.tau_char, expected_delay=pulse_delay)
    input_rates = input_data * FLAGS.f0
    spikes = generate_poisson_noise_np(input_rates)
    data_dict = {input_spikes: spikes, target_nums: target_data, recall_mask: target_mask,
                 target_sequence: targets, batch_size_holder: batch_size}

    return data_dict

# Define the name of spike train for the different models
z_stack, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=init_state_holder, dtype=tf.float32)
z, b_con = z_stack
z_con = []
z_all = z

with tf.name_scope('RecallLoss'):
    target_nums_at_recall = tf.boolean_mask(target_nums, recall_charac_mask)
    Y = tf.one_hot(target_nums_at_recall, depth=n_output_symbols, name='Target')

    # MTP models do not use controller (modulator) population for output
    out_neurons = z_all
    n_neurons = out_neurons.get_shape()[2]
    psp = exp_convolve(out_neurons, decay=decay)

    if 0 < FLAGS.rewiring_connectivity and 0 < FLAGS.readout_rewiring_connectivity:
        w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, n_output_symbols,
                                                         FLAGS.readout_rewiring_connectivity,
                                                         neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])

    out = einsum_bij_jk_to_bik(psp, w_out)
    # out_char_step = tf_downsample(out, new_size=FLAGS.seq_len, axis=1)
    # Y_predict = tf.boolean_mask(out_char_step, recall_charac_mask, name='Prediction')
    Y_predict = tf.boolean_mask(out, recall_charac_mask, name='Prediction')
    # Y_predict = tf.reduce_mean(Y_predict, axis=1)
    # Y = tf.reduce_mean(target_nums_at_recall, axis=1)

    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_nums_at_recall,
                                                                                logits=Y_predict))

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)
        # out_plot_char_step = tf_downsample(out_plot, new_size=FLAGS.seq_len, axis=1)

    # _, recall_errors, false_sentence_id_list = error_rate(out_char_step, target_nums, input_nums, n_charac)
    Y_predict_hard = tf.argmax(Y_predict, axis=1)
    all_accuracy = tf.cast(tf.equal(Y_predict_hard, tf.argmax(Y, axis=1)), dtype=tf.float32)
    recall_accuracy = tf.reduce_mean(all_accuracy)
    recall_errors = 1. - recall_accuracy

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z_all, axis=(0, 1)) / dt
    adaptive_regularization_coeff = tf.Variable(np.ones(n_neurons) * FLAGS.reg, dtype=tf.float32, trainable=False)

    loss_reg = tf.reduce_sum(tf.square(av - regularization_f0) * adaptive_regularization_coeff)

    do_increase_reg = tf.greater(av,regularization_f0_max)
    do_increase_reg = tf.cast(do_increase_reg,dtype=tf.float32)

    new_adaptive_coeff = do_increase_reg * adaptive_regularization_coeff * 1.3 \
                         + (1-do_increase_reg) * adaptive_regularization_coeff * 0.93

    if FLAGS.adaptive_reg:
        update_regularization_coeff = tf.assign(adaptive_regularization_coeff,new_adaptive_coeff)
    else:
        update_regularization_coeff = tf.no_op('SkipAdaptiveRegularization')

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    # scaling loss_recall to match order of magnitude of loss from script_recall.py
    # this is needed to keep the same regularization coefficients (reg, regl2) across scripts
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)

    loss = loss_reg + loss_recall

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if 0 < FLAGS.rewiring_connectivity:

        rewiring_connectivity_list = [FLAGS.rewiring_connectivity, FLAGS.rewiring_connectivity,
                                      FLAGS.readout_rewiring_connectivity]

        train_step = rewiring_optimizer_wrapper(opt, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                                rewiring_connectivity_list,
                                                global_step=global_step,
                                                var_list=tf.trainable_variables())
    else:
        train_step = opt.minimize(loss=loss, global_step=global_step)

# Real-time plotting
sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.device_placement))
sess.run(tf.global_variables_initializer())

last_final_state_state_training_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32))]
last_final_state_state_validation_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_val, dtype=tf.float32))]
last_final_state_state_testing_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_test, dtype=tf.float32))]

# Open an interactive matplotlib window to plot in real time
if FLAGS.do_plot and FLAGS.interactive_plot:
    plt.ion()
if FLAGS.do_plot:
    fig, ax_list = plt.subplots(4, figsize=(5.9, 6))

    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.set_window_title(socket.gethostname() + ' - ' + FLAGS.comment)


def update_plot(plot_result_values, batch=0, n_max_neuron_per_raster=20, n_max_synapses=FLAGS.n_adaptive):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    fs = 12
    plt.rcParams.update({'font.size': fs})
    ylabel_x = -0.09
    ylabel_y = 0.5
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # PLOT Input signals
    ax = ax_list[0]
    data = plot_result_values['input_spikes']
    data = data[batch]
    raster_plot(ax, data, linewidth=0.3)
    ax.set_xlim([0, len(data)])
    ax.set_ylabel('')
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    ax.set_xticks([])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['Input 0', 'Input 1', 'Go-cue'], fontsize=fs-4)

    # PLOT OUTPUT AND TARGET
    ax = ax_list[1]
    mask = plot_result_values['recall_charac_mask'][batch]
    data = plot_result_values['target_nums'][batch]
    presentation_steps = np.arange(data.shape[0])
    line_target, = ax.plot(presentation_steps[mask], data[mask], color='blue', label='Target', alpha=0.7)
    # data = plot_result_values['out_plot'][batch, :, 1]
    # out_avg = data[mask]
    # out_avg = np.ones_like(out_avg) * np.mean(out_avg)
    # line_out, = ax.plot(presentation_steps[mask], out_avg, color='blue', label='avg.out.', alpha=0.7)
    # plot softmax of psp-s per dt for more intuitive monitoring
    # ploting only for second class since this is more intuitive to follow (first class is just a mirror)
    output2 = plot_result_values['out_plot'][batch, :, 1]
    presentation_steps = np.arange(output2.shape[0])
    ax.set_yticks([0, 0.5, 1])
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Output')
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    line_output2, = ax.plot(presentation_steps, output2, color='green', label='Output', alpha=0.7)
    ax.axis([0, presentation_steps[-1] + 1, -0.3, 1.1])
    ax.legend(handles=[line_output2, line_target], loc='lower left', fontsize=7, ncol=3)
    ax.set_xticklabels([])

    # PLOT SPIKES
    ax = ax_list[2]
    data = plot_result_values['z']
    data = data[batch]
    raster_plot(ax, data, linewidth=0.3)
    ax.set_ylabel('LSNN')
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    ax.set_xticklabels([])

    # debug plot for psp-s or biases
    plot_param = 'b_con'  # or 'psp'
    ax.set_xticklabels([])
    ax = ax_list[-1]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Threshold')
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    sub_data = plot_result_values['b_con'][batch]
    sub_data = sub_data + thr
    vars = np.var(sub_data, axis=0)
    # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses * 3:3]
    cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses]
    presentation_steps = np.arange(sub_data.shape[0])
    ax.plot(sub_data[:, cell_with_max_var], color='r', label='Output', alpha=0.4, linewidth=1)
    ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
                 np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]

    ax.set_xlabel('Time in ms')
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(1)

test_loss_list = []
test_loss_with_reg_list = []
validation_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []
results_tensors = {
    'loss': loss,
    'loss_reg': loss_reg,
    'loss_recall': loss_recall,
    'recall_errors': recall_errors,
    'final_state': final_state,
    'av': av,
    'adaptive_regularization_coeff': adaptive_regularization_coeff,
}

results_tensors['w_in_val'] = cell.w_in_val
results_tensors['w_rec_val'] = cell.w_rec_val
results_tensors['w_out'] = w_out

w_in_last = sess.run(cell.w_in_val)
w_rec_last = sess.run(cell.w_rec_val)
w_out_last = sess.run(w_out)

plot_result_tensors = {'input_spikes': input_spikes,
                       'z': z,
                       'z_con': z_con,
                       'target_nums': target_nums,
                       # 'out_avg_per_step': out_plot_char_step,
                       }
t_train = 0
t_ref = time()
# current_delay = 50
current_delay = FLAGS.xor_delay
for k_iter in range(FLAGS.n_iter):

    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Monitor the training with a validation set
    t0 = time()
    val_dict = get_data_dict(FLAGS.batch_val)
    feed_dict_with_placeholder_container(val_dict, init_state_holder, last_final_state_state_validation_pointer[0])

    plot_result_tensors['psp'] = psp
    # plot_result_tensors['out_plot_char_step'] = out_plot_char_step
    plot_result_tensors['out_plot'] = out_plot
    plot_result_tensors['recall_charac_mask'] = recall_charac_mask
    plot_result_tensors['Y'] = Y
    plot_result_tensors['Y_predict'] = Y_predict

    plot_result_tensors['b_con'] = b_con

    results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)
    last_final_state_state_validation_pointer[0] = results_values['final_state']
    last_final_state_state_testing_pointer[0] = results_values['final_state']
    t_run = time() - t0

    # Storage of the results
    test_loss_with_reg_list.append(results_values['loss_reg'])
    test_loss_list.append(results_values['loss_recall'])
    validation_error_list.append(results_values['recall_errors'])
    training_time_list.append(t_train)
    time_to_ref_list.append(time() - t_ref)

    if np.mod(k_iter, print_every) == 0:

        print('''Iteration {}, statistics on the validation set average error {:.2g} +- {:.2g} (trial averaged)'''
              .format(k_iter, np.mean(validation_error_list[-print_every:]),
                      np.std(validation_error_list[-print_every:])))

        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max

        firing_rate_stats = get_stats(results_values['av'] * 1000)
        reg_coeff_stats = get_stats(results_values['adaptive_regularization_coeff'])

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t
            average {:.0f} +- std {:.0f} (averaged over batches and time)
            reg. coeff        min {:.2g} \t max {:.2g} \t average {:.2g} +- std {:.2g}

            comput. time (s)  training {:.2g} \t validation {:.2g}
            loss              classif. {:.2g} \t reg. loss  {:.2g}
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                reg_coeff_stats[0], reg_coeff_stats[1], reg_coeff_stats[2], reg_coeff_stats[3],
                t_train, t_run,
                results_values['loss_recall'], results_values['loss_reg']
            ))

        if 0 < FLAGS.rewiring_connectivity:
            rewired_ref_list = ['w_in_val','w_rec_val','w_out']
            non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
            sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
            empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)

            if 0 < FLAGS.rewiring_connectivity:
                assert empirical_connectivity < FLAGS.rewiring_connectivity * 1.1,\
                    'Rewiring error: found connectivity {:.3g}'.format(empirical_connectivity)

            w_in_new = results_values['w_in_val']
            w_rec_new = results_values['w_rec_val']
            w_out_new = results_values['w_out']

            stay_con_in = np.logical_and(w_in_new != 0, w_in_last != 0)
            stay_con_rec = np.logical_and(w_rec_new != 0, w_rec_last != 0)
            stay_con_out = np.logical_and(w_out_new != 0, w_out_last != 0)

            Dw_in = np.linalg.norm(w_in_new[stay_con_in] - w_in_last[stay_con_in])
            Dw_rec = np.linalg.norm(w_rec_new[stay_con_rec] - w_rec_last[stay_con_rec])
            Dw_out = np.linalg.norm(w_out_new[stay_con_out] - w_out_last[stay_con_out])

            if FLAGS.verbose:
                print('''Connectivity {:.3g} \t Non zeros: W_in {}/{} W_rec {}/{} w_out {}/{} \t
                New zeros: W_in {} W_rec {} W_out {}'''.format(
                    empirical_connectivity,
                    non_zeros[0], sizes[0],
                    non_zeros[1], sizes[1],
                    non_zeros[2], sizes[2],
                    np.sum(np.logical_and(w_in_new == 0, w_in_last != 0)),
                    np.sum(np.logical_and(w_rec_new == 0, w_rec_last != 0)),
                    np.sum(np.logical_and(w_out_new == 0, w_out_last != 0))
                ))

                print('Delta W norms: {:.3g} \t {:.3g} \t {:.3g}'.format(Dw_in,Dw_rec,Dw_out))

            w_in_last = results_values['w_in_val']
            w_rec_last = results_values['w_rec_val']
            w_out_last = results_values['w_out']

        if FLAGS.do_plot and FLAGS.monitor_plot:
            update_plot(plot_results_values)
            tmp_path = os.path.join(result_folder,
                                    'tmp/figure' + start_time.strftime("%H%M") + '_' +
                                    str(k_iter) + '.pdf')
            if not os.path.exists(os.path.join(result_folder, 'tmp')):
                os.makedirs(os.path.join(result_folder, 'tmp'))
            fig.savefig(tmp_path, format='pdf')

        if np.mean(validation_error_list[-print_every:]) < FLAGS.stop_crit:
            print('LESS THAN ' + str(FLAGS.stop_crit) + ' ERROR ACHIEVED - STOPPING - SOLVED at epoch ' + str(k_iter))
            break

    # train
    train_dict = get_data_dict(FLAGS.batch_train, pulse_delay=current_delay)
    feed_dict_with_placeholder_container(train_dict, init_state_holder, last_final_state_state_training_pointer[0])
    t0 = time()
    final_state_value, _, _, train_err = sess.run([final_state, train_step, update_regularization_coeff, recall_errors],
                                                  feed_dict=train_dict)
    # print("train_err ", train_err)
    if train_err < 0.1 and current_delay < FLAGS.xor_delay:
        current_delay += 50
        print("increased train delay to ", current_delay)
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    last_final_state_state_training_pointer[0] = final_state_value
    t_train = time() - t0

print('FINISHED IN {:.2g} s'.format(time() - t_ref))

# Save everything
if FLAGS.save_data:

    # Saving setup
    full_path = os.path.join(result_folder, file_reference)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Save the tensorflow graph
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(full_path, 'session'))
    saver.export_meta_graph(os.path.join(full_path, 'graph.meta'))

    # Save parameters and training log
    try:
        flag_dict = FLAGS.flag_values_dict()
    except:
        print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
        flag_dict = FLAGS.__flags

    results = {
        'error': validation_error_list[-1],
        'loss': test_loss_list[-1],
        'loss_with_reg': test_loss_with_reg_list[-1],
        'loss_with_reg_list': test_loss_with_reg_list,
        'error_list': validation_error_list,
        'loss_list': test_loss_list,
        'time_to_ref': time_to_ref_list,
        'training_time': training_time_list,
        'tau_delay_list': tau_delay_list,
        'flags': flag_dict,
    }

    save_file(flag_dict, full_path, 'flag', file_type='json')
    save_file(results, full_path, 'training_results', file_type='json')

    # Save sample trajectory (input, output, etc. for plotting)
    test_errors = []
    for i in range(16):
        test_dict = get_data_dict(FLAGS.batch_test)
        feed_dict_with_placeholder_container(test_dict, init_state_holder, sess.run(
            cell.zero_state(batch_size=FLAGS.batch_test, dtype=tf.float32)))

        results_values, plot_results_values, in_spk, spk, spk_con, target_nums_np = sess.run(
            [results_tensors, plot_result_tensors, input_spikes, z, z_con, target_nums],
            feed_dict=test_dict)
        # last_final_state_state_testing_pointer[0] = results_values['final_state']
        test_errors.append(results_values['recall_errors'])

    print('''Statistics on the test set average error {:.2g} +- {:.2g} (averaged over 16 test batches of size {})'''
          .format(np.mean(test_errors), np.std(test_errors), FLAGS.batch_test))
    save_file(plot_results_values, full_path, 'plot_trajectory_data', 'pickle')

    # Save test results
    results = {
        'test_errors': test_errors,
        'test_errors_mean': np.mean(test_errors),
        'test_errors_std': np.std(test_errors),
    }
    save_file(results, full_path, 'test_results', file_type='json')
    print("saved test_results.json")
    # Save network variables (weights, delays, etc.)
    network_data = tf_cell_to_savable_dict(cell, sess)
    network_data['w_out'] = results_values['w_out']
    save_file(network_data, full_path, 'tf_cell_net_data', file_type='pickle')

del sess
