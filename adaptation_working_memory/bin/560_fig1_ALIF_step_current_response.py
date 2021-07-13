"""
Reproduce the 560 fig1 panel B with:
PYTHONPATH=. python3 bin/560_fig1_ALIF_step_current_response.py --beta=1.
PYTHONPATH=. python3 bin/560_fig1_ALIF_step_current_response.py --beta=0.5
PYTHONPATH=. python3 bin/560_fig1_ALIF_step_current_response.py --beta=1. --tau_a=300 --input_current=0.022
"""
# from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik, einsum_bi_ijk_to_bjk
# import matplotlib
# matplotlib.use('Agg')

import datetime
import os
import socket
from time import time
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import collections as mc, patches
import numpy as np
import numpy.random as rd
import tensorflow as tf

from bin.tutorial_storerecall_utils import generate_poisson_noise_np
from lsnn.guillaume_toolbox.file_saver_dumper_no_h5py import save_file
from lsnn.guillaume_toolbox.matplotlib_extension import strip_right_top_axis, raster_plot
from tutorial_storerecall_utils import generate_storerecall_data, error_rate, gen_custom_delay_batch

from lsnn.guillaume_toolbox.tensorflow_utils import tf_downsample, tf_roll
from lsnn.spiking_models import tf_cell_to_savable_dict, placeholder_container_for_rnn_state, \
    feed_dict_with_placeholder_container, exp_convolve, weight_matrix_with_delay_dimension, LIFStateTuple, \
    SpikeFunction, ALIF, ALIFStateTuple, LIF
from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()

Cell = tf.contrib.rnn.BasicRNNCell

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
##
tf.app.flags.DEFINE_integer('batch_train', 32, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('batch_val', 32, 'batch size of the validation set')
tf.app.flags.DEFINE_integer('batch_test', 32, 'batch size of the testing set')
tf.app.flags.DEFINE_integer('n_charac', 2, 'number of characters in the recall task')
tf.app.flags.DEFINE_integer('n_in', 1, 'number of input units.')
tf.app.flags.DEFINE_integer('n_regular', 0, 'number of recurrent units.')
tf.app.flags.DEFINE_integer('n_adaptive', 1, 'number of controller units')
tf.app.flags.DEFINE_integer('f0', 50, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 40, 'target rate for regularization')
tf.app.flags.DEFINE_integer('reg_max_rate', 100, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 1, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 5, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 5, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('seq_len', 1200, 'Number of character steps')
tf.app.flags.DEFINE_integer('seq_delay', 0, 'Expected delay in character steps. Must be <= seq_len - 2')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('lr_decay_every', 100, 'Decay every')
tf.app.flags.DEFINE_integer('print_every', 1, 'Decay every')
##
tf.app.flags.DEFINE_float('stop_crit', 0.5, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_float('beta', 1.8, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 1000, 'Mikolov model alpha - threshold decay')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in LSNN and output neurons')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.3, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 1, 'regularization coefficient')
tf.app.flags.DEFINE_float('rewiring_connectivity', -1, 'possible usage of rewiring with ALIF and LIF (0.1 is default)')
tf.app.flags.DEFINE_float('readout_rewiring_connectivity', -1, '')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring')
tf.app.flags.DEFINE_float('rewiring_temperature', 0, '')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, '')
tf.app.flags.DEFINE_float('stochastic_factor', -1, '')
tf.app.flags.DEFINE_float('V0', 1, 'unit scaling for LSNN model')
tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', .02, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('thr_min', .005, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('input_current', 0.024, 'input current to the adaptive neuron')
tf.app.flags.DEFINE_float('injected_noise_current', 0.0, 'input current to the adaptive neuron')
##
tf.app.flags.DEFINE_bool('tau_a_spread', False, 'Mikolov model spread of alpha - threshold decay')
tf.app.flags.DEFINE_bool('save_data', True, 'Save the full trajectory data on the testing set')
tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('monitor_plot', True, 'Perform plots during training')
tf.app.flags.DEFINE_bool('interactive_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('rec_to_con', True, 'Hidden units connected to context units in Mikolov')
tf.app.flags.DEFINE_bool('rec_con', False, 'Recurrent context units in Mikolov')
tf.app.flags.DEFINE_bool('device_placement', False, '')
tf.app.flags.DEFINE_bool('verbose', False, '')
tf.app.flags.DEFINE_bool('neuron_sign', True, '')
tf.app.flags.DEFINE_bool('adaptive_reg', False, '')
tf.app.flags.DEFINE_bool('step', True, 'Modulate the input current with a sine wave')
tf.app.flags.DEFINE_bool('bio_interpret', False, 'Plotting: subtract adaptive thr component from memb. potential')
FLAGS.thr = FLAGS.thr * FLAGS.V0  # scaling the threshold too!

# A --n_regular=0 --n_adaptive=1 --input_current=0.05 --injected_noise_current=0.3
# R --n_regular=1 --n_adaptive=0 --input_current=0.05 --injected_noise_current=0.3 --thr=0.05


class ALIFv(ALIF):
    def __init__(self, n_in, n_rec, tau=20, thr=0.01,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=5,
                 tau_adaptation=200., beta=1.6,
                 rewiring_connectivity=-1, dampening_factor=0.3,
                 in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
                 V0=1., trainable=True, add_current=0.):
        LIF.__init__(self, n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt, n_refractory=n_refractory,
                     dtype=dtype, n_delay=n_delay,
                     rewiring_connectivity=rewiring_connectivity,
                     dampening_factor=dampening_factor, in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                     injected_noise_current=injected_noise_current,
                     )

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)
        self.add_current = add_current

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec, self.n_rec, self.n_rec]

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        # i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
        i_future_buffer = state.i_future_buffer + tf.einsum('bi,ijk->bjk', inputs, self.W_in) + tf.einsum('bi,ijk->bjk',
            state.z, self.W_rec)

        new_b = self.decay_b * state.b + (np.ones(self.n_rec) - self.decay_b) * state.z

        thr = tf.maximum(self.thr + new_b * self.beta * self.V0,
                         tf.ones_like(new_b, dtype=dtype) * tf.cast(FLAGS.thr_min, dtype=dtype))

        new_v, new_z, input_current = self.LIF_dynamic(
            v=state.v,
            z=state.z,
            z_buffer=state.z_buffer,
            i_future_buffer=i_future_buffer,
            decay=self._decay,
            thr=thr,
            add_current=self.add_current,
        )

        new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
        new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

        new_state = ALIFStateTuple(v=new_v,
                                   z=new_z,
                                   b=new_b,
                                   i_future_buffer=new_i_future_buffer,
                                   z_buffer=new_z_buffer)
        return [new_z, thr, new_v, input_current], new_state

    def LIF_dynamic(self, v, z, z_buffer, i_future_buffer, thr=None, decay=None, n_refractory=None, add_current=0.):
        if self.injected_noise_current > 0:
            add_current = add_current + tf.random_normal(shape=tf.shape(z), stddev=self.injected_noise_current)

        with tf.name_scope('LIFdynamic'):
            if thr is None: thr = self.thr
            if decay is None: decay = self._decay
            if n_refractory is None: n_refractory = self.n_refractory

            i_t = i_future_buffer[:, :, 0] + add_current

            I_reset = z * thr * self.dt

            new_v = decay * v + (1 - decay) * i_t - I_reset

            # Spike generation
            v_scaled = (v - thr) / thr

            # new_z = differentiable_spikes(v_scaled=v_scaled)
            new_z = SpikeFunction(v_scaled, self.dampening_factor)
            new_z.set_shape([None, v.get_shape()[1]])

            if n_refractory > 0:
                is_ref = tf.greater(tf.reduce_max(z_buffer[:, :, -n_refractory:], axis=2), 0)
                new_z = tf.where(is_ref, tf.zeros_like(new_z), new_z)

            new_z = new_z * 1/ self.dt

            return new_v, new_z, i_t

# Fix the random seed if given as an argument
if FLAGS.seed >= 0:
    seed = FLAGS.seed
else:
    seed = rd.randint(10 ** 6)
rd.seed(seed)
tf.set_random_seed(seed)

# Experiment parameters
dt = FLAGS.dt
print_every = FLAGS.print_every

# Frequencies
input_f0 = FLAGS.f0 / 1000 # in kHz in coherence with the usgae of ms for time
regularization_f0 = FLAGS.reg_rate / 1000
regularization_f0_max = FLAGS.reg_max_rate / 1000

N_batch_plot = 1
plot_sentence = None  # The interactive plot will render a different sequence every time
# plot_sentence = [2, 1, 0, 3, 0, 2, 0, 1, 1, 0, 1, 3] # The interactive plots only plots your favourite sequence
# plot_sentence = [plot_sentence[:FLAGS.seq_len]] # Truncate the sequence to the length expected

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
if FLAGS.tau_a_spread:
    tau_a_spread = np.random.uniform(size=FLAGS.n_regular+FLAGS.n_adaptive) * FLAGS.tau_a
else:
    tau_a_spread = FLAGS.tau_a
beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
cell = ALIFv(
    n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v, n_delay=FLAGS.n_delay,
    n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_spread, beta=beta, thr=thr,
    rewiring_connectivity=FLAGS.rewiring_connectivity,
    in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
    dampening_factor=FLAGS.dampening_factor,
    add_current=0., injected_noise_current=FLAGS.injected_noise_current
)

cell_name = type(cell).__name__
print('\n -------------- \n' + cell_name + '\n -------------- \n')
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_reference = '{}_{}_in{}_R{}_A{}__Iin{:.0f}_V0{:.0f}_comment_{}'.format(
    time_stamp, cell_name, FLAGS.n_in, FLAGS.n_regular, FLAGS.n_adaptive, FLAGS.input_current, FLAGS.V0, FLAGS.comment)
file_reference = '{}_ALIF_{}_{}'.format(time_stamp, 'step' if FLAGS.step else 'spike', FLAGS.comment)
print('FILE REFERENCE: ' + file_reference)

# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
batch_size_holder = tf.placeholder(dtype=tf.int32, name='BatchSize')  # Int that contains the batch size
init_state_holder = placeholder_container_for_rnn_state(cell.state_size, dtype=tf.float32, batch_size=None)


def get_data_dict(batch_size, seq_len=FLAGS.seq_len):
    spikes = np.zeros((batch_size, seq_len, FLAGS.n_in))
    if FLAGS.step:
        # phase = 0
        # period = 1000
        # sine = np.sin(np.linspace(0 + phase, np.pi * 2 * (FLAGS.seq_len // period) + phase, FLAGS.seq_len))
        # sine = np.tile(np.expand_dims(sine, axis=1), (batch_size, 1, 1))
        step = np.ones(seq_len)
        step[0:100] = 0
        step[-100:] = 0
        step = np.tile(np.expand_dims(step, axis=1), (batch_size, 1, 1))

        spikes = step * FLAGS.input_current
    else:
        spikes = np.zeros(seq_len)
        spikes[90:320] = 1/7
        spikes[200:390] = np.ones_like(spikes[200:390]) * 1/10
        spikes[350:820] = np.ones_like(spikes[350:820]) * 1/14
        spikes[700:790] = np.ones_like(spikes[700:790]) * 1/3
        spikes[700:1100] = np.ones_like(spikes[700:1100]) * 1/10
        spikes = generate_poisson_noise_np(spikes)
        spikes = np.tile(np.expand_dims(spikes, axis=1), (batch_size, 1, 1))
        spikes = spikes * FLAGS.input_current * 2
    # spikes[0, 10:20:5, 0] = 1  # 4 spikes on input
    data_dict = {input_spikes: spikes, batch_size_holder: batch_size}
    return data_dict

# Define the name of spike train for the different models
z_stack, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=init_state_holder, dtype=tf.float32)
z, b_con, v, input_current = z_stack
z_con = []
n_neurons = z.get_shape()[2]

# with tf.name_scope('RecallLoss'):
#     # MTP models do not use controller (modulator) population for output
#     out_neurons = z
#     n_neurons = out_neurons.get_shape()[2]
#     psp = exp_convolve(out_neurons, decay=decay)
#
#     if 0 < FLAGS.rewiring_connectivity and 0 < FLAGS.readout_rewiring_connectivity:
#         w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, n_output_symbols,
#                                                          FLAGS.readout_rewiring_connectivity,
#                                                          neuron_sign=rec_neuron_sign)
#     else:
#         w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])
#
#     out = einsum_bij_jk_to_bik(psp, w_out)
#     out_char_step = tf_downsample(out, new_size=FLAGS.seq_len, axis=1)
#
#     with tf.name_scope('PlotNodes'):
#         out_plot = tf.nn.softmax(out)
#         out_plot_char_step = tf_downsample(out_plot, new_size=FLAGS.seq_len, axis=1)


# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    loss_reg = tf.reduce_sum(tf.square(av - regularization_f0) * FLAGS.reg)
    err = tf.reduce_mean(tf.abs(av - regularization_f0)) * 1000


# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    # scaling loss_recall to match order of magnitude of loss from script_recall.py
    # this is needed to keep the same regularization coefficients (reg, regl2) across scripts
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if 0 < FLAGS.rewiring_connectivity:
        pass  # not yet implemented for rewiring
        #
        # rewiring_connectivity_list = [FLAGS.rewiring_connectivity, FLAGS.rewiring_connectivity,
        #                               FLAGS.readout_rewiring_connectivity]
        #
        # train_step = rewiring_optimizer_wrapper(opt, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
        #                                         rewiring_connectivity_list,
        #                                         global_step=global_step,
        #                                         all_trained_var_list=tf.trainable_variables())
    else:
        train_step = opt.minimize(loss=loss_reg, global_step=global_step)


# Real-time plotting
sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.device_placement))
sess.run(tf.global_variables_initializer())

if (FLAGS.n_regular + FLAGS.n_adaptive) == 1:
    set_w_in = tf.assign(cell.w_in_var,  np.array([[1]]))
    set_w_rec = tf.assign(cell.w_rec_var,  np.array([[0]]))
    sess.run([set_w_in, set_w_rec])

last_final_state_state_training_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32))]
last_final_state_state_validation_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_val, dtype=tf.float32))]
last_final_state_state_testing_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_test, dtype=tf.float32))]

# Open an interactive matplotlib window to plot in real time
if FLAGS.do_plot and FLAGS.interactive_plot:
    plt.ion()
if FLAGS.do_plot:
    fig, ax_list = plt.subplots(3, figsize=(6, 4))

    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.set_window_title(socket.gethostname() + ' - ' + FLAGS.comment)


def update_plot(plot_result_values, batch=0, n_max_neuron_per_raster=20, n_max_synapses=FLAGS.n_adaptive):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """

    def get_adaptation_index(spike_train):
        spike_times = np.where(spike_train)[0]
        ISI = np.diff(spike_times)
        diff_ISI = ISI[1:] - ISI[:-1]
        add_ISI = ISI[1:] + ISI[:-1]
        normed_diff_ISI = diff_ISI / add_ISI
        return np.mean(normed_diff_ISI)
    ylabel_x = -0.1
    ylabel_y = 0.5
    fs = 10
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # debug plot membrane potential
    ax = ax_list[0]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.spines['bottom'].set_visible(False)
    if FLAGS.step:
        sub_data = plot_result_values['input_current'][batch]
        presentation_steps = np.arange(sub_data.shape[0])
        ax.set_ylabel('input current', fontsize=fs)
        ax.plot(sub_data[:, :], color='k', label='input current', alpha=0.6, linewidth=1)
    else:
        sub_data = plot_result_values['input_spikes'][batch]
        presentation_steps = np.arange(sub_data.shape[0])
        ax.set_ylabel('input spikes', fontsize=fs)
        ax.set_yticks([])
        ax.set_yticklabels([])
        raster_plot(ax, sub_data[:, :], linewidth=0.3)

    # ax.set_yticks([FLAGS.input_current])
    ax.set_xticks([])
    ax.axis([0, presentation_steps[-1],
             0.0, 0.03])  # [xmin, xmax, ymin, ymax]
             # np.min(sub_data[:, :]), np.max(sub_data[:, :])])  # [xmin, xmax, ymin, ymax]
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)

    # Plot neuron spikes
    ax = ax_list[1]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.spines['bottom'].set_visible(False)
    data = plot_result_values['z']
    if np.size(data) > 0:
        data = data[batch]
        print("AI =", get_adaptation_index(data[:, 0]))
        n_max = min(data.shape[1], n_max_neuron_per_raster)
        cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
        data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
        raster_plot(ax, data, linewidth=0.3)
        ax.set_ylabel('neuron spikes', fontsize=fs)
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.axis([0, presentation_steps[-1], 0, 1])
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)

    # debug plot membrane potential
    ax = ax_list[-1]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    # ax.spines['bottom'].set_visible(False)
    sub_data = plot_result_values['v'][batch]
    vars = np.var(sub_data, axis=0)
    # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses * 3:3]
    # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses]
    presentation_steps = np.arange(sub_data.shape[0])
    # ax.plot(sub_data[:, cell_with_max_var], color='r', label='Output', alpha=0.6, linewidth=1)
    if FLAGS.bio_interpret:
        ax.set_ylabel('V(t) mV', fontsize=fs)
        adaptive_thr_data = plot_result_values['b_con'][batch] - FLAGS.thr
        sub_data = sub_data - adaptive_thr_data
    else:
        ax.set_ylabel('V(t), A(t) mV', fontsize=fs)
    ax.plot(sub_data[:, :], color='b', label='membrane potential V(t)', alpha=0.6, linewidth=1)
    # ax.axis([0, presentation_steps[-1], np.min(sub_data[:, :]), np.max(sub_data[:, :])])  # [xmin, xmax, ymin, ymax]
    ax.axis([0, presentation_steps[-1], 0.0, 0.03])  # [xmin, xmax, ymin, ymax]
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    temp = ax.get_yticks()  # [0.   0.01 0.02 0.03]
    temp = [str(int(t*1000)) for t in temp]
    ax.set_yticklabels(temp)

    # ax.set_xticks([])
    if not FLAGS.bio_interpret:
        # debug plot for psp-s or biases
        # ax.set_xticklabels([])
        # ax = ax_list[-1]
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        # ax.set_ylabel('PSPs' if plot_param == 'psp' else 'threshold', fontsize=fs)
        sub_data = plot_result_values['b_con'][batch]
        # if plot_param == 'b_con':
        #     sub_data = sub_data * FLAGS.V0 * beta + thr
        # vars = np.var(sub_data, axis=0)
        # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses * 3:3]
        # presentation_steps = np.arange(sub_data.shape[0])
        ax.plot(sub_data[:, :], color='r', label='threshold A(t)', alpha=0.4, linewidth=1)
        # ax.axis([0, presentation_steps[-1], np.min(sub_data[:, :]), np.max(sub_data[:, :])])  # [xmin, xmax, ymin, ymax]
        ax.legend()
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    temp = ax.get_xticks()  # [0.   0.01 0.02 0.03]
    temp_s = ["{:.1f}".format(t/1000) for t in temp]
    ax.set_xticks([temp[0], temp[-1]])
    ax.set_xticklabels([temp_s[0], temp_s[-1]])
    ax.set_xlabel('time (s)')
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.tight_layout()
        plt.draw()
        plt.pause(10)

test_loss_list = []
test_loss_with_reg_list = []
validation_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []
results_tensors = {
    'loss_reg': loss_reg,
    'final_state': final_state,
    'av': av,
    'err': err,
    'w_in_val': cell.w_in_val,
    'w_rec_val': cell.w_rec_val,
}

w_in_last = sess.run(cell.w_in_val)
w_rec_last = sess.run(cell.w_rec_val)

plot_result_tensors = {'input_spikes': input_spikes,
                       'v': v,
                       'z': z,
                       'z_con': z_con,
                       'input_current': input_current,
                       }
t_train = 0
t_ref = time()
if FLAGS.save_data:
    # Saving setup
    full_path = os.path.join(result_folder, file_reference)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

for k_iter in range(FLAGS.n_iter):

    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Monitor the training with a validation set
    t0 = time()
    val_dict = get_data_dict(FLAGS.batch_val)
    feed_dict_with_placeholder_container(val_dict, init_state_holder,
                                         sess.run(cell.zero_state(batch_size=FLAGS.batch_val, dtype=tf.float32)))

    plot_result_tensors['b_con'] = b_con

    results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)
    last_final_state_state_validation_pointer[0] = results_values['final_state']
    last_final_state_state_testing_pointer[0] = results_values['final_state']
    t_run = time() - t0

    # Storage of the results
    test_loss_with_reg_list.append(results_values['loss_reg'])
    validation_error_list.append(results_values['err'])
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

        print('''
        firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t
        average {:.0f} +- std {:.0f} (averaged over batches and time)
        comput. time (s)  training {:.2g} \t validation {:.2g}
        '''.format(
            firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
            firing_rate_stats[2], firing_rate_stats[3],
            t_train, t_run
        ))

        print("Input weight", w_in_last)

        if True:  # 0 < FLAGS.rewiring_connectivity or FLAGS.l1 > 0:
            w_in_new = results_values['w_in_val']
            w_rec_new = results_values['w_rec_val']
            rewired_ref_list = ['w_in_val', 'w_rec_val']
            non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
            sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
            empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)

            if 0 < FLAGS.rewiring_connectivity:
                assert empirical_connectivity < FLAGS.rewiring_connectivity * 1.1,\
                    'Rewiring error: found connectivity {:.3g}'.format(empirical_connectivity)

            if FLAGS.verbose:
                print('''Connectivity {:.3g} \t Non zeros: W_in {}/{} W_rec {}/{} \t
                New zeros: W_in {} W_rec {} '''.format(
                    empirical_connectivity,
                    non_zeros[0], sizes[0],
                    non_zeros[1], sizes[1],
                    np.sum(np.logical_and(w_in_new == 0, w_in_last != 0)),
                    np.sum(np.logical_and(w_rec_new == 0, w_rec_last != 0)),
                ))

                stay_con_in = np.logical_and(w_in_new !=0, w_in_last != 0)
                stay_con_rec = np.logical_and(w_rec_new !=0, w_rec_last != 0)

                Dw_in = np.linalg.norm(w_in_new[stay_con_in] - w_in_last[stay_con_in])
                Dw_rec = np.linalg.norm(w_rec_new[stay_con_rec] - w_rec_last[stay_con_rec])

                print('Delta W norms: {:.3g} \t {:.3g}'.format(Dw_in,Dw_rec))

            w_in_last = results_values['w_in_val']
            w_rec_last = results_values['w_rec_val']

        if FLAGS.do_plot and FLAGS.monitor_plot:
            update_plot(plot_results_values)
            if FLAGS.save_data:
                tmp_path = os.path.join(full_path,
                                        'figure_' + str(k_iter) + '.pdf')
                fig.savefig(tmp_path, format='pdf')
                save_file(plot_results_values, full_path, 'plot_trajectory_data_' + str(k_iter), 'pickle')

        if np.abs(FLAGS.reg_rate - np.mean(firing_rate_stats[2])) < FLAGS.stop_crit:
            print('LESS THAN ' + str(FLAGS.stop_crit) + ' ERROR ACHIEVED - STOPPING - SOLVED at epoch ' + str(k_iter))
            break

    # train
    train_dict = get_data_dict(FLAGS.batch_train)
    feed_dict_with_placeholder_container(train_dict, init_state_holder, last_final_state_state_training_pointer[0])
    t0 = time()
    final_state_value, _ = sess.run([final_state, train_step], feed_dict=train_dict)
    last_final_state_state_training_pointer[0] = final_state_value
    t_train = time() - t0

print('FINISHED IN {:.2g} s'.format(time() - t_ref))

# Save everything
if FLAGS.save_data:
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
    if type(tau_a_spread) is list:
        flag_dict['tauas'] = tau_a_spread
    results = {
        'time_to_ref': time_to_ref_list,
        'training_time': training_time_list,
        'tau_delay_list': tau_delay_list,
        'flags': flag_dict,
    }

    save_file(flag_dict, full_path, 'flag', file_type='json')
    save_file(results, full_path, 'training_results', file_type='json')

    # # Save sample trajectory (input, output, etc. for plotting)
    # test_errors = []
    # for i in range(16):
    #     test_dict = get_data_dict(FLAGS.batch_test)
    #     feed_dict_with_placeholder_container(test_dict, init_state_holder, sess.run(
    #         cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32)))
    #
    #     results_values, plot_results_values, in_spk, spk, spk_con = sess.run(
    #         [results_tensors, plot_result_tensors, input_spikes, z, z_con],
    #         feed_dict=test_dict)
    # print('''Statistics on the test set average error {:.2g} +- {:.2g} (averaged over 16 test batches of size {})'''
    #       .format(np.mean(test_errors), np.std(test_errors), FLAGS.batch_test))

    # Save network variables (weights, delays, etc.)
    network_data = tf_cell_to_savable_dict(cell, sess)
    save_file(network_data, full_path, 'tf_cell_net_data', file_type='pickle')

del sess
