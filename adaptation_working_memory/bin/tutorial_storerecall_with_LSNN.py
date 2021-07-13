"""
    Store-recall task solved with LSNN model
    Results in 560 produced with following settings:
    # 20 neurons setup
    python3 bin/tutorial_storerecall_with_LSNN.py --reproduce=560_A
    # 2 neurons setup
    python3 bin/tutorial_storerecall_with_LSNN.py --reproduce=560_B
"""

# import matplotlib
# matplotlib.use('Agg')

import datetime
import os
import socket
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik
from lsnn.guillaume_toolbox.file_saver_dumper_no_h5py import save_file

from tutorial_storerecall_utils import generate_storerecall_data, error_rate, gen_custom_delay_batch, \
    update_plot, update_stp_plot


from lsnn.guillaume_toolbox.tensorflow_utils import tf_downsample
from lsnn.spiking_models import tf_cell_to_savable_dict, placeholder_container_for_rnn_state,\
    feed_dict_with_placeholder_container, exp_convolve, ALIF, STP, SynSTP
from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()

##
tf.app.flags.DEFINE_string('model', None, 'lsnn or stp')
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
tf.app.flags.DEFINE_string('reproduce', '', 'set flags to reproduce results from paper [560_A, ...]')
tf.app.flags.DEFINE_string('checkpoint', '', 'path to pre-trained model to restore')
##
tf.app.flags.DEFINE_integer('batch_train', 128, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('batch_val', None, 'batch size of the validation set')
tf.app.flags.DEFINE_integer('batch_test', None, 'batch size of the testing set')
tf.app.flags.DEFINE_integer('n_charac', 2, 'number of characters in the recall task')
tf.app.flags.DEFINE_integer('n_in', 40, 'number of input units.')
tf.app.flags.DEFINE_integer('n_regular', 0, 'number of recurrent units.')
tf.app.flags.DEFINE_integer('n_adaptive', 60, 'number of controller units')
tf.app.flags.DEFINE_integer('f0', 50, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('reg_max_rate', 100, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 400, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 1, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 3, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('seq_len', 20, 'Number of character steps')
tf.app.flags.DEFINE_integer('seq_delay', 10, 'Expected delay in character steps. Must be <= seq_len - 2')
tf.app.flags.DEFINE_integer('tau_char', 200, 'Duration of symbols')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('lr_decay_every', 100, 'Decay every')
tf.app.flags.DEFINE_integer('print_every', 20, 'Decay every')
##
tf.app.flags.DEFINE_float('stop_crit', 0.0, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_float('beta', 1, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 1200, 'Mikolov model alpha - threshold decay')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in LSNN and output neurons')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.3, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 1., 'regularization coefficient')
tf.app.flags.DEFINE_float('rewiring_connectivity', -1, 'possible usage of rewiring with ALIF and LIF (0.1 is default)')
tf.app.flags.DEFINE_float('readout_rewiring_connectivity', -1, '')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring')
tf.app.flags.DEFINE_float('rewiring_temperature', 0, '')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, '')
tf.app.flags.DEFINE_float('stochastic_factor', -1, '')
tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', .01, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('thr_min', .005, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('U', .2, 'STP baseline value of u')
tf.app.flags.DEFINE_float('tauF', 100, 'STP tau facilitation')
tf.app.flags.DEFINE_float('tauD', 1200, 'STP tau depression')
tf.app.flags.DEFINE_float('tauF_err', 10, 'STP tau facilitation range')
tf.app.flags.DEFINE_float('tauD_err', 10, 'STP tau depression range')
##
tf.app.flags.DEFINE_bool('tau_a_spread', False, 'Uniform spread of adaptation time constants')
tf.app.flags.DEFINE_bool('tau_a_power', False, 'Power law spread of adaptation time constants')
tf.app.flags.DEFINE_float('power_exp', 2.5, 'Scale parameter of power distribution')
tf.app.flags.DEFINE_bool('save_data', True, 'Save the data (training, test, network, trajectory for plotting)')
tf.app.flags.DEFINE_bool('do_plot', False, 'Perform plots')
tf.app.flags.DEFINE_bool('monitor_plot', True, 'Perform plots during training')
tf.app.flags.DEFINE_bool('interactive_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('device_placement', False, '')
tf.app.flags.DEFINE_bool('verbose', True, '')
tf.app.flags.DEFINE_bool('neuron_sign', True, '')
tf.app.flags.DEFINE_bool('adaptive_reg', False, '')
tf.app.flags.DEFINE_bool('preserve_state', True, 'preserve network state between training trials')
tf.app.flags.DEFINE_bool('synstp', False, 'synapse level simulation of STP (distribution of time constants and states)')

if FLAGS.batch_val is None:
    FLAGS.batch_val = FLAGS.batch_train
if FLAGS.batch_test is None:
    FLAGS.batch_test = FLAGS.batch_train

if FLAGS.reproduce == '560_LIF':
    print("Using the hyperparameters as in 560 paper: pure ELIF network")
    FLAGS.model = 'lsnn'
    FLAGS.beta = 0.0
    FLAGS.thr = 0.01
    FLAGS.n_regular = 60
    FLAGS.n_adaptive = 0
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400

if FLAGS.reproduce == '560_ELIF':
    print("Using the hyperparameters as in 560 paper: pure ELIF network")
    FLAGS.model = 'lsnn'
    FLAGS.beta = -0.5
    FLAGS.thr = 0.02
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.tau_a = 2000
    FLAGS.n_in = 40
    FLAGS.n_iter = 400

if FLAGS.reproduce == '560_ALIF':
    print("Using the hyperparameters as in 560 paper: pure ALIF network")
    FLAGS.model = 'lsnn'
    FLAGS.beta = 1
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.tau_a = 2000
    FLAGS.n_in = 40
    FLAGS.n_iter = 400

if FLAGS.reproduce == '560_LSNN':
    print("Using the hyperparameters as in 560 paper for LSNN autocorr intrinsic timescale analysis (Stokes)")
    FLAGS.model = 'lsnn'
    FLAGS.beta = 1
    FLAGS.thr = 0.01
    FLAGS.n_regular = 200
    FLAGS.n_adaptive = 200
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.tau_a = 2000
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.batch_train = 64
    FLAGS.batch_test = 64
    FLAGS.batch_val = 64
    # FLAGS.stop_crit = 0.05

if FLAGS.reproduce == '560_table':
    print("Using the hyperparameters as in 560 paper: ALIF table")
    FLAGS.model = 'lsnn'
    FLAGS.batch_train = 64
    FLAGS.batch_val = 64
    FLAGS.batch_test = 64
    # FLAGS.beta = 1
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.n_in = 40
    FLAGS.n_iter = 400

if FLAGS.reproduce == '560_STP_F':
    print("Using the hyperparameters as in 560 paper: LSNN - STP F network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 500
    FLAGS.tauD = 200

if FLAGS.reproduce == '560_STP_D':
    print("Using the hyperparameters as in 560 paper: LSNN - STP D network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 20
    FLAGS.tauD = 700

if FLAGS.reproduce == '560_STP_D_syn':
    print("Using the hyperparameters as in 560 paper: LSNN - STP D network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 17
    FLAGS.tauF_err = 5
    FLAGS.tauD = 671
    FLAGS.tauD_err = 17
    FLAGS.U = 0.25
    FLAGS.synstp = True

if FLAGS.reproduce == '560_STP_D_syn_scaleAll':
    print("Using the hyperparameters as in 560 paper: LSNN - STP D network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 51
    FLAGS.tauF_err = 15
    FLAGS.tauD = 2000
    FLAGS.tauD_err = 51
    FLAGS.U = 0.25
    FLAGS.synstp = True

if FLAGS.reproduce == '560_STP_F_syn':
    print("Using the hyperparameters as in 560 paper: LSNN - STP F network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 507
    FLAGS.tauF_err = 37
    FLAGS.tauD = 194
    FLAGS.tauD_err = 18
    FLAGS.U = 0.28
    FLAGS.synstp = True

if FLAGS.reproduce == '560_STP_F_syn_scaleAll':
    print("Using the hyperparameters as in 560 paper: LSNN - STP F network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 2000
    FLAGS.tauF_err = 146
    FLAGS.tauD = 765
    FLAGS.tauD_err = 71
    FLAGS.U = 0.28
    FLAGS.synstp = True

if FLAGS.reproduce == '560_STP_F_scaleAll':
    print("Using the hyperparameters as in 560 paper: LSNN - STP F network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 2000
    FLAGS.tauD = 800

if FLAGS.reproduce == '560_STP_D_scaleAll':
    print("Using the hyperparameters as in 560 paper: LSNN - STP D network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 60
    FLAGS.tauD = 2000

if FLAGS.reproduce == '560_STP_F_scaleLarge':
    print("Using the hyperparameters as in 560 paper: LSNN - STP F network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 2000
    FLAGS.tauD = 200

if FLAGS.reproduce == '560_STP_D_scaleLarge':
    print("Using the hyperparameters as in 560 paper: LSNN - STP D network")
    FLAGS.model = 'stp'
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.n_in = 40
    FLAGS.n_iter = 400
    FLAGS.tauF = 20
    FLAGS.tauD = 2000

assert FLAGS.model in ['lsnn', 'stp']


def custom_sequence():
    s = rd.choice([0, 1], size=FLAGS.seq_len)
    s[0] = FLAGS.n_charac  # store
    s[7] = FLAGS.n_charac + 1  # recall
    s[12] = FLAGS.n_charac  # store
    s[13] = 1 if s[1] == 0 else 0
    s[19] = FLAGS.n_charac + 1  # recall
    return s


if FLAGS.comment == '':
    FLAGS.comment = FLAGS.reproduce
# custom_plot = None
custom_plot = np.stack([custom_sequence() for _ in range(FLAGS.batch_test)], axis=0)

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
if FLAGS.model == 'lsnn':
    if FLAGS.tau_a_spread:
        tau_a_spread = np.random.uniform(size=FLAGS.n_regular+FLAGS.n_adaptive) * FLAGS.tau_a
    elif FLAGS.tau_a_power:
        tau_a_spread = (1. - np.random.power(a=FLAGS.power_exp, size=FLAGS.n_regular+FLAGS.n_adaptive)) * FLAGS.tau_a
    else:
        tau_a_spread = FLAGS.tau_a
    beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
    cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v, n_delay=FLAGS.n_delay,
                n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_spread, beta=beta, thr=FLAGS.thr,
                rewiring_connectivity=FLAGS.rewiring_connectivity,
                in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                dampening_factor=FLAGS.dampening_factor, thr_min=FLAGS.thr_min
                )
else:
    if FLAGS.synstp:
        cell = SynSTP(
            n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v,
            n_refractory=FLAGS.n_ref, dt=dt, thr=FLAGS.thr,
            rewiring_connectivity=FLAGS.rewiring_connectivity,
            in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
            dampening_factor=FLAGS.dampening_factor,
            U=FLAGS.U,
            tau_D=FLAGS.tauD, tau_D_err=FLAGS.tauD, tau_F=FLAGS.tauF, tau_F_err=5.
        )
    else:
        cell = STP(
            n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v,
            n_refractory=FLAGS.n_ref, dt=dt, thr=FLAGS.thr,
            rewiring_connectivity=FLAGS.rewiring_connectivity,
            in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
            dampening_factor=FLAGS.dampening_factor,
            tau_F=FLAGS.tauF, tau_D=FLAGS.tauD, U=FLAGS.U,
        )

cell_name = type(cell).__name__
print('\n -------------- \n' + cell_name + '\n -------------- \n')
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_reference = '{}_{}_seqlen{}_seqdelay{}_in{}_R{}_A{}_lr{}_tauchar{}_comment{}'.format(
    time_stamp, cell_name, FLAGS.seq_len, FLAGS.seq_delay, FLAGS.n_in, FLAGS.n_regular, FLAGS.n_adaptive,
    FLAGS.learning_rate, FLAGS.tau_char, FLAGS.comment)
print('FILE REFERENCE: ' + file_reference)

# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
input_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                            name='InputNums')  # Lists of input character for the recall task
target_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                             name='TargetNums')  # Lists of target characters of the recall task
recall_mask = tf.placeholder(dtype=tf.bool, shape=(None, None),
                             name='RecallMask')  # Binary tensor that points to the time of presentation of a recall

# Other placeholder that are useful for computing accuracy and debuggin
target_sequence = tf.placeholder(dtype=tf.int64, shape=(None, None),
                                 name='TargetSequence')  # The target characters with time expansion
batch_size_holder = tf.placeholder(dtype=tf.int32, name='BatchSize')  # Int that contains the batch size
init_state_holder = placeholder_container_for_rnn_state(cell.state_size, dtype=tf.float32, batch_size=None)
recall_charac_mask = tf.equal(input_nums, recall_symbol, name='RecallCharacMask')


def get_data_dict(batch_size, seq_len=FLAGS.seq_len, batch=None, override_input=None):
    p_sr = 1/(1 + FLAGS.seq_delay)
    spk_data, is_recall_data, target_seq_data, memory_seq_data, in_data, target_data = generate_storerecall_data(
        batch_size=batch_size,
        f0=input_f0,
        sentence_length=seq_len,
        n_character=FLAGS.n_charac,
        n_charac_duration=FLAGS.tau_char,
        n_neuron=FLAGS.n_in,
        prob_signals=p_sr,
        with_prob=True,
        override_input=override_input,
    )
    data_dict = {input_spikes: spk_data, input_nums: in_data, target_nums: target_data, recall_mask: is_recall_data,
                 target_sequence: target_seq_data, batch_size_holder: batch_size}

    return data_dict


# Define the name of spike train for the different models
z_stack, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=init_state_holder, dtype=tf.float32)
if FLAGS.model == 'lsnn':
    z, b_con = z_stack
else:
    # z, stp_u, stp_x = z_stack
    z = z_stack
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
    out_char_step = tf_downsample(out, new_size=FLAGS.seq_len, axis=1)
    Y_predict = tf.boolean_mask(out_char_step, recall_charac_mask, name='Prediction')

    # loss_recall = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_predict))
    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_nums_at_recall,
                                                                                logits=Y_predict))

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)
        out_plot_char_step = tf_downsample(out_plot, new_size=FLAGS.seq_len, axis=1)

    _, recall_errors, false_sentence_id_list = error_rate(out_char_step, target_nums, input_nums, n_charac)

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

config = tf.ConfigProto(log_device_placement=FLAGS.device_placement)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())


if len(FLAGS.checkpoint) > 0:
    saver = tf.train.Saver(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess, FLAGS.checkpoint)
    print("Model restored from ", FLAGS.checkpoint)
else:
    saver = tf.train.Saver()


last_final_state_state_training_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32))]
last_final_state_state_validation_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_val, dtype=tf.float32))]
last_final_state_state_testing_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_test, dtype=tf.float32))]

if FLAGS.do_plot:
    # Open an interactive matplotlib window to plot in real time
    if FLAGS.interactive_plot:
        plt.ion()
    fig, ax_list = plt.subplots(nrows=5, figsize=(6, 7.5), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.set_window_title(socket.gethostname() + ' - ' + FLAGS.comment)


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
    'w_in_val': cell.w_in_val,
    'w_rec_val': cell.w_rec_val,
    'w_out': w_out,
}

w_in_last = sess.run(cell.w_in_val)
w_rec_last = sess.run(cell.w_rec_val)
w_out_last = sess.run(w_out)

plot_result_tensors = {
    'input_spikes': input_spikes,
    'z': z,
    'z_con': z_con,
    'input_nums': input_nums,
    'target_nums': target_nums,
    'out_plot_char_step': out_plot_char_step,
    'psp': psp,
    'out_plot': out_plot,
    'recall_charac_mask': recall_charac_mask,
    'Y': Y,
    'Y_predict': Y_predict,
}
if FLAGS.model == 'lsnn':
    plot_result_tensors['b_con'] = b_con
# else:
#     plot_result_tensors['stp_u'] = stp_u
#     plot_result_tensors['stp_x'] = stp_x

t_train = 0
t_ref = time()
for k_iter in range(FLAGS.n_iter):

    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Monitor the training with a validation set
    t0 = time()
    val_dict = get_data_dict(FLAGS.batch_val)
    feed_dict_with_placeholder_container(val_dict, init_state_holder, last_final_state_state_validation_pointer[0])

    results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)
    if FLAGS.preserve_state:
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
            if FLAGS.model == 'lsnn':
                update_plot(plt, ax_list, FLAGS, plot_results_values)
            else:
                update_stp_plot(plt, ax_list, FLAGS, plot_results_values)
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
    train_dict = get_data_dict(FLAGS.batch_train)
    feed_dict_with_placeholder_container(train_dict, init_state_holder, last_final_state_state_training_pointer[0])
    t0 = time()
    final_state_value, _, _ = sess.run([final_state, train_step, update_regularization_coeff], feed_dict=train_dict)
    if FLAGS.preserve_state:
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
    saver.save(sess, os.path.join(full_path, 'model'))
    saver.export_meta_graph(os.path.join(full_path, 'graph.meta'))

    # Save parameters and training log
    try:
        flag_dict = FLAGS.flag_values_dict()
    except:
        print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
        flag_dict = FLAGS.__flags
    if FLAGS.model == 'lsnn':
        flag_dict['tauas'] = tau_a_spread
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

    save_file(flag_dict, full_path, 'flags', file_type='json')
    save_file(results, full_path, 'training_results', file_type='json')

    if custom_plot is not None:
        test_dict = get_data_dict(FLAGS.batch_test, override_input=custom_plot)
        feed_dict_with_placeholder_container(test_dict, init_state_holder, sess.run(
            cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32)))
        plot_custom_results_values = sess.run(plot_result_tensors, feed_dict=test_dict)
        save_file(plot_custom_results_values, full_path, 'plot_custom_trajectory_data', 'pickle')
        if FLAGS.do_plot and FLAGS.monitor_plot:
            for batch in range(10):  # FLAGS.batch_test
                if FLAGS.model == 'lsnn':
                    update_plot(plt, ax_list, FLAGS, plot_custom_results_values, batch=batch)
                else:
                    update_stp_plot(plt, ax_list, FLAGS, plot_custom_results_values, batch=batch)
                plt.savefig(os.path.join(full_path, 'figure_custom' + str(batch) + '.pdf'), format='pdf')

    # Save sample trajectory (input, output, etc. for plotting)
    test_errors = []
    for i in range(16):
        test_dict = get_data_dict(FLAGS.batch_test)
        feed_dict_with_placeholder_container(test_dict, init_state_holder, sess.run(
            cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32)))

        results_values, plot_results_values, in_spk, spk, spk_con, target_nums_np, z_sum_np = sess.run(
            [results_tensors, plot_result_tensors, input_spikes, z, z_con, target_nums, out_plot_char_step],
            feed_dict=test_dict)
        # if FLAGS.preserve_state:
        #   last_final_state_state_testing_pointer[0] = results_values['final_state']
        test_errors.append(results_values['recall_errors'])

    if FLAGS.do_plot and FLAGS.monitor_plot:
        if FLAGS.model == 'lsnn':
            update_plot(plt, ax_list, FLAGS, plot_results_values)
        else:
            update_stp_plot(plt, ax_list, FLAGS, plot_results_values)
        fig.savefig(os.path.join(full_path, 'figure_test' + start_time.strftime("%H%M") + '.pdf'), format='pdf')

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
