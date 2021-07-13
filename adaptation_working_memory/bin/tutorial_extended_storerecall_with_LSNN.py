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
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik
from lsnn.guillaume_toolbox.file_saver_dumper_no_h5py import save_file

from tutorial_extended_storerecall_utils import generate_storerecall_data, error_rate, gen_custom_delay_batch, \
    update_plot, update_stp_plot, generate_spiking_storerecall_batch, generate_value_dicts, storerecall_error


from lsnn.guillaume_toolbox.tensorflow_utils import tf_downsample
from lsnn.spiking_models import tf_cell_to_savable_dict, placeholder_container_for_rnn_state,\
    feed_dict_with_placeholder_container, exp_convolve, ALIF, STP, FastALIF
from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()

##
tf.app.flags.DEFINE_string('model', 'lsnn', 'lsnn or stp')
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
tf.app.flags.DEFINE_string('reproduce', '', 'set flags to reproduce results from paper [560_A, ...]')
tf.app.flags.DEFINE_string('checkpoint', '', 'path to pre-trained model to restore')
##
tf.app.flags.DEFINE_integer('batch_train', 256, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('batch_val', 512, 'batch size of the validation set')
tf.app.flags.DEFINE_integer('batch_test', 512, 'batch size of the testing set')
tf.app.flags.DEFINE_integer('n_charac', 20, 'number of characters in the recall task')
tf.app.flags.DEFINE_integer('n_in', 88, 'number of spiking input units. Must be divisable by (n_charac*2)')
tf.app.flags.DEFINE_integer('min_hamming_dist', 5, 'minimal hamming distance in bits between test and training words')
tf.app.flags.DEFINE_integer('train_dict_size', 0, 'Not used! (use only a constrained word dictionary for training')
tf.app.flags.DEFINE_integer('test_dict_size', 20, 'Num. of test dict. words (min_hamming_dist away from training data)')
tf.app.flags.DEFINE_integer('n_regular', 0, 'number of recurrent units.')
tf.app.flags.DEFINE_integer('n_adaptive', 500, 'number of controller units')
tf.app.flags.DEFINE_integer('f0', 400, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('reg_max_rate', 100, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 4000, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 1, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 3, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('seq_len', 10, 'Number of character steps')
tf.app.flags.DEFINE_integer('seq_delay', 4, 'Expected delay in character steps. Must be <= seq_len - 2')
tf.app.flags.DEFINE_integer('tau_char', 200, 'Duration of symbols')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('lr_decay_every', 200, 'Decay every')
tf.app.flags.DEFINE_integer('print_every', 20, 'Decay every')
tf.app.flags.DEFINE_integer('n_per_channel', 2, 'input spiking neurons per input channel')
##
tf.app.flags.DEFINE_float('max_in_bit_prob', 0.2, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_float('stop_crit', 0.01, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_float('beta', 4, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 800, 'Mikolov model alpha - threshold decay')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in LSNN and output neurons')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 0.001, 'regularization coefficient')
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
##
tf.app.flags.DEFINE_bool('tau_a_spread', False, 'Uniform spread of adaptation time constants')
tf.app.flags.DEFINE_bool('tau_a_power', False, 'Power law spread of adaptation time constants')
tf.app.flags.DEFINE_float('power_exp', 2.5, 'Scale parameter of power distribution')
tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('monitor_plot', True, 'Perform plots during training')
tf.app.flags.DEFINE_bool('interactive_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('device_placement', False, '')
tf.app.flags.DEFINE_bool('verbose', True, '')
tf.app.flags.DEFINE_bool('neuron_sign', True, '')
tf.app.flags.DEFINE_bool('adaptive_reg', False, 'Regularization coefficient incread when avg fr > reg_max_rate')
tf.app.flags.DEFINE_bool('preserve_state', True, 'preserve network state between training trials')
tf.app.flags.DEFINE_bool('ramping_learning_rate', True, 'ramp up learning rate in first 100 steps')
tf.app.flags.DEFINE_bool('distractors', True, 'show random inputs during delays')
tf.app.flags.DEFINE_bool('entropy_loss', True, 'include entropy in the loss')
tf.app.flags.DEFINE_bool('b_out', False, 'include bias in readout')
tf.app.flags.DEFINE_bool('onehot', False, 'use onehot style input')
tf.app.flags.DEFINE_bool('analog_in', False, 'feed analog input to the network')
tf.app.flags.DEFINE_bool('no_recall_distr', True, 'do not show any input values during recall command')
tf.app.flags.DEFINE_bool('hamm_among_each_word', True, 'enforce hamming dist also among each test string')
tf.app.flags.DEFINE_bool('FastALIF', True, 'use simpler ALIF model without synaptic delay')
tf.app.flags.DEFINE_bool('eprop', False, 'enable symmetric eprop in FastALIF model')

assert FLAGS.FastALIF or not FLAGS.eprop, "eprop implemented only with FastALIF model"

assert FLAGS.n_charac % 2 == 0, "Please have even number of bits in value word"

if FLAGS.reproduce == '560_extSR':
    FLAGS.model = 'lsnn'
    FLAGS.n_in = (FLAGS.n_charac * 2 + 2 * 2) * FLAGS.n_per_channel
    # FLAGS.n_charac = 20
    # FLAGS.seq_len = 10
    # FLAGS.seq_delay = 4
    # FLAGS.test_dict_size = 20
    # FLAGS.min_hamming_dist = 5
    # FLAGS.tau_char = 200
    # FLAGS.f0 = 400
    # FLAGS.tau_a = 800
    # FLAGS.beta = 4
    # FLAGS.n_per_channel = 2
    # FLAGS.n_regular = 0
    # FLAGS.n_adaptive = 500
    # FLAGS.lr_decay = 0.8
    # FLAGS.batch_train = 256
    # FLAGS.entropy_loss = True
    # FLAGS.distractors = True
    # FLAGS.do_plot = True


if FLAGS.batch_val is None:
    FLAGS.batch_val = FLAGS.batch_train
if FLAGS.batch_test is None:
    FLAGS.batch_test = FLAGS.batch_train

assert FLAGS.model in ['lsnn', 'stp', 'lstm']
assert FLAGS.n_in == (FLAGS.n_charac * 2 + 2 * 2) * FLAGS.n_per_channel,\
    "Number of input neurons not compatible with other parameters."


def custom_seqence():
    s = rd.choice([0, 1], size=FLAGS.seq_len)
    s[0] = FLAGS.n_charac  # store
    s[7] = FLAGS.n_charac + 1  # recall
    s[12] = FLAGS.n_charac  # store
    s[13] = 1 if s[1] == 0 else 0
    s[19] = FLAGS.n_charac + 1  # recall
    return s


if FLAGS.comment == '':
    FLAGS.comment = FLAGS.reproduce
custom_plot = None
# custom_plot = np.stack([custom_seqence() for _ in range(FLAGS.batch_test)], axis=0)

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
n_total_neurons = FLAGS.n_regular + FLAGS.n_adaptive

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
recall_symbol = n_input_symbols - 1  # ID of the recall symbol
store_symbol = n_input_symbols - 2  # ID of the store symbol

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(0.75 * FLAGS.n_in)
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    np.random.shuffle(in_neuron_sign)

    n_excitatory = int(0.75 * n_total_neurons)
    n_inhibitory = n_total_neurons - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not FLAGS.neuron_sign: print('WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Save parameters and training log
try:
    flag_dict = FLAGS.flag_values_dict()
except:
    print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
    flag_dict = FLAGS.__flags
print(json.dumps(flag_dict, indent=4))

tau_a_spread = None
# Generate the cell
if FLAGS.model == 'lsnn':
    if FLAGS.tau_a_spread:
        tau_a_spread = np.random.uniform(size=FLAGS.n_regular+FLAGS.n_adaptive) * FLAGS.tau_a
    elif FLAGS.tau_a_power:
        tau_a_spread = (1. - np.random.power(a=FLAGS.power_exp, size=FLAGS.n_regular+FLAGS.n_adaptive)) * FLAGS.tau_a
    else:
        tau_a_spread = FLAGS.tau_a
    flag_dict['tauas'] = tau_a_spread
    beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
    cell = ALIF(n_in=FLAGS.n_in, n_rec=n_total_neurons, tau=tau_v, n_delay=FLAGS.n_delay,
                n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_spread, beta=beta, thr=FLAGS.thr,
                rewiring_connectivity=FLAGS.rewiring_connectivity,
                in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                dampening_factor=FLAGS.dampening_factor, thr_min=FLAGS.thr_min
                ) if not FLAGS.FastALIF else \
        FastALIF(n_in=FLAGS.n_in, n_rec=n_total_neurons, tau=tau_v, n_delay=FLAGS.n_delay,
             n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_spread, beta=beta, thr=FLAGS.thr,
             rewiring_connectivity=FLAGS.rewiring_connectivity,
             in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
             dampening_factor=FLAGS.dampening_factor, thr_min=FLAGS.thr_min,
             stop_z_gradients=FLAGS.eprop
             )
elif FLAGS.model == 'stp':
    cell = STP(
        n_in=FLAGS.n_in, n_rec=n_total_neurons, tau=tau_v,
        n_refractory=FLAGS.n_ref, dt=dt, thr=FLAGS.thr,
        rewiring_connectivity=FLAGS.rewiring_connectivity,
        in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
        dampening_factor=FLAGS.dampening_factor,
        tau_F=FLAGS.tauF, tau_D=FLAGS.tauD, U=FLAGS.U,
    )
elif FLAGS.model == 'lstm':
    cell = tf.contrib.rnn.LSTMCell(n_total_neurons, forget_bias=1.0)
else:
    raise ValueError("Unknown model: " + FLAGS.model)

if FLAGS.model != 'lstm':
    # balance the input weights of store-recall signals with the rest of the bits
    sr_win_coeff = FLAGS.n_charac
    increase_sr_weights = tf.assign(cell.w_in_var[:FLAGS.n_per_channel*2*2, :],
                                    cell.w_in_init[:FLAGS.n_per_channel*2*2, :] * sr_win_coeff)

cell_name = type(cell).__name__
print('\n -------------- \n' + cell_name + '\n -------------- \n')
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_reference = '{}_{}_seqlen{}_seqdelay{}_in{}_R{}_A{}_lr{}_tauchar{}_comment{}'.format(
    time_stamp, cell_name, FLAGS.seq_len, FLAGS.seq_delay, FLAGS.n_in, FLAGS.n_regular, FLAGS.n_adaptive,
    FLAGS.learning_rate, FLAGS.tau_char, FLAGS.comment)
print('FILE REFERENCE: ' + file_reference)

# Saving setup
full_path = os.path.join(result_folder, file_reference)
if not os.path.exists(full_path):
    os.makedirs(full_path)

input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in), name='InputSpikes')
recall_charac_mask = tf.placeholder(dtype=tf.bool, shape=(None, None), name='RecallMask')
target_sequence = tf.placeholder(dtype=tf.int64, shape=(None, None, FLAGS.n_charac), name='TargetSequence')
batch_size_holder = tf.placeholder(dtype=tf.int32, name='BatchSize')  # Int that contains the batch size
store_signal_to_batch_map_holder = tf.placeholder(shape=(None), dtype=tf.int32, name='StoreBatchMap')
init_state_holder = placeholder_container_for_rnn_state(cell.state_size, dtype=tf.float32, batch_size=None)

if not FLAGS.onehot:
    train_value_dict, test_value_dict = generate_value_dicts(
        n_values=FLAGS.n_charac,
        train_dict_size=FLAGS.train_dict_size, test_dict_size=FLAGS.test_dict_size,
        max_prob_active=FLAGS.max_in_bit_prob,
        min_hamming_dist=FLAGS.min_hamming_dist,
        hamm_among_each_word=FLAGS.hamm_among_each_word,
    )
    # NOTE: currently train_value_dict is empty and is not used anywhere in the simulations
    save_file({"train_value_dict": train_value_dict, "test_value_dict": test_value_dict},
              full_path, 'value_dicts', file_type='json')
save_file(flag_dict, full_path, 'flags', file_type='json')


def get_data_dict(batch_size, seq_len=FLAGS.seq_len, batch=None, override_input=None, test=False):
    p_sr = 1/(1 + FLAGS.seq_delay)
    spk_data, input_data, target_seq_data, is_recall_data, store_signal_to_batch_map = \
        generate_spiking_storerecall_batch(
            batch_size=batch_size, length=seq_len, prob_storerecall=p_sr,
            value_dict=(test_value_dict if test else None) if not FLAGS.onehot else None,
            n_charac_duration=FLAGS.tau_char,
            n_neuron=FLAGS.n_in,
            n_per_channel=FLAGS.n_per_channel,
            f0=FLAGS.f0 / 1000,  # convert frequency in Hz to kHz or probability of firing every dt=1ms step
            test_dict=test_value_dict if not FLAGS.onehot else None,
            max_prob_active=None,
            min_hamming_dist=FLAGS.min_hamming_dist if not FLAGS.onehot else None,
            distractors=FLAGS.distractors,
            n_values=FLAGS.n_charac,
            onehot=FLAGS.onehot,
            no_distractors_during_recall=FLAGS.no_recall_distr,
        )
    # data_dict = {input_spikes: spk_data, input_nums: in_data, target_nums: target_data,
    data_dict = {input_spikes: spk_data if not FLAGS.analog_in else input_data,
                 recall_charac_mask: is_recall_data, store_signal_to_batch_map_holder: store_signal_to_batch_map,
                 target_sequence: target_seq_data, batch_size_holder: batch_size}

    return data_dict


# Define the name of spike train for the different models
z_stack, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=init_state_holder, dtype=tf.float32)
if FLAGS.model == 'lsnn':
    z, b_con = z_stack
elif FLAGS.model == 'stp':
    z, stp_u, stp_x = z_stack
elif FLAGS.model == 'lstm':
    z = z_stack
z_con = []
z_all = z

with tf.name_scope('RecallLoss'):
    epsilon = tf.constant(1e-8, name="epsilon")
    # target_nums_at_recall = tf.boolean_mask(target_nums, recall_charac_mask)
    # Y = tf.one_hot(target_nums_at_recall, depth=FLAGS.n_charac, name='Target')
    Y = tf.boolean_mask(target_sequence, recall_charac_mask, name='Target')

    # MTP models do not use controller (modulator) population for output
    out_neurons = z_all
    n_neurons = out_neurons.get_shape()[2]
    psp = exp_convolve(out_neurons, decay=decay)

    if 0 < FLAGS.rewiring_connectivity and 0 < FLAGS.readout_rewiring_connectivity:
        w_out, w_out_sign, w_out_var, _ = weight_sampler(n_total_neurons, FLAGS.n_charac,
                                                         FLAGS.readout_rewiring_connectivity,
                                                         neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, FLAGS.n_charac])

    if FLAGS.b_out:
        b_out = tf.Variable(tf.zeros([FLAGS.n_charac]), name='out_bias')
        out = einsum_bij_jk_to_bik(psp, w_out) + b_out
    else:
        out = einsum_bij_jk_to_bik(psp, w_out)

    # out_char_step = tf_downsample(out, new_size=FLAGS.seq_len, axis=1)
    out_char_step = out[:, FLAGS.tau_char//2::FLAGS.tau_char]  # take middle ms of every word step
    Y_predict = tf.boolean_mask(out_char_step, recall_charac_mask, name='Prediction')
    Y_predict_sigm = tf.sigmoid(Y_predict)

    # loss_recall = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_predict))
    # loss_recall = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_predict))
    if FLAGS.onehot:
        target_nums_at_recall = tf.boolean_mask(target_sequence, recall_charac_mask)
        target_nums_at_recall = tf.argmax(target_nums_at_recall, axis=1)
        loss_recall = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_nums_at_recall, logits=Y_predict)
    else:
        Y = tf.cast(Y, tf.float32)
        loss_recall = Y * -tf.log(Y_predict_sigm + epsilon) + (1 - Y) * -tf.log(1 - (Y_predict_sigm + epsilon))
    loss_recall = tf.reduce_mean(loss_recall)

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out) if FLAGS.onehot else tf.sigmoid(out)
        out_plot_char_step = tf_downsample(out_plot, new_size=FLAGS.seq_len, axis=1)

    recall_acc, recall_errors, per_bit_accuracy, per_bit_error, per_word_error, _, failed_store_idxs = \
        storerecall_error(Y_predict, target_nums_at_recall if FLAGS.onehot else Y, onehot=FLAGS.onehot)

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

    if FLAGS.entropy_loss:
        loss_entropy = tf.reduce_mean(Y_predict_sigm * tf.log(Y_predict_sigm + epsilon))
        loss = loss + 0.3 * loss_entropy

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
# sess.run(increase_sr_weights)

# w_in_init = sess.run(cell.w_in_var)
# w_in_init_avg = np.reshape(w_in_init, ((FLAGS.n_charac * 2 + 2), FLAGS.n_per_channel, w_in_init.shape[1]))
# w_in_init_avg = np.mean(np.abs(w_in_init_avg), axis=(1, 2))
# print(w_in_init_avg)

# if FLAGS.handcraft:
    # # print("PRE w_rec", sess.run(cell.w_rec_var))
    # set_w_rec = tf.assign(cell.w_rec_var, np.zeros((n_total_neurons, n_total_neurons)))
    # sess.run(set_w_rec)
    #
    # w_in_b0 = np.repeat(np.array([[0.002, -0.1]]), FLAGS.n_per_channel, axis=0)
    # w_in_b1 = np.repeat(np.array([[-0.1, 0.002]]), FLAGS.n_per_channel, axis=0)
    # w_in_s = np.repeat(np.array([[0.1 for _ in range(n_total_neurons)]]), FLAGS.n_per_channel * (FLAGS.n_charac // 2), axis=0)
    # w_in_r = np.repeat(np.array([[0.1 for _ in range(n_total_neurons)]]), FLAGS.n_per_channel * (FLAGS.n_charac // 2), axis=0)
    # w_in = np.vstack((w_in_s, w_in_r, w_in_b0, w_in_b1))
    # # print("w_in shape", w_in.shape)
    # set_w_in = tf.assign(cell.w_in_var, w_in)
    # sess.run(set_w_in)
    #
    # w_out_v = np.array([[0, 10], [10, 0]])
    # set_w_out = tf.assign(w_out, w_out_v)
    # sess.run(set_w_out)


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
    fig, ax_list = plt.subplots(nrows=6, figsize=(8, 9), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.set_window_title(socket.gethostname() + ' - ' + FLAGS.comment)


test_loss_list = []
test_loss_with_reg_list = []
validation_error_list = []
validation_word_error_list = []
train_errors = [1.]
train_word_errors = [1.]
tau_delay_list = []
training_time_list = []
time_to_ref_list = []
results_tensors = {
    'loss': loss,
    'loss_reg': loss_reg,
    'loss_recall': loss_recall,
    'recall_errors': recall_errors,
    'word_errors': per_word_error,
    'final_state': final_state,
    'av': av,
    'adaptive_regularization_coeff': adaptive_regularization_coeff,
    'w_out': w_out,
}
if FLAGS.model != 'lstm':
    results_tensors['w_in_val'] = cell.w_in_val
    results_tensors['w_rec_val'] = cell.w_rec_val
    w_in_last = sess.run(cell.w_in_val)
    w_rec_last = sess.run(cell.w_rec_val)
    w_out_last = sess.run(w_out)

plot_result_tensors = {
    'input_spikes': input_spikes,
    'z': z,
    'z_con': z_con,
    # 'input_nums': input_nums,
    # 'target_nums': target_nums,
    'out_plot_char_step': out_plot_char_step,
    'psp': psp,
    'out_plot': out_plot,
    'recall_charac_mask': recall_charac_mask,
    'Y': Y,
    'Y_predict': Y_predict,
    'failed_store_idxs': failed_store_idxs,
    'store_signal_to_batch_map_holder': store_signal_to_batch_map_holder,
}
if FLAGS.model == 'lsnn':
    plot_result_tensors['b_con'] = b_con
elif FLAGS.model == 'stp':
    plot_result_tensors['stp_u'] = stp_u
    plot_result_tensors['stp_x'] = stp_x


ramping_iterations = FLAGS.lr_decay_every
ramping_learning_rate_values = tf.linspace(0.001, 1., num=ramping_iterations)
clipped_global_step = tf.minimum(global_step, ramping_iterations - 1)
ramping_learning_rate_op = tf.assign(learning_rate,
                                     FLAGS.learning_rate * ramping_learning_rate_values[clipped_global_step])

smallest_error = 999.

pbar = tqdm(total=print_every, desc="training")
train_bit_error = None
t_train = 0
t_ref = time()
for k_iter in range(FLAGS.n_iter):

    if k_iter < ramping_iterations and FLAGS.ramping_learning_rate:
        new_lr = sess.run(ramping_learning_rate_op)

    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # train
    train_dict = get_data_dict(FLAGS.batch_train)
    feed_dict_with_placeholder_container(train_dict, init_state_holder, last_final_state_state_training_pointer[0])
    t0 = time()
    final_state_value, _, _, train_error, train_bit_error, train_word_error, plot_results_values = sess.run(
        [final_state, train_step, update_regularization_coeff, recall_errors,
         per_bit_error, per_word_error, plot_result_tensors],
        feed_dict=train_dict)
    # print(plot_results_values['failed_store_idxs'])
    if FLAGS.preserve_state:
        last_final_state_state_training_pointer[0] = final_state_value
    t_train = time() - t0
    train_errors.append(train_error)
    train_word_errors.append(train_word_error)
    pbar.update(1)

    if np.mod(k_iter, print_every) == 0:
        pbar.close()
        # Monitor the training with a validation set
        t0 = time()
        val_dict = get_data_dict(FLAGS.batch_val, test=True)
        feed_dict_with_placeholder_container(val_dict, init_state_holder, last_final_state_state_validation_pointer[0])
        # results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)
        results_values = sess.run(results_tensors, feed_dict=val_dict)
        if FLAGS.preserve_state:
            last_final_state_state_validation_pointer[0] = results_values['final_state']
            # last_final_state_state_testing_pointer[0] = results_values['final_state']
        t_run = time() - t0

        # Storage of the results
        test_loss_with_reg_list.append(results_values['loss_reg'])
        test_loss_list.append(results_values['loss_recall'])
        validation_error_list.append(results_values['recall_errors'])
        validation_word_error_list.append(results_values['word_errors'])
        training_time_list.append(t_train)
        time_to_ref_list.append(time() - t_ref)
        results = {
            'error': validation_error_list[-1],
            'word_error': validation_word_error_list[-1],
            'loss': test_loss_list[-1],
            'loss_with_reg': test_loss_with_reg_list[-1],
            'loss_with_reg_list': test_loss_with_reg_list,
            'val_error_list': validation_error_list,
            'val_word_error_list': validation_word_error_list,
            'train_error_list': train_errors,
            'train_word_error_list': train_word_errors,
            'loss_list': test_loss_list,
            'time_to_ref': time_to_ref_list,
            'training_time': training_time_list,
            'tau_delay_list': tau_delay_list,
            'flags': flag_dict,
        }

        save_file(results, full_path, 'training_results', file_type='json')

        if validation_word_error_list[-1] < smallest_error:
            early_stop_valid_results = {
                'val_error': validation_error_list[-1],
                'val_word_error': validation_word_error_list[-1],
                'train_error': np.mean(train_errors[-print_every:]),
                'train_word_error': np.mean(train_word_errors[-print_every:]),
                'val_error_list': validation_error_list,
                'val_word_error_list': validation_word_error_list,
                'train_error_list': train_errors,
                'train_word_error_list': train_word_errors,
                'flags': flag_dict,
                'k_iter': k_iter,
            }
            smallest_error = validation_word_error_list[-1]
            print("Early stopping checkpoint! Smallest validation error so far: " + str(smallest_error))
            save_file(early_stop_valid_results, full_path, 'early_stop_valid_results', file_type='json')
            saver.save(sess, os.path.join(full_path, 'model'))
            saver.export_meta_graph(os.path.join(full_path, 'graph.meta'))
            save_file(plot_results_values, full_path, 'plot_trajectory_data', 'pickle')

        print(("Iter {}, avg.error on the train set BIT: {:.2g} WORD: {:.2g} and "
               "test set BIT: {:.2g} WORD: {:.2g}")
              .format(k_iter, np.mean(train_errors[-print_every:]), np.mean(train_word_errors[-print_every:]),
                      validation_error_list[-1], validation_word_error_list[-1]))

        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max

        firing_rate_stats = get_stats(results_values['av'] * 1000)

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f}
            comput. time (s)  train {:.2g} \t valid {:.2g} \t loss: classif. {:.2g},  reg. loss  {:.2g}
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                t_train, t_run, results_values['loss_recall'], results_values['loss_reg']
            ))
            # print(train_bit_error)
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
            if FLAGS.model in ['lsnn', 'lstm']:
                update_plot(plt, ax_list, FLAGS, plot_results_values)
            else:
                update_stp_plot(plt, ax_list, FLAGS, plot_results_values)
            tmp_path = os.path.join(full_path,
                                    'fig_train_' + start_time.strftime("%H%M") + '_' +
                                    str(k_iter) + '.pdf')
            fig.savefig(tmp_path, format='pdf')

        if np.mean(validation_word_error_list[-1]) < FLAGS.stop_crit:
            print('LESS THAN ' + str(FLAGS.stop_crit) + ' ERROR ACHIEVED - STOPPING - SOLVED at epoch ' + str(k_iter))
            break
        pbar = tqdm(total=print_every, desc="training")


print('FINISHED IN {:.2g} s'.format(time() - t_ref))

if custom_plot is not None:
    test_dict = get_data_dict(FLAGS.batch_test, override_input=custom_plot)
    feed_dict_with_placeholder_container(test_dict, init_state_holder, sess.run(
        cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32)))
    plot_custom_results_values = sess.run(plot_result_tensors, feed_dict=test_dict)
    save_file(plot_custom_results_values, full_path, 'plot_custom_trajectory_data', 'pickle')
    if FLAGS.do_plot and FLAGS.monitor_plot:
        for batch in range(10):  # FLAGS.batch_test
            if FLAGS.model in ['lsnn', 'lstm']:
                update_plot(plt, ax_list, FLAGS, plot_custom_results_values, batch=batch)
            else:
                update_stp_plot(plt, ax_list, FLAGS, plot_custom_results_values, batch=batch)
            plt.savefig(os.path.join(full_path, 'figure_custom' + str(batch) + '.pdf'), format='pdf')

# Save network variables (weights, delays, etc.)
network_data = tf_cell_to_savable_dict(cell, sess)
network_data['w_out'] = results_values['w_out']
save_file(network_data, full_path, 'tf_cell_net_data', file_type='pickle')

del sess
