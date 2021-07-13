'''
Authors: Darjan Salaj and Guillaume Bellec 2017 - 2018

The purpose of this script is to test the ALIF cell model and
provide a basic tutorial.

Out of the box you should get good performance (it uses a 20%
rewiring), please report any bug or unexpected performance.

One should get approximately:
- 40% accuracy in 100 iterations
- 60% in 200 iterations (about 30 minutes in our fast crunchers - figipc157 to 164)
- you should eventually get above 90% with 20k ~ 30k iterations (about 24h).
Best runs should achieve up to 96% in 36k iterations.

'''

import matplotlib

from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik

import datetime
import os
import socket
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lsnn.guillaume_toolbox.file_saver_dumper_no_h5py import save_file, get_storage_path_reference, NumpyAwareEncoder
from tutorial_sequential_mnist_plot import update_mnist_plot

from lsnn.spiking_models import tf_cell_to_savable_dict, exp_convolve, ALIF
from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper
from lsnn.guillaume_toolbox.tensorflow_utils import tf_downsample
import json
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
tf.app.flags.DEFINE_string('reproduce', '', 'set flags to reproduce results from paper [560_ELIF, 560_ALIF]')
tf.app.flags.DEFINE_bool('save_data', True, 'whether to save simulation data in result folder')
##
tf.app.flags.DEFINE_integer('n_batch', 256, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('n_in', 80, 'number of input units to convert gray level input spikes.')
tf.app.flags.DEFINE_integer('n_regular', 0, 'number of regular spiking units in the recurrent layer.')
tf.app.flags.DEFINE_integer('n_adaptive', 200, 'number of adaptive spiking units in the recurrent layer')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target firing rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 36000, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 1, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 5, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('lr_decay_every', 2500, 'Decay learning rate every n steps')
tf.app.flags.DEFINE_integer('print_every', 400, '')
tf.app.flags.DEFINE_integer('n_repeat', 1, 'Repeat factor to extend time of mnist task')
##
tf.app.flags.DEFINE_float('beta', 1., 'Scaling constant of the adaptive threshold')
# to solve safely set tau_a == expected recall delay
tf.app.flags.DEFINE_float('tau_a', 700, 'Adaptation time constant')
tf.app.flags.DEFINE_float('tau_v', 20, 'Membrane time constant of output readouts')
tf.app.flags.DEFINE_float('thr', 0.01, 'Baseline threshold voltage')
tf.app.flags.DEFINE_float('thr_min', .005, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 0.1, 'regularization coefficient to target a specific firing rate')
tf.app.flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
tf.app.flags.DEFINE_float('proportion_excitatory', 0.75, 'proportion of excitatory neurons')
##
tf.app.flags.DEFINE_bool('tau_a_spread', False, 'Uniform spread of adaptation time constants')
tf.app.flags.DEFINE_bool('tau_a_power', False, 'Power law spread of adaptation time constants')
tf.app.flags.DEFINE_float('power_exp', 2.5, 'Scale parameter of power distribution')
tf.app.flags.DEFINE_bool('interactive_plot', False, 'Perform plots')
tf.app.flags.DEFINE_bool('verbose', True, 'Print many info during training')
tf.app.flags.DEFINE_bool('neuron_sign', True,
                         'If rewiring is active, this will fix the sign of input and recurrent neurons')
tf.app.flags.DEFINE_bool('crs_thr', True, 'Generate spikes with threshold crossing method')
tf.app.flags.DEFINE_bool('prm', False, 'Fixed permutation of pixels')

tf.app.flags.DEFINE_float('rewiring_connectivity', 0.12,
                          'possible usage of rewiring with ALIF and LIF (0.2 and 0.5 have been tested)')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring (irrelevant without rewiring)')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Parameter necessary to approximate the spike derivative')

if not FLAGS.crs_thr:
    FLAGS.n_in = 1

if FLAGS.comment == '':
    FLAGS.comment = FLAGS.reproduce

if FLAGS.reproduce == '560_ELIF':
    print("Using the hyperparameters as in 560 paper: LSNN - ELIF network")
    FLAGS.beta = -0.9
    FLAGS.thr = 0.08
    FLAGS.tau_a = 700
    FLAGS.rewiring_connectivity = -1
    FLAGS.n_regular = 120
    FLAGS.n_adaptive = 100

if FLAGS.reproduce == '560_ALIF':
    print("Using the hyperparameters as in 560 paper: LSNN - ALIF network")
    FLAGS.beta = 1.8
    FLAGS.thr = 0.08
    FLAGS.tau_a = 700
    FLAGS.rewiring_connectivity = -1
    FLAGS.n_regular = 120
    FLAGS.n_adaptive = 100

if FLAGS.reproduce == '560_LIF':
    print("Using the hyperparameters as in 560 paper: LSNN - LIF network")
    FLAGS.beta = 0.
    FLAGS.thr = 0.08
    FLAGS.rewiring_connectivity = -1
    FLAGS.n_regular = 220
    FLAGS.n_adaptive = 0

# Define the flag object as dictionnary for saving purposes
_, storage_path, flag_dict = get_storage_path_reference(__file__, FLAGS, './results/', flags=False)
storage_path = storage_path + '_' + FLAGS.comment
if FLAGS.save_data:
    os.makedirs(storage_path, exist_ok=True)
    save_file(flag_dict, storage_path, 'flag', 'json')
    print('saving data to: ' + storage_path)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Fix the random seed if given as an argument
dt = 1.  # Time step is by default 1 ms
n_output_symbols = 10

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(FLAGS.proportion_excitatory * FLAGS.n_in) + 1
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    np.random.shuffle(in_neuron_sign)

    n_excitatory = int(FLAGS.proportion_excitatory * (FLAGS.n_regular + FLAGS.n_adaptive)) + 1
    n_inhibitory = FLAGS.n_regular + FLAGS.n_adaptive - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not (FLAGS.neuron_sign == False): print(
        'WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Define the cell
if FLAGS.tau_a_spread:
    tau_a_spread = np.random.uniform(size=FLAGS.n_regular+FLAGS.n_adaptive) * FLAGS.tau_a
elif FLAGS.tau_a_power:
    tau_a_spread = (1. - np.random.power(a=FLAGS.power_exp, size=FLAGS.n_regular+FLAGS.n_adaptive)) * FLAGS.tau_a
else:
    tau_a_spread = FLAGS.tau_a
beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=FLAGS.tau_v, n_delay=FLAGS.n_delay,
            n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_spread, beta=beta, thr=FLAGS.thr,
            rewiring_connectivity=FLAGS.rewiring_connectivity,
            in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
            dampening_factor=FLAGS.dampening_factor)

if FLAGS.tau_a_power:
    count, bins, ignored = plt.hist(tau_a_spread, bins=30)
    plt.savefig(os.path.join(storage_path, 'taua_power_dist.pdf'), format='pdf')
    plt.clf()

flag_dict['tauas'] = tau_a_spread
print(json.dumps(flag_dict, indent=4, cls=NumpyAwareEncoder))

# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(FLAGS.n_batch, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
targets = tf.placeholder(dtype=tf.int64, shape=(FLAGS.n_batch,),
                         name='Targets')  # Lists of target characters of the recall task


def find_onset_offset(y, threshold):
    """
    Given the input signal `y` with samples,
    find the indices where `y` increases and descreases through the value `threshold`.
    Return stacked binary arrays of shape `y` indicating onset and offset threshold crossings.
    `y` must be 1-D numpy arrays.
    """
    if threshold == 1:
        equal = y == threshold
        transition_touch = np.where(equal)[0]
        touch_spikes = np.zeros_like(y)
        touch_spikes[transition_touch] = 1
        return np.expand_dims(touch_spikes, axis=0)
    else:
        # Find where y crosses the threshold (increasing).
        lower = y < threshold
        higher = y >= threshold
        transition_onset = np.where(lower[:-1] & higher[1:])[0]
        transition_offset = np.where(higher[:-1] & lower[1:])[0]
        onset_spikes = np.zeros_like(y)
        offset_spikes = np.zeros_like(y)
        onset_spikes[transition_onset] = 1
        offset_spikes[transition_offset] = 1

        return np.stack((onset_spikes, offset_spikes))


if FLAGS.prm:
    permutation = np.random.permutation(np.arange(28*28))


def get_data_dict(batch_size, type='train'):
    '''
    Generate the dictionary to be fed when running a tensorflow op.

    :param batch_size:
    :param test:
    :return:
    '''
    if type == 'test':
        input_px, target_oh = mnist.test.next_batch(batch_size, shuffle=False)
    elif type == 'validation':
        input_px, target_oh = mnist.validation.next_batch(batch_size)
    elif type == 'train':
        input_px, target_oh = mnist.train.next_batch(batch_size)
    else:
        raise ValueError("Wrong data group: " + str(type))

    if FLAGS.prm:
        input_px[:] = input_px[:, permutation]

    target_num = np.argmax(target_oh, axis=1)

    if FLAGS.n_repeat > 1:
        input_px = np.repeat(input_px, FLAGS.n_repeat, axis=1)

    if FLAGS.crs_thr:
        # GENERATE THRESHOLD CROSSING SPIKES
        thrs = np.linspace(0, 1, FLAGS.n_in // 2)  # number of input neurons determins the resolution
        spike_stack = []
        for img in input_px:  # shape img = (784)
            Sspikes = None
            for thr in thrs:
                if Sspikes is not None:
                    Sspikes = np.concatenate((Sspikes, find_onset_offset(img, thr)))
                else:
                    Sspikes = find_onset_offset(img, thr)
            Sspikes = np.array(Sspikes)  # shape Sspikes = (31, 784)
            Sspikes = np.swapaxes(Sspikes, 0, 1)
            spike_stack.append(Sspikes)
        spike_stack = np.array(spike_stack)
        # add output cue neuron, and expand time for two image rows (2*28)
        out_cue_duration = 2 * 28 * FLAGS.n_repeat
        spike_stack = np.lib.pad(spike_stack, ((0, 0), (0, out_cue_duration), (0, 1)), 'constant')
        # output cue neuron fires constantly for these additional recall steps
        spike_stack[:, -out_cue_duration:, -1] = 1
    else:
        spike_stack = input_px
        spike_stack = np.expand_dims(spike_stack, axis=2)
        # # match input dimensionality (add inactive output cue neuron)
        # spike_stack = np.lib.pad(spike_stack, ((0, 0), (0, 0), (0, 1)), 'constant')

    # transform target one hot from batch x classes to batch x time x classes
    data_dict = {input_spikes: spike_stack, targets: target_num}
    return data_dict, input_px


outputs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
z, b = outputs
z_regular = z[:, :, :FLAGS.n_regular]
z_adaptive = z[:, :, FLAGS.n_regular:]

with tf.name_scope('ClassificationLoss'):
    psp_decay = np.exp(-dt / FLAGS.tau_v)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
    psp = exp_convolve(z, decay=psp_decay)
    n_neurons = z.get_shape()[2]

    # Define the readout weights
    if 0 < FLAGS.rewiring_connectivity:
        w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, n_output_symbols,
                                                         FLAGS.rewiring_connectivity,
                                                         neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])
    b_out = tf.get_variable(name='out_bias', shape=[n_output_symbols], initializer=tf.zeros_initializer())

    # Define the loss function
    out = einsum_bij_jk_to_bik(psp, w_out) + b_out

    if FLAGS.crs_thr:
        outt = tf_downsample(out, new_size=(28+2), axis=1)  # n_batch x 30 x 10
        Y_predict = outt[:, -1, :]  # shape batch x classes == n_batch x 10
    else:
        Y_predict = out[:, -1, :]  # shape batch x classes == n_batch x 10

    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=Y_predict))

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)

    # Define the accuracy
    Y_predict_num = tf.argmax(Y_predict, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, Y_predict_num), dtype=tf.float32))

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    regularization_f0 = FLAGS.reg_rate / 1000
    loss_regularization = tf.reduce_sum(tf.square(av - regularization_f0)) * FLAGS.reg

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)  # Op to decay learning rate

    loss = loss_regularization + loss_recall

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if 0 < FLAGS.rewiring_connectivity:

        train_step = rewiring_optimizer_wrapper(optimizer, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                                FLAGS.rewiring_connectivity,
                                                global_step=global_step,
                                                var_list=tf.trainable_variables())
    else:
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

# Real-time plotting
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Open an interactive matplotlib window to plot in real time
if FLAGS.interactive_plot:
    plt.ion()
    fig, ax_list = plt.subplots(5, figsize=(6, 7.5), gridspec_kw={'wspace':0, 'hspace':0.2})

# Store some results across iterations
test_loss_list = []
test_loss_with_reg_list = []
test_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []

# Dictionaries of tensorflow ops to be evaluated simualtenously by a session
results_tensors = {'loss': loss,
                   'loss_reg': loss_regularization,
                   'loss_recall': loss_recall,
                   'accuracy': accuracy,
                   'av': av,
                   'learning_rate': learning_rate,

                   'w_in_val': cell.w_in_val,
                   'w_rec_val': cell.w_rec_val,
                   'w_out': w_out,
                   'b_out': b_out
                   }

plot_result_tensors = {'input_spikes': input_spikes,
                       'z': z,
                       'psp': psp,
                       'out_plot': out_plot,
                       'Y_predict': Y_predict,
                       'b_con': b,
                       'z_regular': z_regular,
                       'z_adaptive': z_adaptive,
                       'targets': targets}

t_train = 0
for k_iter in range(FLAGS.n_iter):

    # Decaying learning rate
    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0 and mnist.train._epochs_completed > 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Print some values to monitor convergence
    if np.mod(k_iter, FLAGS.print_every) == 0:

        val_dict, input_img = get_data_dict(FLAGS.n_batch, type='validation')
        results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)

        if FLAGS.save_data:
            save_file(results_values, storage_path, 'results_values', 'pickle')
            save_file(plot_results_values, storage_path, 'plot_results_values', 'pickle')

        # Storage of the results
        test_loss_with_reg_list.append(results_values['loss_reg'])
        test_loss_list.append(results_values['loss_recall'])
        test_error_list.append(results_values['accuracy'])
        training_time_list.append(t_train)

        print(
            '''Iteration {}, epoch {} validation accuracy {:.3g} '''
                .format(k_iter, mnist.train._epochs_completed, test_error_list[-1],))


        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max


        firing_rate_stats = get_stats(results_values['av'] * 1000)

        # some connectivity statistics
        rewired_ref_list = ['w_in_val', 'w_rec_val', 'w_out']
        non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
        sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
        empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)
        empirical_connectivities = [nz / size for nz, size in zip(non_zeros, sizes)]

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f} (over neurons)
            connectivity (total {:.3g})\t W_in {:.3g} \t W_rec {:.2g} \t\t w_out {:.2g}
            number of non zero weights \t W_in {}/{} \t W_rec {}/{} \t w_out {}/{}

            classification loss {:.2g} \t regularization loss {:.2g}
            learning rate {:.2g} \t training op. time {:.2g}
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                empirical_connectivity,
                empirical_connectivities[0], empirical_connectivities[1], empirical_connectivities[2],
                non_zeros[0], sizes[0],
                non_zeros[1], sizes[1],
                non_zeros[2], sizes[2],
                results_values['loss_recall'], results_values['loss_reg'],
                results_values['learning_rate'], t_train,
            ))

        # Save files result
        if FLAGS.save_data:
            results = {
                'error': test_error_list[-1],
                'loss': test_loss_list[-1],
                'loss_with_reg': test_loss_with_reg_list[-1],
                'loss_with_reg_list': test_loss_with_reg_list,
                'error_list': test_error_list,
                'loss_list': test_loss_list,
                'time_to_ref': time_to_ref_list,
                'training_time': training_time_list,
                'tau_delay_list': tau_delay_list,
                'flags': flag_dict,
            }
            save_file(results, storage_path, 'results', file_type='json')

        if FLAGS.interactive_plot:
            update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_results_values)

    # train
    t0 = time()
    train_dict, input_img = get_data_dict(FLAGS.n_batch, type='train')
    final_state_value, _ = sess.run([final_state, train_step], feed_dict=train_dict)
    t_train = time() - t0

if FLAGS.interactive_plot:
    update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_results_values)


# Saving setup
# Get a meaning full fill name and so on

# Save a sample trajectory
if FLAGS.save_data:
    # Save the tensorflow graph
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(storage_path, 'session'))
    saver.export_meta_graph(os.path.join(storage_path, 'graph.meta'))

    test_errors = []
    n_test_batches = (mnist.test.num_examples//FLAGS.n_batch) + 1
    for i in range(n_test_batches):  # cover the whole test set
        test_dict, input_img = get_data_dict(FLAGS.n_batch, type='test')

        results_values, plot_results_values, in_spk, spk, targets_np = sess.run(
            [results_tensors, plot_result_tensors, input_spikes, z, targets],
            feed_dict=test_dict)
        test_errors.append(results_values['accuracy'])

    print('''Statistics on the test set: average accuracy {:.2g} +- {:.2g} (averaged over {} test batches of size {})'''
          .format(np.mean(test_errors), np.std(test_errors), n_test_batches, FLAGS.n_batch))
    plot_results_values['test_imgs'] = np.array(input_img)
    save_file(plot_results_values, storage_path, 'plot_results_values', 'pickle')
    save_file(results_values, storage_path, 'results_values', 'pickle')

    # Save files result
    results = {
        'test_errors': test_errors,
        'test_errors_mean': np.mean(test_errors),
        'test_errors_std': np.std(test_errors),
        'error': test_error_list[-1],
        'loss': test_loss_list[-1],
        'loss_with_reg': test_loss_with_reg_list[-1],
        'loss_with_reg_list': test_loss_with_reg_list,
        'error_list': test_error_list,
        'loss_list': test_loss_list,
        'time_to_ref': time_to_ref_list,
        'training_time': training_time_list,
        'tau_delay_list': tau_delay_list,
        'flags': flag_dict,
    }

    save_file(results, storage_path, 'results', file_type='json')

    if FLAGS.interactive_plot:
        for i in range(min(8, FLAGS.n_batch)):
            update_mnist_plot(ax_list, fig, plt, cell, FLAGS, plot_results_values, batch=i)
            fig.savefig(os.path.join(storage_path, 'figure_TEST_' + str(i) + '.pdf'), format='pdf')
            plt.show()
            plt.ioff()
del sess
