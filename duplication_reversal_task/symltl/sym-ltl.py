import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import yaml
import pickle
from randomgen import RandomGenerator
from sdict import sdict
from simmanager import SimManager
from simrecorder import Recorder, ZarrDataStore

from lsnn.spiking_models import ALIF, exp_convolve
from lsnn.toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper
from symltl import Timer, Task
from symltl.dataset import generate_string_single_task, spike_encode, generate_string_multi_task, generate_string_multi_task_dual, spike_encode_dual, generate_unseen_string_multi_task
from symltl.ftools import multiprocify, generatorify, fchain, fify

from tensorflow.core.protobuf import rewriter_config_pb2
config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.memory_optimization = off


def record_dict(prefix, d):
    for k, v in d.items():
        recorder.record('{}/{}'.format(prefix, k), np.array(v))


def bin_dec(a):
    return sum([int(a[-i]) * 2 ** (i - 1) for i in range(1, len(a) + 1)])


@tf.custom_gradient
def BA_out(psp, W_out, BA_out):
    logits = tf.einsum('bij,jk->bik', psp, W_out)
    def grad(dy):
        dloss_dw_out = tf.einsum('btj,btk->jk', psp, dy)
        dloss_dba_out = tf.zeros_like(BA_out)
        dloss_dpsp = tf.einsum('bik,jk->bij', dy, BA_out)
        return [dloss_dpsp, dloss_dw_out, dloss_dba_out]
    return logits, grad


def main(train_set):
    rg = RandomGenerator()
    rg.seed(c.seed)

    if c.debug:
        mfy = generatorify
    else:
        mfy = multiprocify

    if c.multi_task:
        print("Running for multiple tasks")
        if c.generate_dual_op_in_dataset:
            str_gen_fn = fify(generate_string_multi_task_dual, length=c.length, width=c.width,
                              binary_encoding=c.binary_encoding, task_cue_given_once=c.task_cue_given_once)
            print('Generate string multi task on DUAL operation.')
        elif c.generate_unseen_episodes:
            str_gen_fn = fify(generate_unseen_string_multi_task, length=c.length, width=c.width,
                              binary_encoding=c.binary_encoding, task_cue_given_once=c.task_cue_given_once,
                              train_set=train_set)
            print('Generate previously unseen episodes.')
        else:
            str_gen_fn = fify(generate_string_multi_task, length=c.length, width=c.width,
                              binary_encoding=c.binary_encoding, task_cue_given_once=c.task_cue_given_once)
            print('Generate multi task.')
    else:
        str_gen_fn = fify(generate_string_single_task, length=c.length, width=c.width, task=c.task)
        print('Generate single task.')

    if not c.generate_dual_op_in_dataset:
        input_time_steps = c.total_input_length * c.spiking_params.n_dt_per_step
        gen_fn_basic = fchain(
            fify(spike_encode, n_dt_per_step=c.spiking_params.n_dt_per_step,
                 n_input_code=c.spiking_params.n_input_code,
                 dt=c.spiking_params.dt),
            str_gen_fn)
        gen_fn = mfy(gen_fn_basic, seed=c.seed)
        tf_dataset = tf.data.Dataset.from_generator(
            gen_fn,
            {'spike_input': tf.float32, 'input': tf.float32, 'target': tf.float32},
            {'spike_input': (input_time_steps, c.total_input_width * c.spiking_params.n_input_code
                             ),
             'input': (c.total_input_length, c.total_input_width), 'target': (c.target_length, c.target_width)},
        ) \
            .batch(c.batch_size) \
            .prefetch(100)
        iterator = tf_dataset.make_one_shot_iterator()
        values = iterator.get_next()
        input_, target_ = values['spike_input'], values['target']
        input_analog_ = values['input']
    else:  # For testing dual operations - one string with both cues
        input_time_steps = c.total_input_length * c.spiking_params.n_dt_per_step
        gen_fn_basic = fchain(
            fify(spike_encode_dual, n_dt_per_step=c.spiking_params.n_dt_per_step,
                 n_input_code=c.spiking_params.n_input_code,
                 dt=c.spiking_params.dt),
            str_gen_fn)
        gen_fn = mfy(gen_fn_basic, seed=c.seed)
        tf_dataset = tf.data.Dataset.from_generator(
            gen_fn,
            {'spike_input': tf.float32, 'input': tf.float32, 'target': tf.float32},
            {'spike_input': (2, input_time_steps, c.total_input_width * c.spiking_params.n_input_code),
             'input': (2, c.total_input_length, c.total_input_width), 'target': (2, c.target_length, c.target_width)},
        ) \
            .batch(c.batch_size//2) \
            .prefetch(100)
        iterator = tf_dataset.make_one_shot_iterator()
        values = iterator.get_next()
        input_, target_ = values['spike_input'], values['target']
        input_analog_ = values['input']

        input_ = tf.reshape(input_, (c.batch_size, input_time_steps, c.total_input_width * c.spiking_params.n_input_code)
                            )
        target_ = tf.reshape(target_, (c.batch_size, c.target_length, c.target_width))
        input_analog_ = tf.reshape(input_analog_, (c.batch_size, c.total_input_length, c.total_input_width))

    # Second dimension is time
    X = tf.placeholder_with_default(input_, [None, input_time_steps,
                                             c.total_input_width * c.spiking_params.n_input_code], name='X')
    Y = tf.placeholder_with_default(target_, [None, c.target_length, c.target_width], name='Y')
    batch_size = tf.placeholder_with_default(c.batch_size, [], name='batch_size')

    n_regular = int(c.n_hidden * (1. - c.alif_params.adaptive_fraction))
    n_adaptive = int(c.n_hidden - n_regular)
    assert n_regular + n_adaptive == c.n_hidden

    beta = np.concatenate([np.zeros(n_regular), np.ones(n_adaptive) * c.alif_params.beta])
    tau_adaptation = rg.uniform(1, c.alif_params.tau_adaptation, c.n_hidden)

    if c.restore_from != '':
        print("Loading tau_adaptation from {}".format(os.path.join(c.restore_from, 'data', 'cell_params.pkl')))
        with open(os.path.join(c.restore_from, 'data', 'cell_params.pkl'), 'rb') as f:
            cell_params = pickle.load(f)
        tau_adaptation = cell_params['tau_adaptation']

    # Sign of the neurons
    is_rewiring_enabled = (0 < c.spiking_params.rewiring_connectivity < 1)
    weight_ops = {}
    n_input_neurons = c.total_input_width * c.spiking_params.n_input_code

    if is_rewiring_enabled and c.spiking_params.neuron_sign:
        n_excitatory_in = int(c.spiking_params.proportion_excitatory * n_input_neurons)
        n_inhibitory_in = n_input_neurons - n_excitatory_in
        in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
        rg.shuffle(in_neuron_sign)

        n_excitatory = int(c.spiking_params.proportion_excitatory * (n_regular + n_adaptive))
        n_inhibitory = n_regular + n_adaptive - n_excitatory
        rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
    else:
        if c.spiking_params.neuron_sign:
            print('WARNING: Neuron sign is set to None without rewiring but sign is requested')
        in_neuron_sign = None
        rec_neuron_sign = None

    cell = ALIF(
        n_in=n_input_neurons,
        n_rec=c.n_hidden,
        tau=c.alif_params.tau,
        eprop=c.eprop,
        n_delay=c.alif_params.n_delay,
        n_refractory=c.alif_params.n_refractory,
        dt=c.spiking_params.dt,
        tau_adaptation=tau_adaptation,
        beta=beta,
        thr=c.alif_params.thr,
        dampening_factor=c.spiking_params.dampening_factor,
        rewiring_connectivity=c.spiking_params.rewiring_connectivity,
        in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign)

    if is_rewiring_enabled:
        assert not c.eprop, "Eprop with rewiring not implemented"
        print("Using weight sampler")
        w_out, w_out_sign, w_out_var, w_out_conn = weight_sampler(n_regular + n_adaptive, c.n_out,
                                                   c.spiking_params.rewiring_connectivity, neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.Variable(initial_value=rg.randn(c.n_hidden, c.n_out), dtype=tf.float32, name='WOut')
    b_out = tf.Variable(initial_value=np.zeros(c.n_out), dtype=tf.float32, name='BOut')
    weight_ops['w_in_val'] = cell.w_in_val
    weight_ops['w_rec_val'] = cell.w_rec_val
    weight_ops['w_out'] = w_out
    weight_ops['b_out'] = b_out
    if is_rewiring_enabled:
        weight_ops['w_out_connected'] = w_out_conn
        weight_ops['w_in_conn'] = cell.w_in_conn
        weight_ops['w_rec_conn'] = cell.w_rec_conn

    [new_z, new_v, thr], new_state = tf.nn.dynamic_rnn(cell, X, initial_state=cell.zero_state(batch_size, tf.float32))

    av = tf.divide(tf.reduce_mean(new_z, axis=(0, 1)), c.spiking_params.dt)
    loss_av = tf.reduce_mean(tf.square(av - c.spiking_params.reg_rate))

    # The last 'length' outputs are taken later
    fltr_coeff = np.exp(-1/(c.spiking_params.n_dt_per_step/2))
    print("For exp. filter we use: ", fltr_coeff)
    cell_outputs_all_t = exp_convolve(new_z, fltr_coeff)
    cell_outputs = cell_outputs_all_t[:, (c.spiking_params.n_dt_per_step - 1)::c.spiking_params.n_dt_per_step, :]
    # print('cell_outputs: ', cell_outputs.shape)

    if not c.eprop:  # BPTT
        output_1 = tf.einsum('ijk,kl->ijl', cell_outputs, w_out) + b_out
    elif not c.random_eprop:
        output_1 = tf.einsum('ijk,kl->ijl', cell_outputs, w_out)
    else:
        random_eprop_bstd = 2.4792
        ba_out = tf.constant(random_eprop_bstd * rg.randn(c.n_hidden, c.n_out), dtype=tf.float32, name='BAOut')
        output_1 = BA_out(cell_outputs, w_out, ba_out)

    if not c.binary_encoding:
        probs = tf.nn.softmax(output_1)
    else:
        probs = tf.sigmoid(output_1)

    # assert probs.shape == (c.batch_size, c.total_input_length, c.n_out)
    relevant_probs = probs[:, -c.target_length:, :]
    # print('relevant_probs: ', relevant_probs.shape)

    if not c.binary_encoding:
        actual_output = tf.argmax(relevant_probs, axis=-1)
        # print('actual output: ', actual_output.shape)
    else:
        actual_output = tf.where(relevant_probs < 0.5, tf.zeros_like(relevant_probs), tf.ones_like(relevant_probs))

    # assert relevant_probs.shape == Y.shape, "%s, %s" % (relevant_probs.shape, Y.shape)

    if not c.binary_encoding:
        loss = -tf.reduce_mean(Y * tf.log(relevant_probs + 1e-8))
    else: # Softmax for each time step and each element independently
        loss = -tf.reduce_mean(Y * tf.log(relevant_probs + 1e-8) + (1. - Y) * tf.log(1. - relevant_probs + 1e-8))

    total_loss = loss + c.spiking_params.reg_hp * loss_av

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    if is_rewiring_enabled:
        opt = rewiring_optimizer_wrapper(optimizer, total_loss, 1e-3, c.spiking_params.rewiring_l1,
                                              c.spiking_params.rewiring_temperature,
                                              c.spiking_params.rewiring_connectivity,
                                              global_step=global_step,
                                              var_list=tf.trainable_variables())
    else:
        opt = optimizer.minimize(loss=total_loss, global_step=global_step)

    #\/ Mean across batch
    if not c.binary_encoding:
        success_rate = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(actual_output, tf.argmax(Y, axis=-1)), axis=-1), dtype=tf.float32))
        symbol_success_rate = tf.reduce_mean(tf.cast(tf.equal(actual_output, tf.argmax(Y, axis=-1)), dtype=tf.float32))
    else:
        success_rate = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(actual_output, Y), axis=(1, 2)), dtype=tf.float32))
        symbol_success_rate = tf.reduce_mean(tf.cast(tf.equal(actual_output, Y), dtype=tf.float32))

    metrics = {
        'total_loss': total_loss,
        'loss': loss,
        'success_rate': success_rate,
        'symbol_success_rate': symbol_success_rate
    }

    # Inputs, outputs and states
    input_analog_to_save = {'input_analog': input_analog_}
    iostates = {'input': input_, 'all_probs': probs, 'input_analog': input_analog_, 'target': target_,
                'actual_output': actual_output, 'states': new_z} #, 'cell_outputs': cell_outputs_all_t, 'thr': thr}

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config_proto) as sess:
        if c.restore_from != '':
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(c.restore_from, 'results')))
        else:
            sess.run(init)

        with open(os.path.join(paths.data_path, 'cell_params.pkl'), 'wb') as f:
            pickle.dump(dict(tau_adaptation=tau_adaptation, w_in_delay=sess.run(cell.w_in_delay),
                             w_rec_delay=sess.run(cell.w_rec_delay)
                             ), f, protocol=pickle.HIGHEST_PROTOCOL)

        # TRAIN
        for i in range(c.n_training_iterations):

            with Timer() as bt:
                _, metrics_val, inputs_analog_vals = sess.run([opt, metrics, input_analog_to_save])

            with Timer() as st:
                record_dict('train', metrics_val)
                record_dict('train', inputs_analog_vals)


            if i == 1 or i % 100 == 0 or i == c.n_training_iterations - 1:
                print("Training iteration {:d} :: Storage time was {:.4f} :: Batch time was {:.4f}".format(i, st.difftime, bt.difftime))
                print("Loss is {:.4f}; Symbol success rate is: {:.4f}; Success rate is: {:.4f}".format(
                    metrics_val['loss'], metrics_val['symbol_success_rate'], metrics_val['success_rate']))

                weight_results = sess.run(weight_ops)
                record_dict('weight', weight_results)

                if is_rewiring_enabled:
                    # \/ Print rewiring connectivity
                    rewired_ref_list = ['w_in_val', 'w_rec_val', 'w_out']
                    non_zeros = [np.sum(weight_results[ref] != 0) for ref in rewired_ref_list]
                    sizes = [np.size(weight_results[ref]) for ref in rewired_ref_list]
                    empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)

                    assert empirical_connectivity < c.spiking_params.rewiring_connectivity * 1.1, \
                        'Rewiring error: found connectivity {:.3g}'.format(empirical_connectivity)

                    print(
                        '''Connectivity {:.3g} \t Non zeros: W_in {}/{} W_rec {}/{} w_out {}/{}'''.format(
                            empirical_connectivity,
                            non_zeros[0], sizes[0],
                            non_zeros[1], sizes[1],
                            non_zeros[2], sizes[2],
                        ))

                save_path = saver.save(sess, os.path.join(paths.results_path, 'model'), global_step=i)
                print("Saving model in %s" % save_path)

        # TEST
        for i in range(c.n_testing_iterations):

            with Timer() as bt:
                metrics_val, iostates_val, weight_results = sess.run([metrics, iostates, weight_ops])

            with Timer() as st:
                record_dict('test', metrics_val)
                record_dict('test', iostates_val)

                if i == 0:
                    record_dict('weight', weight_results)

            if i % 100 == 0 or i == c.n_testing_iterations:
                print("TEST iteration {:d} :: Storage time was {:.4f} :: Batch time was {:.4f}".format(i, st.difftime,
                                                                                                       bt.difftime))
                print("Loss is {:.4f}; Symbol success rate is: {:.4f}; Success rate is: {:.4f}".format(
                    metrics_val['loss'], metrics_val['symbol_success_rate'], metrics_val['success_rate']))

        print(":::: TEST statistics ::::")
        print("Test success rate: ", np.mean(recorder.get_all('test/success_rate')))
        print("Test loss: ", np.mean(recorder.get_all('test/loss')))
        print("Test symbol success rate: ", np.mean(recorder.get_all('test/symbol_success_rate')))


if __name__ == '__main__':
    flags = tf.flags

    # General
    flags.DEFINE_bool("debug", False, "Debug mode.")

    flags.DEFINE_string('restore_from', '', 'Restore model from')
    flags.DEFINE_integer("seed", 3000, "Seed to use. Default 3000.")

    flags.DEFINE_bool("multi_task", True, "Train the network for multiple tasks")
    flags.DEFINE_bool("task_cue_given_once", True, "Give the cue only in the first step of the episode? "
                                                   "Otherwise while input is given (half of the episode).")
    flags.DEFINE_bool("generate_dual_op_in_dataset", False, "The same string tested on both operations? Otherwise input\
                                                             strings and operations are always randomly generated.")
    flags.DEFINE_bool("generate_unseen_episodes", False, "Used for testing on previously unseen data.") # This is done
                                                         # after the training (by restoring the simulation).

    # Common for all networks
    flags.DEFINE_integer("length", 5, "Length of input sequence")
    flags.DEFINE_integer("width", 5, "Number of bits encoding each symbol") # 2^5 = 32 symbols (one of them will be
                                                                            # End-Of-Sequence (EOS) symbol)
    flags.DEFINE_bool("binary_encoding", False, "Inputs and output binary encoded. Otherwise one-hot encoding is used.")
    flags.DEFINE_integer("training_iterations", 50000, "Number of training iterations")
    flags.DEFINE_integer("n_hidden", 320, "Number of hidden neurons to use")

    # Spiking (input) related
    flags.DEFINE_integer("n_dt_per_step", 500, "Regularization firing rate target")
    flags.DEFINE_integer("n_input_code", 5, "Number of spiking neurons used to encode each input") # Default is 1

    # ALIF related
    flags.DEFINE_float("tau_adaptation", 6000., "Adaptation time constant in ms")
    flags.DEFINE_float("adaptive_fraction", 0.6, "Fraction of neurons that are adaptive")

    # ALIF learning related
    flags.DEFINE_float("dampening_factor", 0.3, "Dampening factor")
    flags.DEFINE_float("reg_hp", 5., "Regularization factor for spiking rate")
    flags.DEFINE_float("reg_rate_hz", 20., "Regularization firing rate target")

    ## Set neuron sign
    tf.app.flags.DEFINE_bool('neuron_sign', False,
                             'If rewiring is active, this will fix the sign of input and recurrent neurons')
    tf.app.flags.DEFINE_float('proportion_excitatory', 0.8, 'proportion of excitatory neurons')

    ## Rewiring related
    flags.DEFINE_float('rewiring_connectivity', -1, #0.2,
                       'possible usage of rewiring with ALIF and LIF (disabled by default)')
    flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
    flags.DEFINE_float('rewiring_l1', 1e-2, 'l1 regularization that goes with rewiring (irrelevant without rewiring)')

    tf.app.flags.DEFINE_bool('eprop', False, 'Enable eprop?')
    tf.app.flags.DEFINE_bool('random_eprop', False, 'Enable random eprop?')

    FLAGS = flags.FLAGS

    # Read out and process parameters from FLAGS
    # BASIC params
    assert FLAGS.width <= 8  # Because of the use of unpackbits in dataset.py
    c = dict(
        debug=FLAGS.debug,
        eprop=(FLAGS.eprop or FLAGS.random_eprop),
        random_eprop=FLAGS.random_eprop,
        restore_from=os.path.expanduser(FLAGS.restore_from),
        seed=FLAGS.seed,
        length=FLAGS.length,  # Also number of time steps
        width=FLAGS.width,
        binary_encoding=FLAGS.binary_encoding,
        batch_size=50,
        n_hidden=FLAGS.n_hidden,
        n_training_iterations=FLAGS.training_iterations,
        n_testing_iterations=1000,
        task=Task.COPY, # Relevant if multi_task is set to False
        # task = Task.REVERSE,
        multi_task=FLAGS.multi_task,
        task_cue_given_once=FLAGS.task_cue_given_once,
        generate_dual_op_in_dataset=FLAGS.generate_dual_op_in_dataset,
        generate_unseen_episodes=FLAGS.generate_unseen_episodes,
        spiking_params=dict(
            dt=1.,
            n_input_code=FLAGS.n_input_code,
            n_dt_per_step=FLAGS.n_dt_per_step,  # 50
            dampening_factor=FLAGS.dampening_factor,
            reg_hp=FLAGS.reg_hp,
            reg_rate=FLAGS.reg_rate_hz / 1000.,
            neuron_sign=FLAGS.neuron_sign,
            proportion_excitatory=FLAGS.proportion_excitatory,
            rewiring_connectivity=FLAGS.rewiring_connectivity,
            rewiring_temperature=FLAGS.rewiring_temperature,
            rewiring_l1=FLAGS.rewiring_l1, ),
        alif_params=dict(
            # General LIF params
            n_refractory=5,
            n_delay=1,
            tau=20.,
            thr=0.03,
            # ALIF specific params
            adaptive_fraction=FLAGS.adaptive_fraction,
            tau_adaptation=FLAGS.tau_adaptation,
            beta=1.7, ), )


    # Derived params
    c.update(dict(total_input_length=2 * c['length'] + 2))  # String 2 times, and eos 2 times

    if FLAGS.multi_task:
        if not FLAGS.binary_encoding:
            c.update(dict(total_input_width=2**c['width'] - 1 + 4))
        else:
            c.update(dict(total_input_width=c['width'] + 4))
    else:
        c.update(dict(total_input_width=c['width'] + 2))

    if not FLAGS.binary_encoding:
        c.update(dict(target_length=c['length'] + 1, target_width=2**c['width']))  # For COPY AND REPEAT tasks, include EOS
    else:
        c.update(dict(target_length=c['length'] + 1, target_width=c['width'] + 1))  # For COPY AND REPEAT tasks, include EOS

    c.update(dict(n_out=c['target_width']))

    if FLAGS.generate_unseen_episodes:
        assert c['restore_from'] != '', "To test it on unseen episodes, restore_from has to be given."

    train_set = set()

    if c['restore_from'] != '':
        print('=== RESTORING FROM {}'.format(c['restore_from']))
        with open(os.path.join(c['restore_from'], 'data', 'config.yaml'), 'r') as f:
            tc = yaml.load(f)
        no_update_keys = ['n_training_iterations', 'batch_size', 'restore_from', 'seed', 'generate_unseen_episodes',
                          'generate_dual_op_in_dataset', 'n_testing_iterations']
        n_iter = 0
        for k, v in tc.items():
            if k not in no_update_keys:
                c[k] = v
            elif k == 'n_training_iterations':
                n_iter = v

        #### Read in all train data
        if c['generate_unseen_episodes']:
            hd = ZarrDataStore(os.path.join(c['restore_from'] + '/results', 'data.mdb'))
            train_input_analog = np.array(hd.get_all('train/input_analog'))

            # Convert inputs to analog values
            eof_sym = 2 ** c['length']

            train_mat_analog_vals = np.zeros((n_iter, c['batch_size'], c['total_input_length']))
            train_ep_task = np.zeros((n_iter, c['batch_size'], 2))
            for i in range(n_iter):
                for j in range(c['batch_size']):
                    train_ep_task[i, j, :] = train_input_analog[i, j, 0, -2:]
                    if c['binary_encoding']:
                        for k in range(c['length']):
                            train_mat_analog_vals[i, j, k] = bin_dec(train_input_analog[i, j, k, :c['length']])
                        train_mat_analog_vals[i, j, k + 1] = eof_sym
                        train_mat_analog_vals[i, j, k + 2: c['total_input_length']] = eof_sym + 1  # '?' symbol
                    else:
                        train_mat_analog_vals[i, j, :] = 1 + np.argmax(train_input_analog[i, j, :, :], axis=-1)

            train_strings = train_mat_analog_vals.reshape((n_iter * c['batch_size'], c['total_input_length']))
            train_ep_task = train_ep_task.reshape((n_iter * c['batch_size'], 2))

            cnt_doubles = 0
            for i in range(train_strings.shape[0]):
                seq = train_strings[i, :6]
                task_id = np.argmax(train_ep_task[i, :])

                str_seq = '.'.join([str(int(x)) for x in seq])
                str_seq = '.'.join([str(task_id), str_seq])
                if str_seq not in train_set:
                    train_set.add(str_seq)
                else:
                    cnt_doubles += 1
            print('Number of unique strings used for training: ', len(train_set))
            print("Number of strings that repeat: ", cnt_doubles)
            assert len(train_set) + cnt_doubles == n_iter * c['batch_size']
        ######

    if c['generate_dual_op_in_dataset']:
        assert c['multi_task'], "To run it on dual operation, the parameter multi_task has to be set to true."

    if c['generate_unseen_episodes']:
        assert not c['generate_dual_op_in_dataset'], "Unseen episodes and dual_op together - not implemented yet."

    if not c['multi_task']:
        assert c['binary_encoding'], "Single task and one-hot encoding not implemented yet."

    if c['debug']:
        print("============= !!!!!!IN DEBUG MODE!!!!!! =============")
        c['n_training_iterations'] = 10
        c['n_testing_iterations'] = 10

    c = sdict(c)  # Also makes c immutable

    del FLAGS

    print("Seed: %d" % c.seed)
    np.random.seed(c.seed)
    tf.set_random_seed(c.seed)

    datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
    if c.binary_encoding:
        name = 'Strings-Bin'
    else:
        name = 'Strings-OH'
    randnum = str(np.random.randint(1e5))

    sim_name = "{}-{}-{}".format('alif', randnum, datetime_suffix)
    root_dir = os.path.expanduser(os.path.join('/calc/ceca/', 'output', name))

    with SimManager(sim_name, root_dir, write_protect_dirs=False, tee_stdx_to='output.log') as simman:
        paths = simman.paths
        # Store config
        with open(os.path.join(paths.data_path, 'config.yaml'), 'w') as f:
            yaml.dump(c.todict(), f, allow_unicode=True, default_flow_style=False)

        # Open recorder
        datastore = ZarrDataStore(os.path.join(paths.results_path, 'data.mdb'))
        recorder = Recorder(datastore)
        print("Results will be stored in %s" % paths.results_path)

        # Run experiment
        main(train_set)

        # Close recorder
        recorder.close()
        print("Results stored in %s" % paths.results_path)
