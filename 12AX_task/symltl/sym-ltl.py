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
from symltl import Timer
from symltl.dataset import generate_string_12_ax_by, spike_encode
from symltl.ftools import multiprocify, generatorify, fchain, fify

from tensorflow.core.protobuf import rewriter_config_pb2
config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.memory_optimization = off


def record_dict(prefix, d):
    for k, v in d.items():
        recorder.record('{}/{}'.format(prefix, k), np.array(v))


@tf.custom_gradient
def BA_out(psp, W_out, BA_out):
    logits = tf.einsum('bij,jk->bik', psp, W_out)
    def grad(dy):
        dloss_dw_out = tf.einsum('btj,btk->jk', psp, dy)
        dloss_dba_out = tf.zeros_like(BA_out)
        dloss_dpsp = tf.einsum('bik,jk->bij', dy, BA_out)
        return [dloss_dpsp, dloss_dw_out, dloss_dba_out]
    return logits, grad


def main():
    rg = RandomGenerator()
    rg.seed(c.seed)

    if c.debug:
        mfy = generatorify
    else:
        mfy = multiprocify

    str_gen_fn = fify(generate_string_12_ax_by, n_tasks=c.n_tasks, length=c.length, width=c.width,
                      binary_encoding=c.binary_encoding)

    input_time_steps = c.total_input_length * c.spiking_params.n_dt_per_step
    gen_fn = mfy(fchain(
            fify(spike_encode, n_dt_per_step=c.spiking_params.n_dt_per_step,
                 n_input_code=c.spiking_params.n_input_code,
                 dt=c.spiking_params.dt),
            str_gen_fn), seed=c.seed)
    tf_dataset = tf.data.Dataset.from_generator(
            gen_fn,
            {'spike_input': tf.float32, 'input': tf.float32, 'target': tf.float32},
            {'spike_input': (input_time_steps, c.total_input_width * c.spiking_params.n_input_code),
             'input': (c.total_input_length, c.total_input_width), 'target': (c.target_length, c.target_width)},
        ) \
            .batch(c.batch_size) \
            .prefetch(100)
    iterator = tf_dataset.make_one_shot_iterator()
    values = iterator.get_next()
    input_, target_ = values['spike_input'], values['target']
    input_analog_ = values['input']

    # Second dimension is time
    X = tf.placeholder_with_default(input_, [c.batch_size, input_time_steps,
                                             c.total_input_width * c.spiking_params.n_input_code], name='X')
    Y = tf.placeholder_with_default(target_, [c.batch_size, c.target_length, c.target_width], name='Y')

    n_regular = int(c.n_hidden * (1. - c.alif_params.adaptive_fraction))
    n_adaptive = int(c.n_hidden - n_regular)
    assert n_regular + n_adaptive == c.n_hidden

    beta = np.concatenate([np.zeros(n_regular), np.ones(n_adaptive) * c.alif_params.beta])
    tau_adaptation = rg.uniform(1, c.alif_params.tau_adaptation, c.n_hidden)

    arr_noise = np.zeros(c.n_hidden)
    # Figure S7 and S8

    # Noise to only one neuron:
    # idx_neuron = 193
    # arr_noise[idx_neuron] = 0.05

    # Noise to all neurons:
    # # arr_noise[:] = 0.5

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
        injected_noise_current=arr_noise,
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

    [new_z, new_v, thr], new_state = tf.nn.dynamic_rnn(cell, X,
                                             initial_state=cell.zero_state(c.batch_size, tf.float32))

    cell_outputs_all_t = new_z

    av = tf.divide(tf.reduce_mean(new_z, axis=(0, 1)), c.spiking_params.dt)
    loss_av = tf.reduce_mean(tf.square(av - c.spiking_params.reg_rate))

    # The last 'length' outputs are taken later
    if not c.use_avg_activity:
        cell_outputs_all_t = exp_convolve(cell_outputs_all_t, 0.9)
        cell_outputs = cell_outputs_all_t[:, (c.spiking_params.n_dt_per_step - 1)::c.spiking_params.n_dt_per_step, :]
        # print('Cell_outputs: ', cell_outputs.shape)
    else:
        # print('Cell_outputs_t: ', cell_outputs_all_t.shape)
        split_outputs = tf.split(cell_outputs_all_t, c.length, axis=1)
        cell_outputs = tf.reduce_mean(split_outputs, axis=2)
        cell_outputs = tf.transpose(cell_outputs, perm=(1, 0, 2))
        # print('Cell_outputs: ', cell_outputs.shape)

    if not c.eprop: # BPTT
        output_1 = tf.einsum('ijk,kl->ijl', cell_outputs, w_out) + b_out
    elif not c.random_eprop: # Symmetric e-prop
        output_1 = tf.einsum('ijk,kl->ijl', cell_outputs, w_out)
    else: # Random e-prop
        random_eprop_bstd = 2.0345
        ba_out = tf.constant(random_eprop_bstd * rg.randn(c.n_hidden, c.n_out), dtype=tf.float32, name='BAOut')
        # output_1 = BA_out(cell_outputs, w_out, ba_out) + b_out
        output_1 = BA_out(cell_outputs, w_out, ba_out)

    if not c.binary_encoding: # One-hot encoding of inputs, one for L and R
        probs = tf.nn.softmax(output_1)
    else:
        probs = tf.sigmoid(output_1) # One output neuron, thresholded gives either 0 or 1 (L or R)

    assert probs.shape == (c.batch_size, c.total_input_length, c.n_out)
    relevant_probs = probs #[:, -c.target_length:, :]  # In this case, all of them
    print('relevant_probs: ', relevant_probs.shape)

    if not c.binary_encoding:
        actual_output = tf.argmax(relevant_probs, axis=-1)
        # print('actual output: ', actual_output.shape)
    else:
        actual_output = tf.where(relevant_probs < 0.5, tf.zeros_like(relevant_probs), tf.ones_like(relevant_probs))

    assert relevant_probs.shape == Y.shape, "%s, %s" % (relevant_probs.shape, Y.shape)

    if not c.binary_encoding:
        loss = -tf.reduce_mean(Y * tf.log(relevant_probs + 1e-8))
    else:
        loss = -tf.reduce_mean(Y * tf.log(relevant_probs + 1e-8) + (1. - Y) * tf.log(1. - relevant_probs + 1e-8))

    # Add the tern for average firing rate regularization
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

    if not c.binary_encoding: # One-hot encoding of outputs
        success_rate = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(actual_output, tf.argmax(Y, axis=-1)), axis=-1),
                                              dtype=tf.float32))
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

    input_analog_to_save = {'input_analog': input_analog_}
    iostates = {'input': input_, 'all_probs': probs, 'input_analog': input_analog_, 'target': target_,
                'actual_output': actual_output, 'states': new_z,
                # 'cell_outputs': cell_outputs_all_t, 'thr': thr, 'voltage_traces': new_v  # Needed for spike rasters, saved only during testing
                }

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config_proto) as sess:
        if c.restore_from != '':
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(c.restore_from, 'results')))
        else:
            sess.run(init)

        with open(os.path.join(paths.data_path, 'cell_params.pkl'), 'wb') as f:
            pickle.dump(dict(tau_adaptation=tau_adaptation, w_in_delay=sess.run(cell.w_in_delay),
                             w_rec_delay=sess.run(cell.w_rec_delay)), f, protocol=pickle.HIGHEST_PROTOCOL)

        # TRAIN
        for i in range(c.n_training_iterations):

            with Timer() as bt:
                _, metrics_val, inputs_analog_vals = sess.run([opt, metrics, input_analog_to_save])

            with Timer() as st:
                record_dict('train', metrics_val)
                record_dict('train', inputs_analog_vals)

            if i == 1 or i % 100 == 0 or i == c.n_training_iterations - 1:
                print(
                    "Training iteration {:d} :: Storage time was {:.4f} :: Batch time was {:.4f}".format(i, st.difftime,
                                                                                                         bt.difftime))
                print("Loss is {:.4f}; Symbol success rate is: {:.4f}; Success rate is: {:.4f}".format(
                    metrics_val['loss'], metrics_val['symbol_success_rate'], metrics_val['success_rate']))

                weight_results = sess.run(weight_ops)
                record_dict('weight', weight_results)

                if is_rewiring_enabled:
                    # Print rewiring connectivity
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

            if i % 100 == 0 or i == c.n_testing_iterations - 1:
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
    flags.DEFINE_bool("debug", False, "Debug mode (output stored to tmp, only run for 100 iterations)")

    flags.DEFINE_string('restore_from', '', 'Restore model from')
    flags.DEFINE_integer("seed", 3000, "Seed to use. Default 3000.")

    # Task related
    flags.DEFINE_integer("n_tasks", 23, "Number of possible tasks in one episode")
    flags.DEFINE_integer("length", 90, "Length of input sequence")
    flags.DEFINE_integer("width", 8, "Number of input symbols")
    flags.DEFINE_bool("binary_encoding", False, "Outputs binary encoded?")
    flags.DEFINE_bool("use_avg_activity", True, "Use average activity for the output? Otherwise an exp. kernel is used.")
    flags.DEFINE_integer("training_iterations", 10000, "Number of training iterations")
    flags.DEFINE_integer("n_hidden", 200, "Number of hidden neurons to use")

    # Spiking (input) related
    flags.DEFINE_integer("n_dt_per_step", 500, "Regularization firing rate target")
    flags.DEFINE_integer("n_input_code", 5, "Number of spiking neurons used to encode each input")  # default is 10

    # ALIF (ALIF = adaptive LIF, i.e., LIF with SFA) related
    flags.DEFINE_float("tau_adaptation", 13500., "Adaptation time constant in ms")
    flags.DEFINE_float("adaptive_fraction", 0.5, "Fraction of neurons that are adaptive")

    flags.DEFINE_float("dampening_factor", 0.3, "Dampening factor")
    flags.DEFINE_float("reg_hp", 15., "Regularization factor for spiking rate")
    flags.DEFINE_float("reg_rate_hz", 10., "Regularization firing rate target")

    # Rewiring related
    # Set neuron sign
    tf.app.flags.DEFINE_bool('neuron_sign', False, #True, #False,
                             'If rewiring is active, this will fix the sign of input and recurrent neurons')
    tf.app.flags.DEFINE_float('proportion_excitatory', 0.8, 'proportion of excitatory neurons')

    flags.DEFINE_float('rewiring_connectivity', -1, # -1 = all-to-all connections, otherwise a value between 0 and 1
                       'possible usage of rewiring with ALIF and LIF (disabled by default)')
    flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
    flags.DEFINE_float('rewiring_l1', 1e-2, 'l1 regularization that goes with rewiring (irrelevant without rewiring)')

    tf.app.flags.DEFINE_bool('eprop', False, 'Enable eprop?')
    tf.app.flags.DEFINE_bool('random_eprop', False, 'Enable random eprop?')

    FLAGS = flags.FLAGS

    # Read out and process parameters from FLAGS
    # BASIC params
    c = dict(
        debug=FLAGS.debug,
        eprop=(FLAGS.eprop or FLAGS.random_eprop),
        random_eprop=FLAGS.random_eprop,
        restore_from=os.path.expanduser(FLAGS.restore_from),
        seed=FLAGS.seed,
        n_tasks=FLAGS.n_tasks,
        length=FLAGS.length,  # Also number of time steps
        width=FLAGS.width,
        binary_encoding=FLAGS.binary_encoding,
        use_avg_activity=FLAGS.use_avg_activity,
        batch_size=20,
        n_hidden=FLAGS.n_hidden,
        n_training_iterations=FLAGS.training_iterations,
        n_testing_iterations=100,
        spiking_params=dict(
            dt=1.,
            n_input_code=FLAGS.n_input_code,
            n_dt_per_step=FLAGS.n_dt_per_step,
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
            beta=1.7,
            ), )

    c.update(dict(total_input_length=c['length']))
    c.update(dict(total_input_width=c['width']))

    if not FLAGS.binary_encoding:
        c.update(dict(target_length=c['length'], target_width=2))
    else:
        c.update(dict(target_length=c['length'], target_width=1))

    c.update(dict(n_out=c['target_width']))

    if c['restore_from'] != '':
        print('=== RESTORING FROM {}'.format(c['restore_from']))
        with open(os.path.join(c['restore_from'], 'data', 'config.yaml'), 'r') as f:
            tc = yaml.load(f)
        no_update_keys = ['n_training_iterations', 'batch_size', 'restore_from', 'seed', 'n_testing_iterations']
        for k, v in tc.items():
            if k not in no_update_keys:
                c[k] = v

    c = sdict(c)  # Also makes c immutable

    del FLAGS

    print("Seed: %d" % c.seed)
    np.random.seed(c.seed)
    tf.set_random_seed(c.seed)

    datetime_suffix = datetime.now().strftime("D%d-%m-%Y-T%H-%M-%S")
    name = 'AX-BY-eprop'
    randnum = str(np.random.randint(1e5))

    sim_name = "{}-{}-{}".format("ALIF", randnum, datetime_suffix) # ALIF = adaptive LIF, i.e., LIF with SFA
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
        main()

        # Close recorder
        recorder.close()
        print("Results stored in %s" % paths.results_path)