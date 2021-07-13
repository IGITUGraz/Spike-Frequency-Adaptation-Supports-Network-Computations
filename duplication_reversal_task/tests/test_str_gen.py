from enum import Enum

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from randomgen import RandomGenerator

from sdict import sdict

from symltl import Timer, Task
from symltl.dataset import generate_string_multi_task, generate_string_single_task

c = sdict(dict(length=5, width=5, task=Task.COPY))

probs = tf.constant(np.repeat(0.5, c.width))


def _generate_string_tf(seed):
    """

    :return: input shape -> total_input_length x total_input_width == 2 * length + 1 x width + 2 == n_steps x input_size
    target shape -> n_steps x target_size == n_steps x input_size
    """
    tf.set_random_seed(seed)

    # Actual input
    strings_input = tf.concat(
        (tf.cast(tf.distributions.Bernoulli(probs=probs).sample(c.length), dtype=tf.float32),
         tf.zeros((c.length + 1, c.width))), axis=0)
    # Cue for EOS
    eos_input = tf.concat((tf.zeros((c.length, 1)), tf.ones((1, 1)), tf.zeros((c.length, 1))), axis=0)
    # Cue for output
    q_input = tf.concat((tf.zeros((c.length + 1, 1)), tf.ones((c.length, 1))), axis=0)
    input_ = tf.cast(tf.concat((strings_input, eos_input, q_input), axis=1), dtype=tf.float32)
    if c.task == Task.COPY:
        target = input_[:c.length, :-2]  # Exclude the cues for eos and output
    elif c.task == Task.REVERSE:
        target = tf.reverse(input_[:c.length, :-2], axis=[0])  # Exclude the cues for eos and output
    else:
        raise RuntimeError("Unknown task %s" % c.task.value)

    with tf.Session() as sess:
        input_v, target_v = sess.run([input_, target])
    return {'input': input_v, 'target': target_v}


def _generate_string_np(seed):
    """

    :return: input shape -> total_input_length x total_input_width == 2 * length + 1 x width + 2 == n_steps x input_size
    target shape -> n_steps x target_size == n_steps x input_size
    """
    rg = RandomGenerator()
    rg.seed(seed)

    # Actual input
    strings_input = np.concatenate(
        (rg.choice([0, 1], (c.length, c.width), replace=True), np.zeros((c.length + 1, c.width))), axis=0)
    # Cue for EOS
    eos_input = np.concatenate((np.zeros((c.length, 1)), np.ones((1, 1)), np.zeros((c.length, 1))), axis=0)
    # Cue for output
    q_input = np.concatenate((np.zeros((c.length + 1, 1)), np.ones((c.length, 1))), axis=0)
    input_ = np.concatenate((strings_input, eos_input, q_input), axis=1).astype(np.float32)
    if c.task == Task.COPY:
        target = input_[:c.length, :-2]  # Exclude the cues for eos and output
    elif c.task == Task.REVERSE:
        target = np.flip(input_[:c.length, :-2], axis=0)  # Exclude the cues for eos and output
    else:
        raise RuntimeError("Unknown task %s" % c.task.value)
    return {'input': input_, 'target': target}


def test_generate_string_np():
    rg = RandomGenerator()
    task = Task.REVERSE
    str_ = generate_string_single_task(rg, c.length, c.width, task)

    print("Length: {}, Width: {}, Input shape: {}, Target shape: {}".format(c.length, c.width, str_['input'].shape,
                                                                            str_['target'].shape))

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    ax.imshow(str_['input'].T, origin='lower')
    ax = axs[1]
    ax.set(title="Task %s" % task.value)
    ax.imshow(str_['target'].T, origin='lower')
    plt.tight_layout()
    plt.show()


def test_generate_string_tf():
    str_ = _generate_string_tf(3000)

    fig, ax = plt.subplots()
    ax.imshow(str_['input'].T)
    plt.show()


def test_generate_string_multi_task():
    rg = RandomGenerator()
    str_ = generate_string_multi_task(rg, c.length, c.width)
    task = [Task.COPY, Task.REVERSE][str_['task_idx']]

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    ax.imshow(str_['input'].T)
    ax = axs[1]
    ax.set(title="Task %s" % task.value)
    ax.imshow(str_['target'].T)
    plt.tight_layout()
    plt.show()


def time_generate_string_np_vs_tf():
    print("Testing Tensorflow version")
    with Timer() as bt:
        with tf.Session() as sess:
            for i in range(1000):
                a = _generate_string_tf(3000)
    print("Time taken: ", bt.difftime)

    print("Testing NumPy version")
    with Timer() as bt:
        for i in range(1000):
            a = _generate_string_np(3000)
    print("Time taken: ", bt.difftime)


if __name__ == '__main__':
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        test_generate_string_np()
        # test_generate_string_multi_task()
