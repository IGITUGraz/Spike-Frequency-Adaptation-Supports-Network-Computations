import numpy as np
from symltl import Task


def bin_dec(a):
    return sum([int(a[-i]) * 2 ** (i - 1) for i in range(1, len(a) + 1)])


def generate_string_single_task(random_gen, length, width, task):

    # Unpackbits always produces 8 bit output. So need to take out the last (most significant) bits.
    # Note that width cannot be larger than 8 here!
    str_ = np.unpackbits(random_gen.choice(range(1, 2 ** width), length).astype(np.uint8).reshape(-1, 1), axis=1)[:, -width:]
    strings_input = np.concatenate((str_, np.zeros((length + 2, width))), axis=0)

    # Cue for EOS
    eos_input = np.concatenate((np.zeros((length, 1)), np.ones((1, 1)), np.zeros((length + 1, 1))), axis=0)

    # Cue for output
    q_input = np.concatenate((np.zeros((length + 1, 1)), np.ones((length + 1, 1))), axis=0)
    input_ = np.concatenate((strings_input, eos_input, q_input), axis=1).astype(np.float32)

    if task == Task.COPY:
        target = input_[:(length + 1), :-1]  # Exclude the cues for output
    elif task == Task.REVERSE:
        # Exclude the cues for output, but include the one for EOS separately
        target = np.concatenate((np.flip(input_[:length, :-1], axis=0), input_[length:(length + 1), :-1]), axis=0)
    else:
        raise RuntimeError("Unknown task %s" % task.value)
    return {'input': input_, 'target': target}


def generate_unseen_string_multi_task(random_gen, length, width, binary_encoding, task_cue_given_once, train_set):

    n_len = 5
    eof_sym = 2 ** n_len

    values = generate_string_multi_task(random_gen, length, width, binary_encoding, task_cue_given_once)
    input_, target_ = values['input'], values['target']

    seq = np.zeros((n_len + 1, 1))
    ep_task = input_[0, -2:]

    if binary_encoding:
        for k in range(n_len):
            seq[k] = bin_dec(input_[k, :n_len])
        seq[k + 1] = eof_sym
    else:
        seq = 1 + np.argmax(input_, axis=-1)

    task_id = np.argmax(ep_task)

    str_seq = '.'.join([str(int(x)) for x in seq])
    str_seq = '.'.join([str(task_id), str_seq])

    while str_seq in train_set:
        values = generate_string_multi_task(random_gen, length, width, binary_encoding, task_cue_given_once)
        input_, target_ = values['input'], values['target']

        seq = np.zeros((n_len + 1, 1))
        ep_task = input_[0, -2:]

        if binary_encoding:
            for k in range(n_len):
                seq[k] = bin_dec(input_[k, :n_len])
            seq[k + 1] = eof_sym
        else:
            seq = 1 + np.argmax(input_, axis=-1)

        task_id = np.argmax(ep_task)

        str_seq = '.'.join([str(int(x)) for x in seq])
        str_seq = '.'.join([str(task_id), str_seq])

    train_set.add(str_seq)
    return values


def generate_string_multi_task(random_gen, length, width, binary_encoding, task_cue_given_once):
    """

    :return: input shape -> total_input_length x total_input_width == 2 * length + 1 x width + 2 == n_steps x input_size
    target shape -> n_steps x target_size == n_steps x input_size
    """

    if binary_encoding:
        # Actual input
        str_ = np.unpackbits(random_gen.choice(range(1, 2 ** width), length).astype(np.uint8).reshape(-1, 1), axis=1)[:, -width:]
        strings_input = np.concatenate((str_, np.zeros((length + 2, width))), axis=0)
    else:  # One-hot encoding of inputs and outputs
        # Actual input
        symbols = random_gen.randint(1, 2 ** width, size=(1, length))
        symbols_ = np.zeros((length, 2 ** width - 1))
        symbols_[np.arange(length), symbols - 1] = 1

        strings_input = np.concatenate((symbols_, np.zeros((length + 2, 2 ** width - 1))), axis=0)

    # Cue for EOS
    eos_input = np.concatenate((np.zeros((length, 1)), np.ones((1, 1)), np.zeros((length + 1, 1))), axis=0)
    # Cue for output
    q_input = np.concatenate((np.zeros((length + 1, 1)), np.ones((length + 1, 1))), axis=0)
    # Task cue
    if not task_cue_given_once:
        task_input = np.concatenate((np.ones((length + 1, 1)), np.zeros((length + 1, 1))), axis=0)
    else:  # Task cue given only once
        task_input = np.zeros((2 * length + 2, 1))
        task_input[0] = 1

    # input_ = np.concatenate((strings_input, eos_input, q_input), axis=1).astype(np.float32)
    task_idx = random_gen.randint(2)
    task = [Task.COPY, Task.REVERSE][task_idx]
    if task == Task.COPY:
        # Note order in which task_input is concatenated
        input_ = np.concatenate((strings_input, eos_input, q_input, task_input, np.zeros((2 * length + 2, 1))),
                                axis=1).astype(np.float32)
        target = input_[:(length + 1), :-3]  # Exclude the cues for output, and two task cue channels
    elif task == Task.REVERSE:
        # Note order in which task_input is concatenated
        input_ = np.concatenate((strings_input, eos_input, q_input, np.zeros((2 * length + 2, 1)), task_input),
                                axis=1).astype(np.float32)
        target = np.concatenate((np.flip(input_[:length, :-3], axis=0), input_[length:(length + 1), :-3]),
                                axis=0)  # Exclude the cues for output, and two task cue channels
    else:
        raise RuntimeError("Unknown task %s" % task.value)
    return {
        'input': input_,
        'target': target,
        # 'task_idx': task_idx
    }


def spike_encode(d, random_gen, n_dt_per_step, n_input_code, dt):
    """
    Assumes input is binary or rate between 0 and 1. Does rate coding between min and max rate. (200Hz and 2Hz resp.)
    :param d: d['input']: shape --> total_input_length x total_input_width == 2 * length + 1 x width + 2 == n_steps x input_size
    :param random_gen:
    :param n_dt_per_step:
    :param n_input_code:
    :param dt:
    :return: spike_input shape --> n_dt_per_step * n_steps x input_size
    target shape -- n_steps x input_size :: NOTE this
    """

    assert dt == 1.
    max_rate_hz = 200
    min_rate_hz = 2
    input_ = d['input']

    probs = (input_ * (max_rate_hz - min_rate_hz) + min_rate_hz) * 1e-3  # ms-1
    probs = np.repeat(probs, repeats=n_input_code, axis=1)

    ##
    # if use_additional_input_neurons:
    #     n_rnd_neurons = 20
    #     table = [[0,12,4,13,1],[6,2,4,3,12],[16,3,12,19,0],[9,17,19,14,10],[1,9,8,5,18],[15,0,7,5,14],[15,0,17,2,5],
    #              [16,19,5,2,3],[5,10,6,19,11],[12,5,0,14,1],[13,15,16,0,9],[2,8,17,7,0],[0,10,16,11,15],[7,0,15,9,11],
    #              [8,4,14,0,5],[12,1,19,6,15],[11,15,2,16,6],[10,13,7,9,16],[4,9,19,15,16],[10,2,3,1,18],[9,7,10,13,1],
    #              [13,2,3,12,6],[14,11,2,17,8],[16,8,2,18,12],[7,14,15,8,5],[12,7,8,1,11],[9,15,18,4,14],[16,2,12,18,8],
    #              [2,6,13,9,3],[15,17,16,6,18],[3,15,16,9,4]]
    #
    #     mat_rnd_neurons = np.zeros((12, n_rnd_neurons))  # 12 steps in total
    #     for i in range(5):  # 5 steps with symbols presented
    #         mat_rnd_neurons[i, table[np.argmax(input_, axis=1)[i]]] = 1
    #
    #     probs_2 = (mat_rnd_neurons * (max_rate_hz - min_rate_hz) + min_rate_hz) * 1e-3  # ms-1
    #     probs = np.concatenate((probs, probs_2), axis=1)
    ###

    spikes = np.array([random_gen.binomial(1, probs) for _ in range(n_dt_per_step)])
    # Split along step axis and concatenate along spike time axis
    spike_input = np.squeeze(np.concatenate(np.split(spikes, indices_or_sections=spikes.shape[1], axis=1), axis=0))

    return {'spike_input': spike_input, 'target': d['target'], 'input': input_}


def generate_string_multi_task_dual(random_gen, length, width, binary_encoding, task_cue_given_once):
    """

    :return: input shape -> total_input_length x total_input_width == 2 * length + 1 x width + 2 == n_steps x input_size
    target shape -> n_steps x target_size == n_steps x input_size
    """

    if binary_encoding:
        # Actual input
        str_ = np.unpackbits(random_gen.choice(range(1, 2 ** width), length).astype(np.uint8).reshape(-1, 1), axis=1)[:, -width:]
        strings_input = np.concatenate((str_, np.zeros((length + 2, width))), axis=0)
    else:  # One-hot encoding of inputs and outputs
        # Actual input
        symbols = random_gen.randint(1, 2 ** width, size=(1, length))
        symbols_ = np.zeros((length, 2 ** width - 1))
        symbols_[np.arange(length), symbols - 1] = 1

        strings_input = np.concatenate((symbols_, np.zeros((length + 2, 2 ** width - 1))), axis=0)

    # Cue for EOS
    eos_input = np.concatenate((np.zeros((length, 1)), np.ones((1, 1)), np.zeros((length + 1, 1))), axis=0)
    # Cue for output
    q_input = np.concatenate((np.zeros((length + 1, 1)), np.ones((length + 1, 1))), axis=0)
    # Task cue
    if not task_cue_given_once:
        task_input = np.concatenate((np.ones((length + 1, 1)), np.zeros((length + 1, 1))), axis=0)
    else:  # Task cue given only once
        task_input = np.zeros((2 * length + 2, 1))
        task_input[0] = 1

    # input_ = np.concatenate((strings_input, eos_input, q_input), axis=1).astype(np.float32)

    # Copy task
    # Note order in which task_input is concatenated
    copy_input_ = np.concatenate((strings_input, eos_input, q_input, task_input, np.zeros((2 * length + 2, 1))),
                                 axis=1).astype(np.float32)
    copy_target = copy_input_[:(length + 1), :-3]  # Exclude the cues for output, and two task cue channels

    # Reversal task
    # Note order in which task_input is concatenated
    reversal_input_ = np.concatenate((strings_input, eos_input, q_input, np.zeros((2 * length + 2, 1)), task_input),
                                     axis=1).astype(np.float32)
    reversal_target = np.concatenate((np.flip(reversal_input_[:length, :-3], axis=0), reversal_input_[length:(length + 1), :-3]),
                                     axis=0)  # Exclude the cues for output, and two task cue channels

    return {
        'input': np.concatenate((np.expand_dims(copy_input_, axis=0), np.expand_dims(reversal_input_, axis=0)), axis=0),
        'target': np.concatenate((np.expand_dims(copy_target, axis=0), np.expand_dims(reversal_target, axis=0)), axis=0)
    }


def spike_encode_dual(d, random_gen, n_dt_per_step, n_input_code, dt):
    """
    Assumes input is binary or rate between 0 and 1. Does rate coding between min and max rate. (200Hz and 2Hz resp.)
    :param d: d['input']: shape --> total_input_length x total_input_width == 2 * length + 1 x width + 2 == n_steps x input_size
    :param random_gen:
    :param n_dt_per_step:
    :param n_input_code:
    :param dt:
    :return: spike_input shape --> n_dt_per_step * n_steps x input_size
    target shape -- n_steps x input_size :: NOTE this
    """

    assert dt == 1.
    max_rate_hz = 200
    min_rate_hz = 2
    input_ = d['input']

    probs = (input_[0] * (max_rate_hz - min_rate_hz) + min_rate_hz) * 1e-3  # ms-1
    probs = np.repeat(probs, repeats=n_input_code, axis=1)
    spikes_copy = np.array([random_gen.binomial(1, probs) for _ in range(n_dt_per_step)])
    # Split along step axis and concatenate along spike time axis

    probs = (input_[1] * (max_rate_hz - min_rate_hz) + min_rate_hz) * 1e-3  # ms-1
    probs = np.repeat(probs, repeats=n_input_code, axis=1)
    spikes_reversal = np.array([random_gen.binomial(1, probs) for _ in range(n_dt_per_step)])

    spike_input_id = np.squeeze(np.concatenate(np.split(spikes_copy, indices_or_sections=spikes_copy.shape[1], axis=1), axis=0))
    spike_input_rev = np.squeeze(np.concatenate(np.split(spikes_reversal, indices_or_sections=spikes_reversal.shape[1], axis=1), axis=0))
    spike_input = np.concatenate((np.expand_dims(spike_input_id, axis=0), np.expand_dims(spike_input_rev, axis=0)), axis=0)

    return {'spike_input': spike_input, 'target': d['target'], 'input': input_}

