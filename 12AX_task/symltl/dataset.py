import numpy as np
import rstr


def generate_string_12_ax_by(random_gen, n_tasks, length, width, binary_encoding):

    episode = ""
    # There are 8 possible symbols: 1, 2, A, B, C, X, Y, Z
    # Encoded as numbers:           0, 1, 2, 3, 4, 5, 6, 7
    for i in range(n_tasks):
        ss = rstr.xeger(r'[0-1][2-7]{1,10}((2[47]{0,6}5|3[47]{0,6}6)|([234][567])){1,2}')
        episode = episode + ss

    episode_to_int = [int(i) for i in episode]

    '''
    # Custom episode, for spike raster
    episode = ['0', '2', '3', '2', '4', '7', '5', '3', '7', '6', '1', '4', '7', '2', '4', '7', '5', '3', '7', '6', '1', 
     '2', '4', '3', '3', '6', '1', '2', '7', '3', '6', '6', '4', '5', '6', '3', '2', '6', '4', '6', '0', '4', '3', '3',
     '4', '7', '7', '4', '4', '4', '6', '2', '4', '4', '7', '4', '7', '5', '1', '4', '6', '5', '3', '3', '7', '5', '3', 
     '4', '7', '6', '4', '6', '0', '4', '4', '2', '4', '7', '5', '0', '2', '5', '2', '4', '7', '5', '3', '5', '3', '6']
    episode_to_int = [int(i) for i in episode]
    episode_to_int_check = [0, 2, 3, 2, 4, 7, 5, 3, 7, 6, 1, 4, 7, 2, 4, 7, 5, 3, 7, 6, 1, 2, 4, 3, 3, 6, 1, 2, 7, 3, 6, 
          6, 4, 5, 6, 3, 2, 6, 4, 6, 0, 4, 3, 3, 4, 7, 7, 4, 4, 4, 6, 2, 4, 4, 7, 4, 7, 5, 1, 4, 6, 5, 3, 3, 7, 5, 3, 4, 
          7, 6, 4, 6, 0, 4, 4, 2, 4, 7, 5, 0, 2, 5, 2, 4, 7, 5, 3, 5, 3, 6]
    assert episode_to_int == episode_to_int_check
    '''

    one_hot_input = np.zeros((length, width))
    one_hot_input[np.arange(length), episode_to_int[:length]] = 1

    target = generate_target_12_ax_by(episode[:length], length, binary_encoding)

    return {'input': one_hot_input, 'target': target}


def generate_target_12_ax_by(next_inputs, length, binary_encoding):
    last_num = ""
    last_letter = ""
    output = np.zeros((1, length))

    i = 0
    for next_input in next_inputs:
        if next_input in ["0", "1"]:
            last_num = next_input
            last_letter = ""
        elif next_input in ["2", "3"]:
            last_letter = next_input
        elif next_input in ["5", "6"]:
            seq = last_num + last_letter + next_input
            last_letter = next_input
            if seq in ["025", "136"]:  # 025 = 1AX, 136 = 2BY
                output[0, i] = 1
        i = i + 1

    if binary_encoding:
        target = output.reshape((-1, 1))
        target = target.astype(int)
    else:
        output = output.astype(int)
        target_one_hot = np.zeros((length, 2))
        target_one_hot[np.arange(length), output] = 1
        target = target_one_hot

    return target


def spike_encode(d, random_gen, n_dt_per_step, n_input_code, dt):
    """
    Assumes input is binary or rate between 0 and 1. Does rate coding between min and max rate. (200Hz and 2Hz resp.)
    :param d: d['input']:
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
    spikes = np.array([random_gen.binomial(1, probs) for _ in range(n_dt_per_step)])
    # Split along step axis and concatenate along spike time axis
    spike_input = np.squeeze(np.concatenate(np.split(spikes, indices_or_sections=spikes.shape[1], axis=1), axis=0))

    return {'spike_input': spike_input, 'target': d['target'], 'input': input_}
