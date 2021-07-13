import numpy as np
import numpy.random as rd
import tensorflow as tf
from matplotlib import collections as mc, patches
from lsnn.guillaume_toolbox.matplotlib_extension import strip_right_top_axis, raster_plot, hide_bottom_axis
# from tqdm import tqdm


# Variations of sequence with specific delay for plotting
def gen_custom_delay_batch(seq_len, seq_delay, batch_size):
    assert type(seq_delay) is int
    assert 2 + 1 + seq_delay + 1 < seq_len

    def gen_custom_delay_input(seq_len, seq_delay):
        seq_delay = 1 + np.random.choice(seq_len - 2) if seq_delay == 0 else seq_delay
        return [np.random.choice([0, 1]) for _ in range(2)] + \
               [2] + [np.random.choice([0, 1]) for _ in range(seq_delay)] + [3] + \
               [np.random.choice([0, 1]) for _ in range(seq_len - (seq_delay + 4))]

    return np.array([gen_custom_delay_input(seq_len, seq_delay) for i in range(batch_size)])


def error_rate(z, num_Y, num_X, n_character):
    # Find the recall index
    n_recall_symbol = n_character + 1
    shp = tf.shape(num_X)

    # Translate the one hot into ints
    char_predict = tf.argmax(z, axis=2)
    char_true = num_Y
    char_input = num_X

    # error rate 1) Wrong characters
    char_correct = tf.cast(tf.equal(char_predict, char_true), tf.float32)
    character_errors = tf.reduce_mean(1 - char_correct)

    # error rate 2) wrong recalls
    recall_mask = tf.equal(char_input, n_recall_symbol)
    recalls_predict = tf.boolean_mask(char_predict, recall_mask)
    recalls_true = tf.boolean_mask(char_true, recall_mask)

    recall_correct = tf.equal(recalls_predict, recalls_true)
    recall_errors = tf.reduce_mean(tf.cast(tf.logical_not(recall_correct), tf.float32))

    # Get wrong samples
    sentence_id = tf.tile(tf.expand_dims(tf.range(shp[0]), axis=1), (1, shp[1]))
    recall_sentence_id = tf.boolean_mask(sentence_id, recall_mask)
    false_sentence_id_list = tf.boolean_mask(recall_sentence_id, tf.logical_not(recall_correct))

    return character_errors, recall_errors, false_sentence_id_list


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern,list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes


def validity_test(seq, recall_char, store_char, contains_digit=True):
    is_valid = True

    # At least a store, a digit and a recall
    if np.max(seq == recall_char) == 0 or np.max(seq == store_char) == 0 or \
            (np.max(seq < store_char) == 0 and contains_digit):
        is_valid = False

    # First store before first recall
    t_first_recall = np.argmax(seq == recall_char)
    t_first_store = np.argmax(seq == store_char)
    if t_first_recall < t_first_store:
        is_valid = False

    # Last recall after last store
    t_last_recall = - np.argmax(seq[::-1] == recall_char)
    t_last_store = - np.argmax(seq[::-1] == store_char)
    if t_last_recall < t_last_store:
        is_valid = False

    # Always a digit after a store
    t_store_list = np.where(seq == store_char)[0]
    for t_store in t_store_list:
        if t_store == seq.size - 1 or seq[t_store + 1] in [recall_char, store_char]:
            is_valid = False
            break

    # Between two recall there is a store
    t_recall_list = np.where(seq == recall_char)[0]
    for k, t_recall in enumerate(t_recall_list[:-1]):
        next_t_recall = t_recall_list[k + 1]

        is_store_between = np.logical_and(t_recall < t_store_list, t_store_list < next_t_recall)
        if not (is_store_between.any()):
            is_valid = False

    # Between two store there is a recall
    for k, t_store in enumerate(t_store_list[:-1]):
        next_t_store = t_store_list[k + 1]

        is_recall_between = np.logical_and(t_store < t_recall_list, t_recall_list < next_t_store)
        if not (is_recall_between.any()):
            is_valid = False
    return is_valid


def generate_input_with_prob(batch_size, length, recall_char, store_char, prob_bit_to_store,
                             prob_bit_to_recall):
    input_nums = np.zeros((batch_size, length), dtype=int)

    for b in range(batch_size):
        last_signal = recall_char

        # init a sequence
        is_valid = False
        seq = rd.choice([0, 1], size=length)

        while not is_valid:
            seq = rd.choice([0, 1], size=length)
            for t in range(length):
                # If the last symbol is a recall we wait for a store
                if last_signal == recall_char and rd.rand() < prob_bit_to_store:
                    seq[t] = store_char
                    last_signal = store_char

                # Otherwise we wait for a recall
                elif last_signal == store_char and rd.rand() < prob_bit_to_recall:
                    seq[t] = recall_char
                    last_signal = recall_char

            is_valid = validity_test(seq, recall_char, store_char)

        input_nums[b, :] = seq

    return input_nums


def generate_data(batch_size, length, n_character, prob_bit_to_store=1. / 3, prob_bit_to_recall=1. / 5, input_nums=None,
                  with_prob=True, delay=None):

    store_char = n_character
    recall_char = n_character + 1

    # Generate the input data
    if input_nums is None:
        if with_prob and prob_bit_to_store < 1. and prob_bit_to_recall < 1.:
            input_nums = generate_input_with_prob(batch_size, length, recall_char, store_char,
                                                  prob_bit_to_store, prob_bit_to_recall)
        else:
            raise ValueError("Only use input generated with probabilities")

    input_nums = np.array(input_nums)

    # generate the output
    target_nums = input_nums.copy()
    inds_recall = np.where(input_nums == recall_char)
    for k_trial, k_t in zip(inds_recall[0], inds_recall[1]):
        assert k_t > 0, 'A recall is put at the beginning to avoid this'
        store_list = np.where(input_nums[k_trial, :k_t] == store_char)[0]
        previous_store_t = store_list[-1]
        target_nums[k_trial, k_t] = input_nums[k_trial, previous_store_t + 1]

    memory_nums = np.ones_like(input_nums) * store_char
    for k_trial in range(batch_size):
        t_store_list = np.where(input_nums[k_trial, :] == store_char)[0]
        for t_store in np.sort(t_store_list):
            if t_store < length - 1:
                memory_nums[k_trial, t_store:] = input_nums[k_trial, t_store + 1]

    return input_nums, target_nums, memory_nums


def generate_mikolov_data(batch_size, length, n_character, with_prob, prob_bit_to_recall,
                          prob_bit_to_store, override_input=None, delay=None):
    if n_character > 2:
        raise NotImplementedError("Not implemented for n_character != 2")
    total_character = n_character + 2
    recall_character = total_character - 1
    store_character = recall_character - 1
    store = np.zeros((batch_size, length), dtype=float)
    recall = np.zeros((batch_size, length), dtype=float)
    channels = [np.zeros((batch_size, length), dtype=float) for _ in range(n_character)] + [store, recall]
    input_nums, target_nums, memory_nums = generate_data(batch_size, length, n_character,
                                                         with_prob=with_prob, prob_bit_to_recall=prob_bit_to_recall,
                                                         prob_bit_to_store=prob_bit_to_store, input_nums=override_input,
                                                         delay=delay)
    # input_nums: (batch, length) every is sequence of sym chars, example:
    # [1 0 2 1 0 1 3 0 0 1 1 0 0 0 2 0 1 3 1 1]
    for c in range(total_character):
        channels[c] = np.isin(input_nums, [c]).astype(int)

    for b in range(batch_size):
        for i in range(length):
            if channels[store_character][b,i] == 1:
                # copy next input to concurrent step with store
                channels[0][b,i] = channels[0][b,i+1]
                channels[1][b,i] = channels[1][b,i+1]
                # sometimes inverse the next input
                if rd.uniform() < 0.5:
                    channels[0][b, i+1] = 1 - channels[0][b, i + 1]
                    channels[1][b, i+1] = 1 - channels[1][b, i + 1]
    # channels: (batch, channel, length)
    # example of channel, length for batch 0:
    # array([[0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    #        [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    #        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1],
    #        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]])
    return channels, target_nums, memory_nums, input_nums


def generate_storerecall_signals_with_prob(length, prob):
    """
    Generate valid store/recall signal sequences based on probability of signals appearing.
    """
    recall_char = 2
    store_char = 3
    last_signal = recall_char
    # init a sequence
    is_valid = False

    while not is_valid:
        seq = np.zeros(length)
        for t in range(length):
            # If the last symbol is a recall we wait for a store
            if last_signal == recall_char and rd.rand() < prob:
                seq[t] = store_char
                last_signal = store_char

            # Otherwise we wait for a recall
            elif last_signal == store_char and rd.rand() < prob:
                seq[t] = recall_char
                last_signal = recall_char

        is_valid = validity_test(seq, recall_char, store_char)

    binary_seq = [(seq == store_char) * 1, (seq == recall_char) * 1]  # * 1 for conversion from boolean to int
    return np.array(binary_seq)  # (store/recall, length)


def random_binary_word(width, max_prob_active=None):
    """Generate random binary word of specific width"""
    word = np.random.randint(2, size=width)
    # don't use blank words (all zeros)
    while sum(word) == 0:
        word = np.random.randint(2, size=width)

    if max_prob_active is None:
        return word
    else:
        while sum(word) > int(width * max_prob_active):
            word = np.random.randint(2, size=width)
        return word


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    # return sum(c1 != c2 for c1, c2 in zip(s1, s2))  # Wikipedia solution
    return np.count_nonzero(s1 != s2)  # a faster solution


def generate_value_dicts(n_values, train_dict_size, test_dict_size, min_hamming_dist=5, max_prob_active=None,
                         hamm_among_each_word=True):
    """
    Generate dictionaries of binary words for training and testing.
    Ensures minimal hamming distance between test words and any training words.
    Ensures sparsity in active bit by limiting the percentage of active bits in a word by max_prob_active.
    """
    from tqdm import tqdm
    common_dict = []
    pbar = tqdm(total=train_dict_size + test_dict_size)
    while len(common_dict) < train_dict_size + test_dict_size:
        test_candidate = random_binary_word(n_values, max_prob_active)
        valid = True
        for word in common_dict:
            if (word == test_candidate).all() or \
                    (hamm_among_each_word and hamming2(word, test_candidate) <= min_hamming_dist):
                valid = False
                break
        if valid:
            common_dict.append(test_candidate)
            pbar.update(1)
    pbar.close()
    return np.array(common_dict[:train_dict_size]), np.array(common_dict[train_dict_size:])

    # # Generate dictionary of unique binary words for training set
    # dict_train = []
    # while len(dict_train) < train_dict_size:
    #     train_candidate = random_binary_word(n_values, max_prob_active)
    #     if not any([(train_candidate == w).all() for w in dict_train]):
    #         dict_train.append(train_candidate)
    # assert len(dict_train) == train_dict_size
    #
    # # Generate dictionary of unique binary words for test set with minimal hamming distance to words in train set
    # dict_test = []
    # valid = True
    # if min_hamming_dist is not None:
    #     while len(dict_test) < test_dict_size:
    #         test_candidate = random_binary_word(n_values, max_prob_active)
    #         for train_word in dict_train:
    #             if hamming2(train_word, test_candidate) <= min_hamming_dist:
    #                 valid = False
    #                 break
    #         if valid:
    #             dict_test.append(test_candidate)
    #         else:
    #             valid = True
    #     assert len(dict_test) == test_dict_size
    #     return np.array(dict_train), np.array(dict_test)
    # else:
    #     return np.array(dict_train), np.array(dict_train)


def randn_except(limit, not_num):
    candidate = np.random.randint(limit)
    return candidate if candidate != not_num else randn_except(limit, not_num)


def remove_consecutive_same_numbers(sequence, max_num):
    for i in range(len(sequence)-1):
        if sequence[i] == sequence[i+1]:
            sequence[i+1] = randn_except(max_num, sequence[i])
    return sequence


def onehot(idx_array):
    onehot_array = np.zeros((idx_array.size, idx_array.max() + 1))  # appropriately sized zero array
    onehot_array[np.arange(idx_array.size), idx_array] = 1  # index columns and set
    return onehot_array


def generate_onehot_storerecall_batch(batch_size, length, prob_storerecall, n_values, distractors,
                                      no_distractors_during_recall=True):
    """
    Given the number of one-hot input channels for generate a batch of store-recall sequences
    with specified probability of store/recall
    :param batch_size: size of mini-batch
    :param length: length of sequences
    :param prob_storerecall: probability of store/recall signal
    :param value_dict: dictionary of binary words to use
    :param no_distractors_during_recall: should a distractor be shown during recall
    :return: mini-batch of store-recall sequences (batch_size, channels, length)
    """
    input_batch = []
    target_batch = []
    output_mask_batch = []
    words_idxs = [i for i in range(n_values)]
    value_dict = onehot(np.array(words_idxs))
    word_count = 0
    # words_batch_stats = {i: 0 for i in range(value_dict.shape[0])}
    store_signal_to_batch_map = []
    for b in range(batch_size):
        # generate valid store/recall signals by probability
        storerecall_sequence = generate_storerecall_signals_with_prob(length, prob_storerecall)
        word_sequence_choice = np.random.choice(n_values, length)

        # optionally we make sure there are no same consecutive words
        word_sequence_choice = remove_consecutive_same_numbers(word_sequence_choice, n_values)

        # ensure that the stored words are balanced in the batch (similar number of each word stored)
        store_sequence = storerecall_sequence[0]
        store_idxs = np.nonzero(store_sequence)[0]  # store step idxs (only one dimension so taking [0]
        for si in store_idxs:
            word_sequence_choice[si] = words_idxs[word_count % len(words_idxs)]  # different word per sequence
            word_count += 1
            store_signal_to_batch_map.append(b)

        if distractors:
            values_sequence = np.array(value_dict[word_sequence_choice]).swapaxes(0, 1)
        else:
            values_sequence = np.array(value_dict[word_sequence_choice])
            values_sequence_z = np.zeros_like(values_sequence)
            for si in store_idxs:
                values_sequence_z[si] = value_dict[word_sequence_choice[si]]
            values_sequence = values_sequence_z.swapaxes(0, 1)

        # if b == 0:
        #     print(word_sequence_choice)
        #     print("actual words in sequence")
        #     print(values_sequence)
        repeated_storerecall_sequence = np.repeat(storerecall_sequence, 2, axis=0)
        inv_values_sequence = 1 - values_sequence
        input_sequence = np.vstack((repeated_storerecall_sequence, values_sequence, inv_values_sequence))
        # input_sequence.shape = (channels, length)
        target_sequence = np.zeros_like(values_sequence)
        for step in range(length):
            store_seq = storerecall_sequence[0]
            recall_seq = storerecall_sequence[1]
            if store_seq[step] == 1:
                next_target = values_sequence[:, step]
            if recall_seq[step] == 1:
                target_sequence[:, step] = next_target
                if no_distractors_during_recall:
                    input_sequence[n_values:, step] = 0

        input_batch.append(input_sequence)
        target_batch.append(target_sequence)
        output_mask_batch.append(storerecall_sequence[1])
    # print("batch stats (instances of words stored)", words_batch_stats)
    return np.array(input_batch), np.array(target_batch), np.array(output_mask_batch), store_signal_to_batch_map


def generate_symbolic_storerecall_batch(batch_size, length, prob_storerecall, value_dict, distractors,
                                        no_distractors_during_recall=True):
    """
    Given the value dictionary generate a batch of store-recall sequences with specified probability of store/recall
    :param batch_size: size of mini-batch
    :param length: length of sequences
    :param prob_storerecall: probability of store/recall signal
    :param value_dict: dictionary of binary words to use
    :param no_distractors_during_recall: should a distractor be shown during recall
    :return: mini-batch of store-recall sequences (batch_size, channels, length)
    """
    n_values = value_dict[0].shape[0]  # number of bits in a value (width of value word)
    input_batch = []
    target_batch = []
    output_mask_batch = []
    words_idxs = [i for i in range(value_dict.shape[0])]
    word_count = 0
    # words_batch_stats = {i: 0 for i in range(value_dict.shape[0])}
    store_signal_to_batch_map = []
    for b in range(batch_size):
        # generate valid store/recall signals by probability
        storerecall_sequence = generate_storerecall_signals_with_prob(length, prob_storerecall)
        word_sequence_choice = np.random.choice(value_dict.shape[0], length)

        # optionally we make sure there are no same consecutive words
        word_sequence_choice = remove_consecutive_same_numbers(word_sequence_choice, value_dict.shape[0])

        # ensure that the stored words are balanced in the batch (similar number of each word stored)
        store_sequence = storerecall_sequence[0]
        store_idxs = np.nonzero(store_sequence)[0]  # store step idxs (only one dimension so taking [0]
        for si in store_idxs:
            word_sequence_choice[si] = words_idxs[word_count % len(words_idxs)]  # different word per sequence
            word_count += 1
            store_signal_to_batch_map.append(b)

        if distractors:
            values_sequence = np.array(value_dict[word_sequence_choice]).swapaxes(0, 1)
        else:
            values_sequence = np.array(value_dict[word_sequence_choice])
            values_sequence_z = np.zeros_like(values_sequence)
            for si in store_idxs:
                values_sequence_z[si] = value_dict[word_sequence_choice[si]]
            values_sequence = values_sequence_z.swapaxes(0, 1)

        # if b == 0:
        #     print(word_sequence_choice)
        #     print("actual words in sequence")
        #     print(values_sequence)
        repeated_storerecall_sequence = np.repeat(storerecall_sequence, 2, axis=0)
        inv_values_sequence = 1 - values_sequence
        input_sequence = np.vstack((repeated_storerecall_sequence, values_sequence, inv_values_sequence))
        # input_sequence.shape = channels, length
        target_sequence = np.zeros_like(values_sequence)
        for step in range(length):
            store_seq = storerecall_sequence[0]
            recall_seq = storerecall_sequence[1]
            if store_seq[step] == 1:
                next_target = values_sequence[:, step]
            if recall_seq[step] == 1:
                target_sequence[:, step] = next_target
                if no_distractors_during_recall:
                    input_sequence[n_values:, step] = 0

        input_batch.append(input_sequence)
        target_batch.append(target_sequence)
        output_mask_batch.append(storerecall_sequence[1])
    # print("batch stats (instances of words stored)", words_batch_stats)
    return np.array(input_batch), np.array(target_batch), np.array(output_mask_batch), store_signal_to_batch_map


def generate_spiking_storerecall_batch(batch_size, length, prob_storerecall, value_dict, n_charac_duration,
                                       n_neuron, f0, test_dict, max_prob_active, min_hamming_dist, distractors,
                                       n_values, n_per_channel, onehot=False, no_distractors_during_recall=True):
    assert n_neuron / (n_values * 2 + 2 * 2) == n_per_channel,\
        "Number of input neurons {} not divisible by number of input channels {}".format(n_neuron, n_values)
    if onehot:
        input_batch, target_batch, output_mask_batch, store_signal_to_batch_map = generate_onehot_storerecall_batch(
            batch_size, length, prob_storerecall, n_values, distractors)
    else:
        n_values = test_dict[0].shape[0]  # number of bits in a value (width of value word)
        n_random_words = 2 * n_values if n_values >= 10 else n_values
        if value_dict is None:
            common_dict = []
            # pbar = tqdm(total=n_random_words)
            while len(common_dict) < n_random_words:  # n_random_words random train words
                test_candidate = random_binary_word(n_values, max_prob_active)
                valid = True
                for word in test_dict:  # make sure sufficiently different from test words
                    if hamming2(word, test_candidate) <= min_hamming_dist:
                        valid = False
                        break
                if valid:
                    for word in common_dict:  # make sure sufficiently different from train words
                        if hamming2(word, test_candidate) <= min_hamming_dist:
                            valid = False
                            break
                if valid:
                    common_dict.append(test_candidate)
                    # pbar.update(1)
            # pbar.close()
            value_dict = np.array(common_dict)
        input_batch, target_batch, output_mask_batch, store_signal_to_batch_map = generate_symbolic_storerecall_batch(
            batch_size, length, prob_storerecall, value_dict, distractors, no_distractors_during_recall)

    input_batch = np.repeat(input_batch, n_per_channel, axis=1)
    input_batch = np.repeat(input_batch, n_charac_duration, axis=2)

    input_rates_batch = input_batch * f0  # convert to firing rates (with firing rate being f0)
    input_spikes_batch = generate_poisson_noise_np(input_rates_batch)

    # convert data to be of shape (batch, time[, channels])
    input_batch = input_batch.swapaxes(1, 2)
    input_spikes_batch = input_spikes_batch.swapaxes(1, 2)
    target_batch = target_batch.swapaxes(1, 2)
    return input_spikes_batch, input_batch, target_batch, output_mask_batch, store_signal_to_batch_map


# def debug_plot_spiking_input_generation():
#     import matplotlib.pyplot as plt
#     n_values = 12
#     train_value_dict, test_value_dict = generate_value_dicts(n_values=n_values, train_dict_size=5,
#                                                              test_dict_size=5,
#                                                              max_prob_active=0.5)
#     n_neuron = 112
#     input_spikes_batch, input_batch, target_batch, output_mask_batch = \
#         generate_spiking_storerecall_batch(
#             batch_size=16, length=10, prob_storerecall=0.2, value_dict=train_value_dict,
#             n_charac_duration=200, n_neuron=n_neuron, f0=500. / 1000.)
#     batch=0
#     print(input_batch[batch].swapaxes(0, 1))
#
#     n_neuron_per_channel = n_neuron // (n_values + 2)  # 8
#     sr_spikes = input_spikes_batch[batch, :, :2 * n_neuron_per_channel]
#     fig, ax = plt.subplots(figsize=(8, 4), gridspec_kw={'wspace': 0, 'hspace': 0.2})
#     raster_plot(ax, sr_spikes)
#     plt.draw()
#     plt.pause(1)


def storerecall_error(output, target, onehot=False):
    """
    Calculate the error of batch. (batch, time (1 or 2), output bits)
    :param output: readout of network at relevant (recall) timesteps
    :param target: if onehot: list of integers (labels) else: list of binary strings at relevant timesteps
    :param onehot: flag if the input is one hot encoded
    :return:
    """
    if onehot:
        output = tf.argmax(output, axis=1)
    else:
        output = tf.where(output < 0.5, tf.zeros_like(output), tf.ones_like(output))
    # target = tf.cast(target, dtype=tf.float32)
    # output = tf.cast(output, dtype=tf.float32)
    # output = tf.Print(output, [output[0], target[0]], message="output, target", summarize=999)
    bit_accuracy = tf.equal(output, target)
    per_word_acc = bit_accuracy if onehot else tf.reduce_all(bit_accuracy, axis=1)
    failed_store_idxs = tf.where(tf.logical_not(per_word_acc))
    per_word_acc = tf.cast(per_word_acc, dtype=tf.float32)
    per_word_acc = tf.reduce_mean(per_word_acc, axis=0)
    per_word_error = 1. - per_word_acc
    per_bit_accuracy = tf.reduce_mean(tf.cast(bit_accuracy, tf.float32), axis=0)
    per_bit_error = 1. - per_bit_accuracy
    mean_bit_accuracy = tf.reduce_mean(per_bit_accuracy)
    mean_bit_error = 1. - mean_bit_accuracy

    return mean_bit_accuracy, mean_bit_error, per_bit_accuracy, per_bit_error,\
           per_word_error, per_word_acc, failed_store_idxs


def generate_storerecall_data(batch_size, sentence_length, n_character, n_charac_duration, n_neuron, f0=200 / 1000,
                                 with_prob=True, prob_signals=1 / 5, override_input=None, delay=None):
    channels, target_nums, memory_nums, input_nums = generate_mikolov_data(
        batch_size, sentence_length, n_character, with_prob=with_prob, prob_bit_to_recall=prob_signals,
        prob_bit_to_store=prob_signals, override_input=override_input, delay=delay)

    total_character = n_character + 2  # number of input gates
    recall_character = total_character - 1
    store_character = recall_character - 1

    neuron_split = np.array_split(np.arange(n_neuron), total_character)
    lstm_in_rates = np.zeros((batch_size, sentence_length*n_charac_duration, n_neuron))
    in_gates = channels
    for b in range(batch_size):
        for c in range(sentence_length):
            for in_gate_i, n_group in enumerate(neuron_split):
                lstm_in_rates[b, c*n_charac_duration:(c+1)*n_charac_duration, n_group] = in_gates[in_gate_i][b][c] * f0

    spikes = generate_poisson_noise_np(lstm_in_rates)
    target_sequence = np.repeat(target_nums, repeats=n_charac_duration, axis=1)
    # Generate the recall mask
    is_recall_table = np.zeros((total_character, n_charac_duration), dtype=bool)
    is_recall_table[recall_character, :] = True
    is_recall = np.concatenate([is_recall_table[input_nums][:, k] for k in range(sentence_length)], axis=1)

    return spikes, is_recall, target_sequence, None, input_nums, target_nums


def update_stp_plot(plt, ax_list, FLAGS, plot_result_values, batch=0, n_max_neuron_per_raster=100):
    raise ValueError("Not implemented")


def update_plot(plt, ax_list, FLAGS, plot_result_values, batch=None, n_max_neuron_per_raster=100):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    failed_batches = plot_result_values['failed_store_idxs']
    if batch is None and len(failed_batches) > 0:
        store_signal_to_batch_map = plot_result_values['store_signal_to_batch_map_holder']
        batch = store_signal_to_batch_map[failed_batches[0][0]]
        ax_list[0].set_title("Failed batch " + str(batch))

    subsample_input = FLAGS.n_per_channel
    subsample_rnn = FLAGS.n_per_channel
    ylabel_x = -0.11
    ylabel_y = 0.5
    fs = 10
    plt.rcParams.update({'font.size': fs})
    batch = np.random.randint(FLAGS.batch_train) if batch is None else batch

    top_margin = 0.08
    left_margin = -0.085

    # PLOT STORE-RECALL SIGNAL SPIKES
    ax = ax_list[0]
    n_neuron_per_channel = FLAGS.n_in // (FLAGS.n_charac * 2 + 2 * 2)
    sr_num_channels = 2 * 2
    sr_num_neurons = sr_num_channels*n_neuron_per_channel
    sr_spikes = plot_result_values['input_spikes'][batch, :, :sr_num_neurons]
    # raster_plot(ax, sr_spikes[:, ::subsample_input], linewidth=0.15)
    sr_channel_neurons = sr_num_neurons // 2
    sr_channels = np.mean(sr_spikes.reshape(sr_spikes.shape[0], -1, sr_channel_neurons), axis=2)
    cax = ax.imshow(sr_channels.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')

    ax.set_yticklabels([])
    ax.text(left_margin, 0.8 - top_margin, 'recall', transform=ax.transAxes, fontsize=7, verticalalignment='top')
    ax.text(left_margin, 0.4 - top_margin, 'store', transform=ax.transAxes, fontsize=7, verticalalignment='top')
    ax.set_xticks([])

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    z = plot_result_values['z']
    raster_data = \
        zip(range(3), [plot_result_values['input_spikes'], z, z], ['input', 'LIF', 'ALIF']) if FLAGS.n_regular > 0 else\
        zip(range(2), [plot_result_values['input_spikes'], z], ['input', 'ALIF'])

    for k_data, data, d_name in raster_data:
        ax = ax_list[k_data+1]
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        hide_bottom_axis(ax)

        if np.size(data) > 0:
            data = data[batch]
            if d_name is 'LIF':
                data = data[:, :FLAGS.n_regular:subsample_rnn]
            elif d_name is 'ALIF':
                data = data[:, FLAGS.n_regular::subsample_rnn]
            elif d_name is 'input':
                data = data[:, sr_num_neurons:]
                max_y_tick_label = str(data.shape[-1])
                data = np.mean(data.reshape(data.shape[0], -1, n_neuron_per_channel), axis=2)

                cax = ax.imshow(data.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
                ax.set_ylabel(d_name, fontsize=fs)
                ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
                # ax.set_yticklabels(['1', max_y_tick_label])
                continue

            if FLAGS.model != 'lstm':
                n_max = min(data.shape[1], n_max_neuron_per_raster)
                cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
                data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
                raster_plot(ax, data, linewidth=0.15)
            else:
                ax.imshow(data.T, origin='lower', aspect='auto', interpolation='none')
            ax.set_ylabel(d_name, fontsize=fs)
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
            ax.set_yticklabels(['1', str(data.shape[-1])])

    if FLAGS.model != 'lstm':
        ax = ax_list[-2]
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        ax.set_ylabel('thresholds of A', fontsize=fs)
        ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        sub_data = plot_result_values['b_con'][batch]
        vars = np.var(sub_data, axis=0)
        cell_with_max_var = np.argsort(vars)[::-1]
        presentation_steps = np.arange(sub_data.shape[0])
        ax.plot(sub_data[:, cell_with_max_var[::subsample_rnn]], color='r', alpha=0.4, linewidth=1)
        ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
                 np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]
        hide_bottom_axis(ax)

    # plot targets
    ax = ax_list[-1]
    mask = plot_result_values['recall_charac_mask'][batch]
    # data = plot_result_values['target_nums'][batch]
    # data[np.invert(mask)] = -1
    # lines = []
    # ind_nt = np.argwhere(data != -1)
    # for idx in ind_nt.tolist():
    #     i = idx[0]
    #     lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    # lc_t = mc.LineCollection(lines, colors='green', linewidths=2, label='target')
    # ax.add_collection(lc_t)  # plot target segments

    # plot output per tau_char
    # data = plot_result_values['out_plot_char_step'][batch]
    # data = np.array([(d[1] - d[0] + 1) / 2 for d in data])
    # data[np.invert(mask)] = -1
    # lines = []
    # ind_nt = np.argwhere(data != -1)
    # for idx in ind_nt.tolist():
    #     i = idx[0]
    #     lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    # lc_o = mc.LineCollection(lines, colors='blue', linewidths=2, label='avg. output')
    # ax.add_collection(lc_o)  # plot target segments

    # plot softmax of psp-s per dt for more intuitive monitoring
    # ploting only for second class since this is more intuitive to follow (first class is just a mirror)
    output2 = plot_result_values['out_plot'][batch, :, :]
    presentation_steps = np.arange(output2.shape[0])
    ax.set_yticks([0, 0.5, 1])
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('output', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    ax.plot(output2, label='output', alpha=0.7)
    ax.axis([0, presentation_steps[-1] + 1, -0.1, 1.1])
    # ax.legend(handles=[line_output2], loc='lower center', fontsize=7,
    #           bbox_to_anchor=(0.5, -0.1), ncol=3)

    ax.set_xlabel('time in ms', fontsize=fs)
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(1)


def offline_plot(data_path, custom_plot=True):
    import matplotlib.pyplot as plt
    import datetime
    import pickle
    import json
    import os

    flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    from types import SimpleNamespace
    flags = SimpleNamespace(**flags_dict)

    plot_data = 'plot_custom_trajectory_data.pickle' if custom_plot else 'plot_trajectory_data.pickle'
    plot_result_values = pickle.load(open(os.path.join(data_path, plot_data), 'rb'))

    plt.ion()
    nrows = 5 if flags.n_regular > 0 else 4
    height = 7.5 if flags.n_regular > 0 else 6
    fig, ax_list = plt.subplots(nrows=nrows, figsize=(6, height), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    for b in range(flags.batch_test):
        update_plot(plt, ax_list, flags, plot_result_values, batch=b, n_max_neuron_per_raster=100)
        start_time = datetime.datetime.now()
        fig.savefig(os.path.join(data_path, 'figure_test' + str(b) + '_' + start_time.strftime("%H%M") + '.pdf'),
                    format='pdf')


def avg_firingrates_during_delay(data_path):
    """
    Calculate average firing rates during delays of custom plot (for two value store-recall task).
    Data is conditioned on the current memory content
    [0 (after storing 0), 1 (after storing 1), blank (after recall)]
    Motivation: check if firing rate is higher during delay if the memory is filled.
    :param data_path:
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime
    import pickle
    import json
    import os

    flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    from types import SimpleNamespace
    flags = SimpleNamespace(**flags_dict)

    plot_result_values = pickle.load(open(os.path.join(data_path, 'plot_custom_trajectory_data.pickle'), 'rb'))
    firing_rates = {'0': [], '1': [], 'blank': []}
    for b in range(flags.batch_test):  # plot_result_values['input_nums'].shape[0]:
        symbolic_input = plot_result_values['input_nums'][b]
        z = plot_result_values['z'][b]
        step = flags.tau_char
        # index 1 and 12 determine the content of memory during the first and third delay periods
        # recalls at 7 and 19
        # relevant periods 1-7 first memory, 8-12 blank, 13-19 second memory
        firing_rates[str(symbolic_input[1])].append(np.mean(z[1*step:7*step]))
        firing_rates['blank'].append(np.mean(z[8*step:12*step]))
        firing_rates[str(symbolic_input[12])].append(np.mean(z[13*step:19*step]))
    for k in firing_rates.keys():
        firing_rates[k] = np.mean(firing_rates[k]) * 1000
        print("AVG Firing rate for memory content ({}) = {:.2g}".format(k, firing_rates[k]))
    return firing_rates


# Covert data units to points
def height_from_data_units(height, axis, reference='y', value_range=None):
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        if value_range is None:
            value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        if value_range is None:
            value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale height to value range
    ms = height * (length / value_range)
    return ms


# This function is called for every subplot that you want to plot.
# Spikes are of the shape: (n_neurons, time), ax is an axes object.
def plot_spikes(ax, spikes, linewidth=None, max_spike=None, color='black'):
    import matplotlib.ticker as ticker
    n_neurons = spikes.shape[0]
    neurons = np.arange(n_neurons) + 1
    sps = spikes * (neurons.reshape(len(spikes), 1))
    sps[sps == 0.] = -10
    sps -= 1
    marker_size = height_from_data_units(0.7, ax, value_range=n_neurons)
    for neuron in range(n_neurons):
        ax.plot(range(spikes.shape[1]), sps[neuron, :], marker='|', linestyle='none', color=color,
                markersize=marker_size, markeredgewidth=0.5)
    ax.set(ylim=(-0.5, n_neurons))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, n_neurons]))


def pretty_560_plot(data_path, custom_plot=True, spikesonly=False, restonly=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import datetime
    import pickle
    import json
    from tqdm import tqdm
    import os

    pbar = tqdm(total=10)
    flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    from types import SimpleNamespace
    FLAGS = SimpleNamespace(**flags_dict)

    plot_data = 'plot_custom_trajectory_data.pickle' if custom_plot else 'plot_trajectory_data.pickle'
    data = pickle.load(open(os.path.join(data_path, plot_data), 'rb'))
    pbar.update(1)

    raw_input = data['input_spikes']  # also for analog input the key is 'input_spikes'
    # print(FLAGS)
    # print(raw_input.shape)  # batch, time, channels (128, 1000, 60)
    shp = raw_input.shape
    ch_in = np.mean(np.reshape(raw_input, (shp[0], -1, FLAGS.tau_char, shp[2])), axis=2)  # avg per char step
    pbar.update(1)
    shp = ch_in.shape
    ch_in = np.mean(np.reshape(ch_in, (shp[0], shp[1], -1, FLAGS.n_per_channel)), axis=3)  # avg per channel
    ch_in = ch_in > 0.0  # convert to binary
    pbar.update(1)
    # print(ch_in.shape)
    n_group = FLAGS.n_charac  # size of a group in input channels. groups: store-recall, input, inv-input
    assert ch_in.shape[2] == 2 * n_group + 2 * 2, \
        "ch_in.shape[2]" + str(ch_in.shape[2]) + " does not contain 2 groups of " + str(n_group) + " + 2"

    store = np.mean(ch_in[:, :, 0:2], axis=2)[..., np.newaxis]  # first half of first group
    recall = np.mean(ch_in[:, :, 2:4], axis=2)[..., np.newaxis]  # second half of first group
    norm_input = ch_in[:, :, 4:4 + n_group]
    pbar.update(1)

    store_idxs = np.nonzero(store)  # list of batch idxs, list of time idxs
    recall_idxs = np.nonzero(recall)  # list of batch idxs, list of time idxs
    delays = [r - s for s, r in zip(store_idxs[1], recall_idxs[1])]
    # long_delay_batch = np.argmax(delays)
    long_delay_batch = np.argpartition(delays, 4)[-1]
    # print("argmax", long_delay_batch)
    # print("np.argpartition(delays, -4)[-4:]", np.argpartition(delays, -4)[-4:])
    # print("np.argsort(delays)[-4:]", np.argsort(delays)[-4:])

    start_time = datetime.datetime.now()
    filename = os.path.join(data_path, 'NEW_figure_test' + str(long_delay_batch) + '_' + start_time.strftime("%H%M"))
    pbar.update(1)

    plt.ion()
    # fig, ax_list = plt.subplots(nrows=5, figsize=(6, 7.3), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    fig = plt.figure(figsize=(7.3, 8), tight_layout=True)
    gs = gridspec.GridSpec(13, 10, wspace=0.45, hspace=0.2)
    plt.subplots_adjust(left=0.13, right=0.96, top=0.99, bottom=0.06)
    pbar.update(1)

    plot_result_values = data
    batch = long_delay_batch

    ylabel_x = -0.11
    ylabel_y = 0.5
    fs = 10
    plt.rcParams.update({'font.size': fs})

    top_margin = 0.08
    left_margin = -0.085
    spikewidth = 0.15

    # PLOT INPUT PATTERNS
    for pi in range(norm_input.shape[1]):  # this should be 10
        ax = fig.add_subplot(gs[:2, pi])
        pattern_flat = norm_input[batch, pi]
        pattern = np.reshape(pattern_flat, (5, 4))
        ax.imshow(pattern, cmap='binary')
        if pi in np.nonzero(store[batch])[0]:
            for spine in ax.spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(4)
        ax.set_xticks([])
        ax.set_yticks([])

    # PLOT STORE-RECALL SIGNAL SPIKES
    max_spike = 20000
    # ax = fig.add_subplot(gs[1, :])
    # ax.clear()
    # strip_right_top_axis(ax)
    n_neuron_per_channel = FLAGS.n_in // (FLAGS.n_charac * 2 + 2 * 2)
    assert n_neuron_per_channel == FLAGS.n_per_channel
    sr_num_channels = 2 * 2
    sr_num_neurons = sr_num_channels * n_neuron_per_channel
    sr_spikes = plot_result_values['input_spikes'][batch, :, :sr_num_neurons]
    #sr_spikes = sr_spikes[:, ::FLAGS.n_per_channel]  # subsample to one neuron per channel
    # plot_spikes(ax, sr_spikes, linewidth=spikewidth, max_spike=max_spike)
    # # sr_channel_neurons = sr_num_neurons // 2
    # # sr_channels = np.mean(sr_spikes.reshape(sr_spikes.shape[0], -1, sr_channel_neurons), axis=2)
    # # cax = ax.imshow(sr_channels.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
    # ax.set_yticklabels([])
    # ax.text(left_margin, 0.8 - top_margin, 'recall', transform=ax.transAxes, fontsize=7, verticalalignment='top')
    # ax.text(left_margin, 0.4 - top_margin, 'store', transform=ax.transAxes, fontsize=7, verticalalignment='top')
    # ax.set_xticks([])

    # PLOT INPUT SPIKES
    k_data = 0
    data = plot_result_values['input_spikes']
    d_name = 'input'
    ax = fig.add_subplot(gs[2:6, :])
    ax.clear()
    strip_right_top_axis(ax)
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    hide_bottom_axis(ax)
    ax.set_yticklabels([])
    if spikesonly:
        ax.spines['left'].set_visible(False)
        ax.set_xticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    data = data[batch]
    # data = data[:, sr_num_neurons::FLAGS.n_per_channel]  # subsample to one neuron per channel
    data = data[:, sr_num_neurons:sr_num_neurons + FLAGS.n_charac*FLAGS.n_per_channel]
    # max_y_tick_label = str(data.shape[-1])
    # data = np.mean(data.reshape(data.shape[0], -1, n_neuron_per_channel), axis=2)
    # cax = ax.imshow(data.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
    input_spikes = np.hstack((sr_spikes, np.zeros_like(sr_spikes), data))
    if not restonly:
        plot_spikes(ax, input_spikes[:, ::FLAGS.n_per_channel].T, linewidth=spikewidth, max_spike=max_spike)
    presentation_steps = np.arange(input_spikes.shape[0])
    ax.set_xticks([0, presentation_steps[-1] + 1])
    ax.set_ylabel(d_name, fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    # ax.set_yticklabels(['1', max_y_tick_label])
    if spikesonly:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename + '_SPIKES_INPUT.png', bbox_inches=extent, dpi=1000)

    pbar.update(1)

    # PLOT ALIF SPIKES
    sub_data = plot_result_values['b_con'][batch]
    vars = np.var(sub_data, axis=0)
    cell_with_max_var = np.argsort(vars)[::-1]

    k_data = 1
    data = plot_result_values['z']
    d_name = 'ALIF'
    ax = fig.add_subplot(gs[6:8, :])
    ax.clear()
    strip_right_top_axis(ax)
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    hide_bottom_axis(ax)
    ax.set_yticklabels([])
    if spikesonly:
        ax.spines['left'].set_visible(False)
        ax.set_xticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        ax.get_yaxis().set_visible(False)

    data = data[batch]
    data = data[:, cell_with_max_var[::20]]

    # cell_select = np.linspace(start=0, stop=data.shape[1] - 1, dtype=int)
    # data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
    if not restonly:
        plot_spikes(ax, data.T, linewidth=spikewidth, max_spike=max_spike)

    ax.set_ylabel(d_name, fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    # ax.set_yticklabels(['1', str(data.shape[-1])])
    if spikesonly:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename + '_SPIKES_ALIF.png', bbox_inches=extent, dpi=1000)
    pbar.update(1)

    if not spikesonly:
        ax = fig.add_subplot(gs[8:10, :])
        ax.clear()
        strip_right_top_axis(ax)
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        ax.set_ylabel('thresholds', fontsize=fs)
        ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        presentation_steps = np.arange(sub_data.shape[0])
        # ax.plot(sub_data[:, cell_with_max_var[::10]], color='r', alpha=0.4, linewidth=1)
        ax.plot(sub_data[:, cell_with_max_var[::20]], alpha=0.7, linewidth=1)
        # ax.plot(sub_data[:, cell_with_max_var], alpha=0.7, linewidth=1)
        ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
                 np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]
        hide_bottom_axis(ax)
        pbar.update(1)

        # plot targets
        ax = fig.add_subplot(gs[10, :])
        ax.clear()
        strip_right_top_axis(ax)
        # plot softmax of psp-s per dt for more intuitive monitoring
        # ploting only for second class since this is more intuitive to follow (first class is just a mirror)
        output2 = plot_result_values['out_plot'][batch, :, :]
        presentation_steps = np.arange(output2.shape[0])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '', '1'])
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        ax.set_ylabel('sigmoid\noutputs', fontsize=fs)
        ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        ax.plot(output2, label='output', alpha=0.7, linewidth=1)
        ax.axis([0, presentation_steps[-1] + 1, -0.1, 1.1])
        # ax.legend(handles=[line_output2], loc='lower center', fontsize=7,
        #           bbox_to_anchor=(0.5, -0.1), ncol=3)
        ax.set_xticks([0, presentation_steps[-1] + 1])
        ax.set_xticklabels(['0', str(int((presentation_steps[-1] + 1) / 1000))])
        ax.set_xlabel('time in s', fontsize=fs)

        # PLOT OUTPUT PATTERNS
        shp = output2.shape
        ch_out = np.mean(np.reshape(output2, (-1, FLAGS.tau_char, shp[1])), axis=1)  # avg per char step
        for pi in range(norm_input.shape[1]):  # this should be 10
            ax = fig.add_subplot(gs[11:, pi])
            pattern_flat = ch_out[pi]
            pattern = np.reshape(pattern_flat, (5, 4))
            ax.imshow(pattern, cmap='binary')
            if pi in np.nonzero(recall[batch])[0]:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(4)
            ax.set_xticks([])
            ax.set_yticks([])

    pbar.update(1)
    # To plot with interactive python one need to wait one second to the time to draw the axis
    plt.draw()
    plt.pause(2)

    if spikesonly:
        filename += '_SPIKES'
        fig.savefig(filename + '.png', format='png')
    else:
        fig.savefig(filename + '.pdf', format='pdf')
        fig.savefig(filename + '.png', format='png', dpi=1000)
    pbar.update(1)
    pbar.close()
    print("Longest delay", delays[batch], "in batch", batch)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    import pickle
    import json
    import os

    parser = argparse.ArgumentParser(description='Plot extended SR activity for a sequence.')
    parser.add_argument('path', help='Path to directory that contains flags and plot data.')
    parser.add_argument('--customplot', default=False, help='Use custom plot file.')
    # parser.add_argument('--spikesonly', default=False, help='Plot only spikes.')
    args = parser.parse_args()

    print("Attempting to load model from " + args.path)

    # pretty_560_plot(args.path, custom_plot=args.customplot, spikesonly=args.spikesonly)
    # pretty_560_plot(args.path, custom_plot=args.customplot)
    pretty_560_plot(args.path, custom_plot=args.customplot, restonly=True)
    pretty_560_plot(args.path, custom_plot=args.customplot, spikesonly=True)
