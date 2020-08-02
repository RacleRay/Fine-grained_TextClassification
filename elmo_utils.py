#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   elmo_utils.py
'''

import codecs
import os
import json

import numpy as np
import tensorflow as tf


def read_vocab(vocab_file):
    """read vocab from file

    Args:
        vocab_file ([type]): path to the vocab file, the vocab file should contains a word each line

    Returns:
        list of words
    """

    if not os.path.isfile(vocab_file):
        raise ValueError("%s is not a vaild file" % vocab_file)

    vocab = []
    word2id = {}
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        for i, line in enumerate(f):
            word = line.strip()
            if not word:
                raise ValueError("Got empty word at line %d" % (i + 1))
            vocab.append(word)
            word2id[word] = len(word2id)

    print("# vocab size: ", len(vocab))
    return vocab, word2id


def load_embed_file(embed_file):
    """Load embed_file into a python dictionary.

    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:

    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547

    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for i, line in enumerate(f):
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), \
                    "All embedding size should be same, but got {0} at line {1}".format(len(vec), i + 1)
            else:
                emb_size = len(vec)
    return emb_dict, emb_size


def _load_pretrained_emb_from_file(name,
                                   vocab_file,
                                   embed_file,
                                   num_trainable_tokens=0,
                                   dtype=tf.float32):
    print("# Start to load pretrained embedding...")
    vocab, _ = read_vocab(vocab_file)
    if num_trainable_tokens:
        trainable_tokens = vocab[:num_trainable_tokens]
    else:
        trainable_tokens = vocab

    emb_dict, emb_size = load_embed_file(embed_file)
    print("# pretrained embedding size", len(emb_dict), emb_size)

    for token in trainable_tokens:
        if token not in emb_dict:
            if '<average>' in emb_dict:
                emb_dict[token] = emb_dict['<average>']
            else:
                emb_dict[token] = list(np.random.random(emb_size))

    emb_mat = np.array([emb_dict[token] for token in vocab],
                       dtype=dtype.as_numpy_dtype())
    if num_trainable_tokens:
        emb_mat = tf.constant(emb_mat)
        emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
        with tf.device(_get_embed_device(num_trainable_tokens)):
            emb_mat_var = tf.get_variable(name + "_emb_mat_var",
                                          [num_trainable_tokens, emb_size])
        return tf.concat([emb_mat_var, emb_mat_const], 0, name=name)
    else:
        with tf.device(_get_embed_device(len(vocab))):
            emb_mat_var = tf.get_variable(
                name,
                emb_mat.shape,
                initializer=tf.constant_initializer(emb_mat))
        return emb_mat_var


# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 30000


def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size."""
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    else:
        return "/gpu:0"


def create_embedding(name,
                     vocab_size,
                     embed_size,
                     vocab_file=None,
                     embed_file=None,
                     num_trainable_tokens=0,
                     dtype=tf.float32,
                     scope=None,
                     mode='train'):
    '''create a new embedding tensor or load from a pretrained embedding file

    Args:
        name: name of the embedding
        vocab_size : vocab size
        embed_size : embeddign size
        vocab_file ([type], optional): Defaults to None. vocab file
        embed_file ([type], optional): Defaults to None.
        num_trainable_tokens (int, optional): Defaults to 0. the number of tokens to be trained, if 0 then train all the tokens
        dtype ([type], optional): Defaults to tf.float32. [description]
        scope ([type], optional): Defaults to None. [description]

    Returns:
        embedding variable
    '''
    with tf.variable_scope(scope or "embedding", dtype=dtype) as scope:
        if vocab_file and embed_file:
            embedding = _load_pretrained_emb_from_file(name, vocab_file,
                                                       embed_file,
                                                       num_trainable_tokens,
                                                       dtype)
        else:
            device = _get_embed_device(vocab_size)
            if mode == 'inference':
                device = "/cpu:0"
            with tf.device(device):
                embedding = tf.get_variable(name, [vocab_size, embed_size],
                                            dtype)
        return embedding


def reverse_batch_seq(inputs, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return tf.reverse_sequence(input=inputs,
                                   seq_lengths=seq_lengths,
                                   seq_dim=seq_dim,
                                   batch_dim=batch_dim)
    else:
        return tf.reverse(inputs, axis=[seq_dim])


def focal_loss(labels, logits, num_class, gamma=2, alpha=0.0):
    """focal loss"""
    epsilon = 1.e-9

    # label smoothing
    K = float(num_class)
    labels = (1.0 - alpha) * labels + alpha / K

    y_pred = tf.nn.softmax(logits, dim=-1)
    y_pred = y_pred + epsilon  # to avoid 0.0 in log
    L = -labels * tf.pow((1 - y_pred), gamma) * tf.log(y_pred)
    L = tf.reduce_sum(L)
    batch_size = tf.shape(labels)[0]
    return L / tf.to_float(batch_size)


def show_param_num(params, threshold=1):
    total_parameters = 0
    for variable in params:
        local_parameters = 1
        shape = variable.get_shape()
        for i in shape:
            local_parameters *= i.value  # mutiplying dimension values
        if local_parameters >= threshold:
            print("variable {0} with parameter number {1}".format(
                variable, local_parameters))
        total_parameters += local_parameters
    print('# total parameter number', total_parameters)


def cal_f1(class_num, predicted, truth):
    "class_num -- 4; predicted -- [15000, 4]; truth -- [15000, 4]"
    results = []
    for i in range(class_num):
        results.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})

    for i, p in enumerate(predicted):
        t = truth[i]
        for j in range(class_num):
            if p[j] == 1:
                if t[j] == 1:
                    results[j]['TP'] += 1
                else:
                    results[j]['FP'] += 1
            else:
                if t[j] == 1:
                    results[j]['FN'] += 1
                else:
                    results[j]['TN'] += 1

    precision = [0.0] * class_num
    recall = [0.0] * class_num
    f1 = [0.0] * class_num
    for i in range(class_num):
        if results[i]['TP'] == 0:
            if results[i]['FP'] == 0 and results[i]['FN'] == 0:
                precision[i] = 1.0
                recall[i] = 1.0
                f1[i] = 1.0
            else:
                precision[i] = 0.0
                recall[i] = 0.0
                f1[i] = 0.0
        else:
            precision[i] = results[i]['TP'] / (results[i]['TP'] +
                                               results[i]['FP'])
            recall[i] = results[i]['TP'] / (results[i]['TP'] +
                                            results[i]['FN'])
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return sum(f1) / class_num, sum(precision) / class_num, sum(recall) / class_num


def load_config(out_dir, to_overide=None):
    config_file = os.path.join(out_dir, "config")
    print("loading config from %s" % config_file)
    config_json = json.load(open(config_file))

    config = tf.contrib.training.HParams()
    for k, v in config_json.items():
        config.add_hparam(k, v)
    if to_overide:
        for k, v in to_overide.items():
            if k not in config_json:
                config.add_hparam(k, v)
            else:
                config.set_hparam(k, v)
    return config


def save_config(out_dir, config):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    config_file = os.path.join(out_dir, "config")
    print("  saving config to %s" % config_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(config_file, "wb")) as f:
        f.write(config.to_json())


def get_config_proto(log_device_placement=True,
                     allow_soft_placement=True,
                     num_intra_threads=0,
                     num_inter_threads=0,
                     per_process_gpu_memory_fraction=0.95,
                     allow_growth=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                  allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = allow_growth
    config_proto.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads
    return config_proto


def early_stop(values, no_decrease=3):
    if len(values) < 2:
        return False
    best_index = np.argmin(values)
    if values[-1] > values[best_index] and (best_index + no_decrease) <= len(values):
        return True
    else:
        return False