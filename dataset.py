import json
import codecs
import os
from collections import namedtuple

import numpy as np
import tensorflow as tf


def read_vocab(vocab_file):
    """read vocab from file

    Args:
        vocab_file ([type]): path to the vocab file,
            the vocab file should contains a word each line
    Returns:
        list of words
    """

    if not os.path.isfile(vocab_file):
        raise ValueError("%s is not a vaild file"%vocab_file)

    vocab = []
    word2id = {}
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        for i,line in enumerate(f):
            word = line.strip()
            if not word:
                raise ValueError("Got empty word at line %d"%(i+1))
            vocab.append(word)
            word2id[word] = len(word2id)

    print("# vocab size: ", len(vocab))
    return vocab, word2id


class DataItem(namedtuple("DataItem", ('content', 'length', 'labels', 'id'))):
    pass


def _padding(tokens_list, max_len):
    ret = np.zeros((len(tokens_list), max_len), np.int32)
    for i, t in enumerate(tokens_list):
        t = t + (max_len - len(t)) * [EOS]
        ret[i] = t
    return ret


def _tokenize(content, w2i, max_tokens=1200, reverse=False, split=True):
    """word转id

    split -- 当一个词不存在时，退一步查询单个字的id"""
    def get_tokens(content):
        tokens = content.strip().split()
        ids = []
        for t in tokens:
            if t in w2i:
                ids.append(w2i[t])
            else:
                for c in t:
                    ids.append(w2i.get(c, UNK))
        return ids

    if split:
        ids = get_tokens(content)
    else:
        ids = [w2i.get(t, UNK) for t in content.strip().split()]
    if reverse:
        ids = list(reversed(ids))
    tokens = [SOS] + ids[:max_tokens] + [EOS]
    return tokens


UNK = 0
SOS = 1
EOS = 2

class DataSet(object):
    def __init__(self,
                 data_files,
                 vocab_file,
                 label_file,
                 batch_size=32,
                 reverse=False,
                 split_word=True,
                 max_len=1200):
        self.reverse = reverse
        self.split_word = split_word
        self.data_files = data_files
        self.batch_size = batch_size
        self.max_len = max_len

        self.vocab, self.w2i = read_vocab(vocab_file)
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.label_names, self.l2i = read_vocab(label_file)
        self.i2l = {v: k for k, v in self.l2i.items()}

        self.tag_l2i = {"1": 0, "0": 1, "-1": 2, "-2": 3}
        self.tag_i2l = {v: k for k, v in self.tag_l2i.items()}

        self._raw_data = []
        self.items = []
        self._preprocess()

    def get_label(self, labels, l2i, normalize=False):
        "每一种评分，进行onehot的方法. one-hot对应关系：[1, 0, -1, -2]"
        one_hot_labels = np.zeros(len(l2i), dtype=np.float32)
        for n in labels:
            if n:
                one_hot_labels[l2i[n]] = 1
        if normalize:
            one_hot_labels = one_hot_labels / len(labels)
        return one_hot_labels

    def _preprocess(self):
        print("# Start to preprocessing data...")
        idx = 0
        for fname in self.data_files:
            print("# load data from %s ..." % fname)
            for line in open(fname, encoding='utf-8'):
                item = json.loads(line.strip(), encoding='uft-8')
                content = item['content']
                content = _tokenize(content,
                                    self.w2i,
                                    self.max_len,
                                    self.reverse,
                                    self.split_word)
                item_labels = []
                for label_name in self.label_names:
                    labels = [item[label_name]]
                    labels = self.get_label(labels, self.tag_l2i)
                    item_labels.append(labels)
                # item_labels： num_label_type(20) * each_type_class_num(4)，one hot
                self._raw_data.append(
                    DataItem(content=content,
                             labels=np.asarray(item_labels),
                             length=len(content),
                             id=idx))
                self.items.append(item)
                idx += 1

        self.num_batches = len(self._raw_data) // self.batch_size
        self.data_size = len(self._raw_data)
        print("# Got %d data items with %d batches" %
                  (self.data_size, self.num_batches))

    def _shuffle(self):
        # code from https://github.com/fastai/fastai/blob/3f2079f7bc07ef84a750f6417f68b7b9fdc9525a/fastai/text.py#L125
        idxs = np.random.permutation(self.data_size)
        # 划分小的chunk，进行长度排序
        sz = self.batch_size * 50
        ck_idx = [idxs[i:i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([
            sorted(s, key=lambda x: self._raw_data[x].length, reverse=True)
            for s in ck_idx
        ])
        # 划分batch
        sz = self.batch_size
        # batch idxs
        ck_idx = [sort_idx[i:i + sz] for i in range(0, len(sort_idx), sz)]
        # 最长序列
        max_ck = np.argmax([self._raw_data[ck[0]].length for ck in ck_idx])
        # 最长序列为第一个
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]
        # concatenate得到长度接近的idxs排序结果
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

    def process_batch(self, batch):
        contents = [item.content for item in batch]
        lengths = [item.length for item in batch]
        contents = _padding(contents, max(lengths))
        lengths = np.asarray(lengths)
        targets = np.asarray([item.labels for item in batch])
        idx = [item.id for item in batch]
        return contents, lengths, targets, idx

    def get_next(self, shuffle=True):
        "labels： num_label_type(20) * each_type_class_num(4)，one hot"
        if shuffle:
            idxs = self._shuffle()
        else:
            idxs = range(self.data_size)

        batch = []
        for i in idxs:
            item = self._raw_data[i]  # item:('content','length','labels','id')
            if len(batch) >= self.batch_size:
                yield self.process_batch(batch)
                batch = [item]
            else:
                batch.append(item)
        if len(batch) > 0:
            yield self.process_batch(batch)
