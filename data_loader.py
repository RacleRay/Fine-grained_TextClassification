#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_loader.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

# 说明
"""
17 行：For inputs reading
97 行: For inputs analysis
123行: For raw data processing
"""
#================================================================
#For inputs reading
#================================================================
import numpy as np
import tensorflow.contrib.keras as keras
from collections import Counter


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示。完成padding和label process。"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转换为one-hot表示
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成batch数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def read_file(filename, mode='r', encoding='utf-8'):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, mode=mode, encoding=encoding, errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def read_vocab(vocab_dir, mode='r', encoding='utf-8'):
    """读取词汇表"""
    with open(vocab_dir, mode=mode, encoding=encoding, errors='ignore') as f:
        words = [w.strip() for w in f.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = {i: w for w, i in word_to_id.items()}
    return words, word_to_id, id_to_word


def read_category(categories):
    """读取分类目录

    categories -- 手动输入列表, list。
                  e.g: ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    """
    # {'体育':0, '财经':1, '房产':2...}
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


#================================================================
#For inputs analysis
#================================================================
import pickle


def build_vocab(train_dir, vocab_dir, frequence_dir=None, vocab_size=5000):
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)  # [('a', 5), ('b', 4), ('c', 3)]
    words, frequence = list(zip(*count_pairs))                 # ['a', 'b', 'c'], [5, 4, 3]
    # <PAD>, <UNK>
    words = ['<PAD>', '<UNK>'] + list(words)
    with open(vocab_dir, mode='w', encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')

    if frequence_dir:
        with open(frequence_dir, mode='wb') as f:
            pickle.dump(frequence_dir, f)


#================================================================
#For raw data processing
#================================================================
import os
import re


def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        string = f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')
        # 删除对断句没有帮助的符号
        pattern_1 = re.compile(
            ur"\(|\)|（|）|\"|“|”|\*|《|》|<|>|&|#|~|·|`|=|\+|\}|\{|\||、|｛|｝|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〿|–—|…|‧|﹏|")
        string = re.sub(pattern_1, " ", string)
        # 断句符号统一为中文符号
        string = re.sub(r"!", "！", string)
        string = re.sub(r"\?", "？", string)
        string = re.sub(r";", "；", string)
        string = re.sub(r",", "，", string)
        # 去除网站，图片引用
        string = re.sub(r"[！a-zA-z]+://[^\s]*", "", string)
        # 去除邮箱地址
        string = re.sub(r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", "", string)

        string = re.sub(r"@", "", string)
        string = string.replace(' ', '').lower()

        # string = re.sub(r"\'s", " \'s", string)
        # string = re.sub(r"\'ve", " \'ve", string)
        # string = re.sub(r"n\'t", " n\'t", string)
        # string = re.sub(r"\'re", " \'re", string)
        # string = re.sub(r"\'d", " \'d", string)
        # string = re.sub(r"\'ll", " \'ll", string)

    return string


def save_file(dirname, output_dir, trainsize, val_size, test_size, encoding='utf-8'):
    """
    将多个文件整合并存到3个文件中，生成分类数据集。保存为txt格式。
    dirname: 原数据目录，不同类别文件保存在不同文件夹下
    文件内容格式:  类别\t内容
    """
    f_train = open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8')
    f_val = open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8')
    for category in os.listdir(dirname):
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)  # 一个文件为一行
            if count < trainsize:
                f_train.write(category + '\t' + content + '\n')
            elif count < trainsize + test_size:
                f_test.write(category + '\t' + content + '\n')
            elif count < trainsize + test_size + val_size:
                f_val.write(category + '\t' + content + '\n')
            count += 1
        print('Finished:', category)
    f_train.close()
    f_test.close()
    f_val.close()