#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi_class_cnn.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

# 说明: 简单多任务text cnn实现。多种类标签预测。


import tensorflow as tf


class TCNNConfig:
    """CNN配置参数"""

    embedding_dim = 64       # 词向量维度
    seq_length = 600         # 序列长度
    multi_size = 5           # 多任务数量
    num_classes = [4, 4, 4, 4, 4]  # 每个任务类别数
    num_filters = 128        # 卷积核数目
    kernel_sizes = [3, 4, 5]  # 多种卷积核尺寸
    vocab_size = 5000        # 词汇表达小

    hidden_dim = 128         # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3     # 学习率

    batch_size = 64          # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextCNN:
    """多任务文本分类，CNN模型"""

    def __init__(self, config):
        """NOTE:
        input_y在多任务模式下，input_y为列表: [[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1]]。分别表示每个任务下的标签

        train_op：相比于单任务模式下的self.optim，调用self.train_ops列表。
        logits，y_pred_cls，loss，acc：相比与单任务模式，都是多了一维 task dim，都是以列表表示。
        """
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        pooling_output = []
        for kernel_size in self.config.kernel_sizes:
            with tf.name_scope("cnn_size_{kernel_size}"):
                # CNN layer
                conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, kernel_size, name='conv')
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                pooling_output.append(gmp)
        # Combine all the pooled features
        gmp = tf.concat(1, pooling_output)
        gmp = tf.reshape(gmp, [self.config.batch_size, -1])


        self.logits = []
        self.y_pred_cls = []
        self.loss = []
        self.acc = []
        for i in range(self.config.multi_size):
            with tf.name_scope("score_task_{i}"):
                # 全连接层，后面接dropout以及relu激活
                fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)

                # 分类器
                self.logits.append(tf.layers.dense(fc, self.config.num_classes, name='fc2'))
                self.y_pred_cls.append(tf.argmax(tf.nn.softmax(self.logits), 1))  # 预测类别

                # 损失函数，交叉熵
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[i], labels=self.input_y[i])
                self.loss.append(tf.reduce_mean(cross_entropy))

                with tf.name_scope("accuracy_{i}"):
                    # 准确率
                    correct_pred = tf.equal(tf.argmax(self.input_y[i], 1), self.y_pred_cls[i])
                    self.acc.append(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))

        with tf.name_scope("optimize"):
            # 优化器
            self.train_ops = []
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            for i in range(self.config.multi_size):
                grads_and_vars=self.optim.compute_gradients(self.loss[i])
                self.train_ops.append(self.optim.apply_gradients(grads_and_vars))