#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   emlo.py
'''

# 说明
"""Emlo based long text fine-gained classification."""

import os
import numpy as np
import tensorflow as tf
from elmo_utils import create_embedding, reverse_batch_seq, focal_loss, show_param_num


class Model:
    def __init__(self, config):
        self.config = config
        self.mode = self.config.mode

    def build(self):
        """构建模型训练静态图"""
        self.init_placeholders()
        self.init_variables()
        self.init_embeddings()
        self.build_elmo()
        self.build_clf()

        self.params = tf.trainable_variables()
        # ema = decay * ema + (1 - decay) * actual_value
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)

        if self.config.mode in ['train', 'eval']:
            self.build_loss()
            if self.config.mode == 'train':
                self.setup_training()
                self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def init_placeholders(self):
        "输入数据placeholder"
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        self.inputs = tf.placeholder(tf.int32,
                                     shape=[None, None],
                                     name='inputs')
        if self.config.mode in ['train', 'eval']:
            self.targets = tf.placeholder(
                tf.float32,
                shape=[
                    None,
                    self.config.num_labels,
                    self.config.num_classes_each_label,
                ],
                name='targets')

    def init_variables(self):
        "部分全局变量读取"
        self.batch_size = tf.shape(self.inputs)[0]
        self.global_step = tf.Variable(initial_value=0,
                                       trainable=False,
                                       collections=[
                                           tf.GraphKeys.GLOBAL_STEP,
                                           tf.GraphKeys.GLOBAL_VARIABLES
                                       ],
                                       name='global_step')
        self.predict_token_num = tf.reduce_sum(self.seq_len)
        self.embedding_dropout = tf.Variable(self.config.embedding_dropout,
                                             trainable=False)
        self.dropout_keep_prob = tf.Variable(self.config.dropout_keep_prob,
                                             trainable=False)
        self.linear_dropout = tf.Variable(self.config.linear_dropout,
                                             trainable=False)

    def init_embeddings(self):
        """加载预训练词向量，初始化label embedding（20个）"""
        # ### init pretrained embedding ###
        self.embedding = create_embedding("embedding",
                                          self.config.vocab_size,
                                          self.config.embedding_size,
                                          vocab_file=self.config.vocab_file,
                                          embed_file=self.config.embed_file,
                                          mode=self.mode)
        if self.config.embedding_dropout > 0 and self.mode == 'train':
            vocab_size = tf.shape(self.embedding)[0]
            # 这里的dropout不需要normalize => * (1 - self.embedding_dropout)
            mask = tf.nn.dropout(tf.ones([vocab_size]), keep_prob=1 - self.embedding_dropout) * (1 - self.embedding_dropout)
            mask = tf.expand_dims(mask, 1)
            self.embedding = mask * self.embedding
        # [batch_size, seq_len, embed_dim]
        self.inputs_embedding = tf.nn.embedding_lookup(self.embedding,
                                                       self.inputs)

        # ### init label embedding ###
        labels = tf.range(self.config.num_labels, dtype=tf.int32)
        label_embedding_var = create_embedding("label_embedding",
                                               self.config.num_labels,
                                               self.config.embedding_size,
                                               mode=self.mode)
        label_embedding = tf.nn.embedding_lookup(label_embedding_var, labels)
        self.label_embedding = tf.tile(tf.expand_dims(label_embedding, axis=0),
                                       [self.batch_size, 1, 1])

        if self.mode == 'train':
            self.inputs_embedding = tf.nn.dropout(
                self.inputs_embedding, keep_prob=self.dropout_keep_prob)
            self.label_embedding = tf.nn.dropout(
                self.label_embedding, keep_prob=self.dropout_keep_prob)

    def build_elmo(self):
        "Output size: [batch_size, seq_len, 2 * hidden_size]"
        print('构建elmo静态图...')
        with tf.variable_scope("elmo_encoder") as scope:
            # [seq_len, batch_size, embed_dim]
            inputs = tf.transpose(self.inputs_embedding, [1, 0, 2])
            inputs_reverse = reverse_batch_seq(inputs,
                                               seq_lengths=self.seq_len,
                                               seq_dim=0,
                                               batch_dim=1)

            encoder_states = []
            outputs = [tf.concat([inputs, inputs], axis=-1)]

            fw_cell_inputs = inputs
            bw_cell_inputs = inputs_reverse
            for i in range(self.config.num_layers):
                # forword
                with tf.variable_scope("fw_%d" % i) as s:
                    # `FusedRNNCell` operates on the entire time sequence at once,
                    # by putting the loop over time inside the cell.
                    fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(self.config.num_units,
                                                                use_peephole=False)
                    # fw_output: [time_len, batch_size, num_units]
                    # fw_h：LSTMStateTuple(c, h)
                    fw_output, fw_h = fw_cell(fw_cell_inputs,
                                              sequence_length=self.seq_len,
                                              dtype=inputs.dtype)
                    encoder_states.append(fw_h)
                # backward
                with tf.variable_scope("bw_%d" % i) as s:
                    bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(self.config.num_units,
                                                                use_peephole=False)
                    bw_output, bw_h = bw_cell(bw_cell_inputs,
                                              sequence_length=self.seq_len,
                                              dtype=inputs.dtype)
                    bw_output_reverse = reverse_batch_seq(
                        bw_output,
                        seq_lengths=self.seq_len,
                        seq_dim=0,
                        batch_dim=1)
                    encoder_states.append(bw_h)

                # 该层每一步输出的tensor
                output = tf.concat([fw_output, bw_output_reverse], axis=-1)
                outputs.append(output)

                # resitual connections
                if i > 0:
                    fw_cell_inputs = output + fw_cell_inputs
                    bw_cell_inputs = reverse_batch_seq(
                        output,
                        seq_lengths=self.seq_len,
                        seq_dim=0,
                        batch_dim=1) + bw_cell_inputs
                else:
                    fw_cell_inputs = output
                    bw_cell_inputs = reverse_batch_seq(
                        output,
                        seq_lengths=self.seq_len,
                        seq_dim=0,
                        batch_dim=1)

            n = 1 + self.config.num_layers  # embedding + num_layers
            self.weight = tf.get_variable('weight',
                                          initializer=tf.constant([1 / (n)] * n))
            # 每一层lstm hidden state和inputs的权重。取为相同的值
            soft_weight = tf.nn.softmax(self.weight)
            # final_output的缩放比例
            self.scalar = tf.get_variable('scalar',
                                          initializer=tf.constant(0.001))

            final_outputs = None
            for i, output in enumerate(outputs):
                if final_outputs is None:
                    # [batch_size, seq_len, 2 * hidden_size]
                    final_outputs = soft_weight[i] * tf.transpose(
                        output, [1, 0, 2])
                else:
                    final_outputs += soft_weight[i] * tf.transpose(
                        output, [1, 0, 2])

            # [batch_size, seq_len, 2 * hidden_size]
            self.final_outputs = self.scalar * final_outputs
            self.final_state = tuple(encoder_states)

    def build_clf(self):
        "使用类别分布，修正训练预测的结果"
        num_units = self.config.num_units
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE) as scope:
            states = self._attention_op()
            all_logits = []
            all_predicts = []
            weights = self._read_class_weights(self.config.weight_file)
            with tf.variable_scope("predict_clf"):
                hidden_layer = tf.layers.Dense(num_units,
                                               use_bias=True,
                                               activation=tf.nn.swish)
                dropout = tf.layers.Dropout(self.linear_dropout)
                output_layer = tf.layers.Dense(self.config.num_classes_each_label)
                # state ： [context_i, hidden_i]
                for i, state in enumerate(states):  # 每一种label单独输出
                    hidden = hidden_layer(state)
                    hidden = dropout(hidden)
                    each_label_logits = output_layer(hidden)
                    all_logits.append(each_label_logits)

                    if weights is not None:
                        probs = tf.nn.softmax(each_label_logits * 10)
                        weight = tf.constant(weights[i], dtype=tf.float32)
                        each_label_logits = tf.multiply(probs, weight)
                    predict = tf.argmax(each_label_logits, axis=-1)
                    predict = tf.one_hot(predict, self.config.num_classes_each_label)
                    all_predicts.append(predict)

            self.all_logits = tf.concat([tf.expand_dims(l, 1) for l in all_logits], axis=1)
            self.all_predicts = tf.concat([tf.expand_dims(p, 1) for p in all_predicts], axis=1)

            if self.config.mode in ['train', 'eval']:
                self.accurary = tf.contrib.metrics.accuracy(
                    tf.to_int32(self.all_predicts), tf.to_int32(self.targets))

    @staticmethod
    def _read_class_weights(weight_file):
        "注意weight与one-hot对应关系：[1, 0, -1, -2]"
        if weight_file is None:
            return None
        import pickle
        with open(weight_file, 'rb') as f:
            class_weights = pickle.load(f, encoding='utf-8')
        return class_weights

    def _attention_op(self):
        """AttentionWrapper实现相对于理论算法更复杂一些。
        简而言之，
        增加了attention layer，将attention算法中得到的context vector与decoder当前
            输出cell_outputs(即hidden state)通过计算得到一个attention向量。当attention layer没有
            指定时，attention向量直接取context vector(即，算法理论中的计算方式)。

        增加了cell_input_fn，将上一步的attention向量与当前步的inputs，联合成新的cell_inputs。

        attention mechanism：输入decoder的cell_outputs(即hidden state)，与memory(encoder的hidden state)
            计算alignments(权重)
        """
        num_units = self.config.num_units
        with tf.variable_scope("attention_op") as scope:
            cell = self._rnn_cell(self.config.rnn_cell_name,
                                  num_units,
                                  self.mode,
                                  self.dropout_keep_prob,
                                  self.config.weight_keep_drop)
            # memory to query: self.final_outputs
            # num_units must match expected the query dims. memory会dense到num_units的维度
            attention = tf.contrib.seq2seq.LuongAttention(num_units,
                                                          self.final_outputs,
                                                          self.seq_len,
                                                          scale=True)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention, attention_layer_size=300, output_attention=True)

            # 初始化decoder state
            if 'lstm' in self.config.rnn_cell_name.lower():
                h = tf.layers.dense(tf.concat([state.h for state in self.final_state], axis=-1),
                                    num_units,
                                    use_bias=True)
                c = tf.layers.dense(tf.concat([state.c for state in self.final_state], axis=-1),
                                    num_units,
                                    use_bias=True)
                initial_state = attn_cell.zero_state(
                    self.batch_size, dtype=tf.float32).clone(
                        cell_state=tf.contrib.rnn.LSTMStateTuple(c=c, h=h))
            else:
                h = tf.layers.dense(tf.concat([state.h for state in self.final_state], axis=-1),
                                    num_units,
                                    use_bias=True)
                initial_state = attn_cell.zero_state(
                    self.batch_size, dtype=tf.float32).clone(cell_state=h)

            outputs = []
            state = initial_state
            for i in range(self.config.num_labels):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # label_embedding 作为query
                inputs = self.label_embedding[:, i, :]
                # 计算decoder每一步的输出
                context, state = attn_cell(inputs, state)
                if 'lstm' in self.config.rnn_cell_name.lower():
                    out_state = tf.concat([state.cell_state.h + inputs, context + inputs],
                                          axis=-1)
                else:
                    out_state = tf.concat([state.cell_state + inputs, context + inputs],
                                          axis=-1)
                outputs.append(out_state)
            return outputs

    @staticmethod
    def _rnn_cell(cell_name,
                  num_units,
                  is_training=True,
                  keep_prob=0.75,
                  weight_keep_drop=0.65):
        "LSTM + dropout"
        cell_name = cell_name.upper()
        if cell_name == "GRU":
            cell = tf.contrib.rnn.GRUCell(num_units)
        elif cell_name == "LSTM":
            cell = tf.contrib.rnn.LSTMCell(num_units)
        elif cell_name == 'BLOCK_LSTM'.upper():
            cell = tf.contrib.rnn.LSTMBlockCell(num_units)
        elif cell_name == 'WEIGHT_LSTM':
            from weight_drop_lstm import WeightDropLSTMCell
            if is_training and weight_keep_drop < 1.0:
                mode = tf.estimator.ModeKeys.TRAIN
            else:
                mode = tf.estimator.ModeKeys.PREDICT
            cell = WeightDropLSTMCell(num_units,
                                      weight_keep_drop=weight_keep_drop,
                                      mode=mode)
        elif cell_name == 'LAYERNORM_LSTM':
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(num_units)

        if is_training:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                 input_keep_prob=keep_prob,
                                                 output_keep_prob=keep_prob,
                                                 dtype=tf.float32)

        return cell

    def build_loss(self):
        "选择两种loss计算方式，加入label smoothing. 20和4直接是输入了num_labels, num_classes_each_label。"
        if self.config.focal_loss_gamma > 0:
            self.gamma = tf.Variable(self.config.focal_loss_gamma,
                                     dtype=tf.float32,
                                     trainable=False)
            label_losses = tf.constant(0.0, tf.float32)
            for i in range(20):
                label_losses += focal_loss(self.targets[i * 4:(i + 1) * 4],
                                           self.all_logits[i * 4:(i + 1) * 4],
                                           self.config.num_classes_each_label,
                                           self.gamma,
                                           self.config.label_smoothing)
        elif self.config.loss_name=='softmax_multi':
            label_losses = tf.constant(0.0, tf.float32)
            for i in range(20):
                label_losses += tf.losses.softmax_cross_entropy(
                    onehot_labels=self.targets[i * 4:(i + 1) * 4],
                    logits=self.all_logits[i * 4:(i + 1) * 4],
                    reduction=tf.losses.Reduction.MEAN,
                    label_smoothing=self.config.label_smoothing)
        elif self.config.loss_name=='softmax':
            label_losses = tf.losses.softmax_cross_entropy(onehot_labels=self.targets,
                                                        logits=self.all_logits,
                                                        reduction=tf.losses.Reduction.MEAN)
        elif self.config.loss_name=='sigmoid':
            label_losses = tf.losses.sigmoid_cross_entropy(onehot_labels=self.targets,
                                                        logits=self.all_logits,
                                                        reduction=tf.losses.Reduction.MEAN)
        self.losses = label_losses

    def setup_training(self):
        # learning rate decay
        if self.config.decay_schema == 'exp':
            self.learning_rate = tf.train.exponential_decay(
                self.config.learning_rate,
                self.global_step,
                self.config.decay_steps,
                0.96,
                staircase=True)
        else:
            self.learning_rate = tf.Variable(self.config.learning_rate,
                                             dtype=tf.float32,
                                             trainable=False)

        self.param_norm = tf.global_norm(self.params)
        if self.config.fix_embedding:
            train_var_list = [var for var in tf.trainable_variables() if 'embedding' not in var.name]
            params = train_var_list
            show_param_num(params)
        else:
            params = self.params
            # show params and statistic
            show_param_num(params)

        # L2 norm
        if self.config.l2_loss_ratio > 0:
            l2_loss = self.config.l2_loss_ratio * tf.add_n([
                tf.nn.l2_loss(p) for p in params
                if ('predict_clf' in p.name and 'bias' not in p.name)
            ])
            self.losses += l2_loss

        # gradients clipping
        gradients = tf.gradients(self.losses,
                                 params,
                                 colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.config.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)

        # optimizer, exponential_moving_average
        if self.config.optimizer.lower() == 'rms':
            opt = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.config.optimizer.lower() == 'adam':
            opt = tf.train.AdamOptimizer(self.learning_rate)
        train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step=self.global_step)
        with tf.control_dependencies([train_op]):
            train_op = self.ema.apply(self.params)
        self.train_op = train_op

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(self.config.checkpoint_dir,
                                                    tf.get_default_graph())
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy", self.accurary)
        tf.summary.scalar('gradient_norm', self.gradient_norm)
        tf.summary.scalar('param_norm', self.param_norm)
        self.summary_op = tf.summary.merge_all()

    def init_model(self, sess, initializer=None):
        if initializer:
            sess.run(initializer)
        else:
            sess.run(tf.global_variables_initializer())

    def save_model(self, sess, global_step=None):
        return self.saver.save(sess,
                               os.path.join(self.config.checkpoint_dir, "model.ckpt"),
                               global_step=global_step if global_step else self.global_step)

    def restore_best_model(self, sess):
        self.saver.restore(sess,
                           tf.train.latest_checkpoint(self.config.checkpoint_dir + '/best_dev'))

    def restore_ema_model(self, sess, path):
        shadow_vars = {self.ema.average_name(v): v for v in self.params}
        saver = tf.train.Saver(shadow_vars)
        saver.restore(sess, path)

    def restore_model(self, sess, global_step=None):
        if global_step is None:
            self.saver.restore(sess,
                               tf.train.latest_checkpoint(self.config.checkpoint_dir))
        else:
            print(os.path.join(self.config.checkpoint_dir, "model.ckpt-%d" % global_step))
            self.saver.restore(sess,
                               os.path.join(self.config.checkpoint_dir, "model.ckpt-%d" % global_step))
        print('')
        print("!!! Restored model")

    def train_clf_one_step(self, sess, source, lengths, targets, add_summary=False, run_info=False):
        feed_dict = {}
        feed_dict[self.inputs] = source
        feed_dict[self.seq_len] = lengths
        feed_dict[self.targets] = targets
        if run_info:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, batch_loss, summary, global_step, accuracy, token_num, batch_size = sess.run(
                [
                    self.train_op,
                    self.losses,
                    self.summary_op,
                    self.global_step,
                    self.accurary,
                    self.predict_token_num,
                    self.batch_size
                ],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)

        else:
            _, batch_loss, summary, global_step, accuracy, token_num, batch_size = sess.run(
                [
                    self.train_op,
                    self.losses,
                    self.summary_op,
                    self.global_step,
                    self.accurary,
                    self.predict_token_num,
                    self.batch_size
                ],
                feed_dict=feed_dict)

        if run_info:
            self.summary_writer.add_run_metadata(run_metadata, 'step%03d' % global_step)
            print("adding run meta for", global_step)

        if add_summary:
            self.summary_writer.add_summary(summary, global_step=global_step)
        return batch_loss, global_step, accuracy, token_num, batch_size

    def eval_clf_one_step(self, sess, source, lengths, targets):
        feed_dict = {}
        feed_dict[self.inputs] = source
        feed_dict[self.seq_len] = lengths
        feed_dict[self.targets] = targets

        batch_loss, accuracy, batch_size, predict = sess.run(
            [self.losses, self.accurary,self.batch_size, self.all_predicts],
            feed_dict = feed_dict
        )
        return batch_loss, accuracy, batch_size, predict

    def inference_clf_one_batch(self, sess, source, lengths):
        feed_dict = {}
        feed_dict[self.inputs] = source
        feed_dict[self.seq_len] = lengths

        predict, logits = sess.run([self.all_predicts, self.all_logits],
                                   feed_dict=feed_dict)
        return predict, logits