#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import tensorflow as tf


class DropConnectLayer(tf.layers.Dense):
    def __init__(self,
                 units,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 keep_prob=0.7,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(DropConnectLayer, self).__init__(units,
                                            activation=activation,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint,
                                            trainable=trainable,
                                            name=name,
                                            **kwargs)
        self.mode = mode
        self.keep_prob = keep_prob
        self.mask = None

    def build(self, input_shape):
        from tensorflow.python.layers import base
        from tensorflow.python.framework import tensor_shape

        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            if self.mask is None:
                mask = tf.ones_like(self.kernel)
                self.mask = tf.nn.dropout(mask, keep_prob=self.keep_prob) * self.keep_prob
            self.kernel = self.kernel * self.mask
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[self.units, ],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True


class WeightDropLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    '''Apply dropout on recurrent hidden to hidden weights.'''
    def __init__(self,
                 num_units,
                 weight_keep_drop=0.7,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None):
        super(WeightDropLSTMCell,self).__init__(num_units,
                                                forget_bias,
                                                state_is_tuple,
                                                activation,
                                                reuse)
        self.w_layer = tf.layers.Dense(4 * num_units)
        self.h_layer = DropConnectLayer(4 * num_units,
                                        mode,
                                        weight_keep_drop,
                                        use_bias=False)

    def build(self, inputs_shape):
        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`.
            state: An `LSTMStateTuple` of state tensors, each shaped
                `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
                `True`.  Otherwise, a `Tensor` shaped
                `[batch_size x 2 * self.state_size]`.
        Returns:
            A pair containing the new hidden state, and the new state (either a
                `LSTMStateTuple` or a concatenated state, depending on
                `state_is_tuple`).
        """
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        # W * x + b
        inputs = self.w_layer(inputs)
        # U * h(t-1)
        h = self.h_layer(h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=inputs + h, num_or_size_splits=4, axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state