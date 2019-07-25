# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 20:03
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model.py
# @Software: PyCharm


import tensorflow as tf
import os
from .model_utils import feature2cos_sim
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer


class SiameseQACNN(object):
    def __init__(self, config):
        self.ques_len = config.ques_length
        self.ans_len = config.ans_length
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.learning_rate = config.learning_rate
        self.optimizer = config.optimizer
        self.l2_lambda = config.l2_lambda
        self.clip_value = config.clip_value
        self.embeddings = config.embeddings
        self.window_sizes = config.window_sizes
        self.n_filters = config.n_filters
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size

        self._placeholder_init_pointwise()
        self.logits, self.res = self._build(self.embeddings)
        # 损失和精确度
        self.total_loss= self._add_loss_op(self.logits)
        # 训练节点
        self.train_op = self._add_train_op(self.total_loss)

    def _placeholder_init_pointwise(self):
        self._ques = tf.placeholder(tf.int32, [None, self.ques_len], name='ques_point')
        self._ans = tf.placeholder(tf.int32, [None, self.ans_len], name='ans_point')
        self._y = tf.placeholder(tf.int32, [None], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self._ans)[0], tf.shape(self._ans)[1]

    def _HL_layer(self, bottom, n_weight, name):
        """
        全连接层
        """
        assert len(bottom.get_shape()) == 3
        n_prev_weight = bottom.get_shape()[-1]
        max_len = bottom.get_shape()[1]
        initer = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight],
                            initializer=initer,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001))
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.1, shape=[n_weight], dtype=tf.float32),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001))
        bottom_2 = tf.reshape(bottom, [-1, n_prev_weight])
        hl = tf.nn.bias_add(tf.matmul(bottom_2, W), b)
        hl_tanh = tf.nn.tanh(hl)
        HL = tf.reshape(hl_tanh, [-1, max_len, n_weight])
        return HL

    def fc_layer(self, bottom, n_weight, name):
        """
        全连接层
        """
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer,
                            regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001))
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def _network(self, x):
        fc1 = self.fc_layer(x, self.hidden_size, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, self.hidden_size, "fc2")
        return fc2

    def _cnn_layer(self, input):
        """
        卷积层
        """
        all = []
        max_len = input.get_shape()[1]
        for i, filter_size in enumerate(self.window_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                # 卷积
                cnn_out = tf.layers.conv1d(input, self.n_filters, filter_size, padding='valid',
                                           activation=tf.nn.relu, name='q_conv_' + str(i))
                # 池化
                pool_out = tf.reduce_max(cnn_out, axis=1, keepdims=True)
                tanh_out = tf.nn.tanh(pool_out)
                all.append(tanh_out)
        cnn_outs = tf.concat(all, axis=-1)
        dim = cnn_outs.get_shape()[-1]
        cnn_outs = tf.reshape(cnn_outs, [-1, dim])
        return cnn_outs

    def _build(self, embeddings):
        if embeddings is not None:
            self.Embedding = tf.Variable(tf.to_float(embeddings), trainable=False, name='Embedding')
        else:
            self.Embedding = tf.get_variable('Embedding', shape=[self.vocab_size, self.embedding_size],
                                         initializer=tf.uniform_unit_scaling_initializer())
        self.q_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self._ques), keep_prob=self.dropout_keep_prob)
        self.a_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self._ans), keep_prob=self.dropout_keep_prob)

        with tf.variable_scope('siamese') as scope:
            # 计算隐藏和卷积层
            hl_q = self._HL_layer(self.q_embed, self.hidden_size, 'HL_layer')
            conv1_q = self._cnn_layer(hl_q)
            scope.reuse_variables()
            hl_a = self._HL_layer(self.a_embed, self.hidden_size, 'HL_layer')
            conv1_a = self._cnn_layer(hl_a)
        with tf.variable_scope('fc') as scope:
            con = tf.concat([conv1_q, conv1_a], axis=-1)
            logits = self.fc_layer(con, 1, 'fc_layer')
            res = tf.nn.sigmoid(logits)
        return logits, res


    def _add_loss_op(self, logits):
        """
        损失节点
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                         labels=tf.cast(tf.reshape(self._y, [-1, 1]),
                                                                        dtype=tf.float32))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = sum(reg_losses)
        pointwise_loss = tf.reduce_mean(loss) + l2_loss
        tf.summary.scalar('pointwise_loss', pointwise_loss)

        return pointwise_loss

    def _add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # 计算梯度,得到梯度和变量
            gradsAndVars = optimizer.compute_gradients(loss)
            # 将梯度应用到变量下，生成训练器
            train_op = optimizer.apply_gradients(gradsAndVars, global_step=self.global_step)
            # 用summary绘制tensorBoard
            for g, v in gradsAndVars:
                if g is not None:
                    tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

            self.summary_op = tf.summary.merge_all()
            return train_op