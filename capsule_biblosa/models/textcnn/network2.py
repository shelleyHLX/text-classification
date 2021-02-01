# -*- coding:utf-8 -*-

from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from models.capsule.utils_wrapper import _conv2d_wrapper
from models.capsule.layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer
import tensorflow.contrib.slim as slim

class Settings(object):
    def __init__(self):
        self.model_name = 'TextCNN'
        self.fact_len = 200
        self.filter_sizes = [2, 3, 4, 5, 7]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 256
        self.n_class = 202
        self.batch_size = 32
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'

class capusle(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size
        self.n_class = settings.n_class
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.fc_hidden_size = settings.fc_hidden_size

        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'), tf.device('/cpu:0'):
            self._X_inputs = tf.placeholder(tf.int32, [None, self.fact_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')
            print('self._X_inputs.shape', self._X_inputs.shape)  # (32, 200)
            print('self._y_inputs.shape', self._y_inputs.shape)  # (32, 202)

        with tf.variable_scope('embedding'), tf.device('/cpu:0'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('capsule'):
            self._inference(self._X_inputs)
            # self._y_pred = tf.sigmoid(self.logits)
            # self._y_pred = self.logits
            # exit(0)

        # with tf.variable_scope('fc-bn-layer'):
        #     W_fc = self.weight_variable([self.n_filter_total, self.fc_hidden_size], name='Weight_fc')
        #     tf.summary.histogram('W_fc', W_fc)
        #     h_fc = tf.matmul(activations, W_fc, name='h_fc')
        #     beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
        #     tf.summary.histogram('beta_fc', beta_fc)
        #     fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
        #     self.update_emas.append(update_ema_fc)
        #     self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

        # with tf.variable_scope('out_layer'):
        #     W_out = self.weight_variable([self.fc_hidden_size, self.n_class], name='Weight_out')
        #     tf.summary.histogram('Weight_out', W_out)
        #     b_out = self.bias_variable([self.n_class], name='bias_out')
        #     tf.summary.histogram('bias_out', b_out)
        #     self._y_pred = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')  # 每个类别的分数 scores

        # with tf.name_scope('loss'):
        #     self._loss = self.margin_loss(self._y_inputs, self.y_pred)
        #     tf.summary.scalar('loss', self._loss)

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self._y_inputs))
            tf.summary.scalar('loss', self._loss)

        self.saver = tf.train.Saver(max_to_keep=1)

    @property
    def tst(self):
        return self._tst

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X_inputs(self):
        return self._X_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def loss(self):
        return self._loss

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def margin_loss(self, y, preds):
        y = tf.cast(y,tf.float32)
        loss = y * tf.square(tf.maximum(0., 0.9 - preds)) + \
               0.25 * (1.0 - y) * tf.square(tf.maximum(0., preds - 0.1))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        #    loss = tf.reduce_mean(loss)
        return loss

    def batchnorm(self, Ylogits, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, self._global_step)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(self.tst, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(self.tst, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def baseline_model_kimcnn(self, X, max_sent, num_classes):
        pooled_outputs = []
        for i, filter_size in enumerate([3,4,5]):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.n_filter]), name="b")
                conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool( h, ksize=[1, max_sent - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                    padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = 256 * 3
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print('h_pool_flat ', h_pool_flat)
        self.logits = slim.fully_connected(h_pool_flat, num_classes, scope='final_layer', activation_fn=None)
        self._y_pred = tf.sigmoid(self.logits)
        # return tf.zeros([0]), activations

    def _inference(self, X_inputs):
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        inputs = tf.expand_dims(inputs, -1)
        self.baseline_model_kimcnn(inputs, self.fact_len, self.n_class)
        # exit(0)
        # return poses, activations

"""
self._X_inputs.shape (32, 200)
self._y_inputs.shape (32, 202)
capsule_3
cnnout (32, 99, 1, 256)
into capsules_init
activations  (32, 99, 1, 16)
poses  (32, 99, 1, 16, 16)
out of capsules_init
poses_init (32, 99, 1, 16, 16)
activations_init (32, 99, 1, 16)
into capsule_conv_layer
hk_offsets  (97, 3)
wk_offsets  (1, 1)
inputs_poses_patches  (32, 97, 1, 3, 1, 16, 16)
inputs_poses_patches (3104, 48, 16)
i_activations_patches (32, 97, 1, 3, 1, 16)
i_activations_patches (3104, 48)
u_hat_vecs (3104, 16, 48, 16)
poses  (32, 97, 1, 16, 16)
activations  (32, 97, 1, 16)
out of capsule_conv_layer
poses_conv (32, 97, 1, 16, 16)
activations_conv (32, 97, 1, 16)
into capsule_flatten
activations  (32, 1552)
poses  (32, 1552, 16)
out of capsule_flatten
capsule_flatten (32, 1552, 16)
activations_flat (32, 1552)
into capsule_fc_layer
activations  (32, 202)
poses  (32, 202, 16)
out of capsule_fc_layer
poses  (32, 202, 16)
activations  (32, 202)
"""

"""
a
self._X_inputs.shape (32, 200)
self._y_inputs.shape (32, 202)
capsule_2
cnnout (32, 67, 1, 128)
into capsules_init
activations  (32, 23, 1, 64)
poses  (32, 23, 1, 64, 64)
out of capsules_init
poses_init (32, 23, 1, 64, 64)
activations_init (32, 23, 1, 64)
into capsule_conv_layer
hk_offsets  (21, 3)
wk_offsets  (1, 1)
inputs_poses_patches  (32, 21, 1, 3, 1, 64, 64)
inputs_poses_patches (672, 192, 64)
i_activations_patches (32, 21, 1, 3, 1, 64)
i_activations_patches (672, 192)
u_hat_vecs (672, 32, 192, 64)
poses  (32, 21, 1, 32, 64)
activations  (32, 21, 1, 32)
out of capsule_conv_layer
poses_conv (32, 21, 1, 32, 64)
activations_conv (32, 21, 1, 32)
into capsule_flatten
activations  (32, 672)
poses  (32, 672, 64)
out of capsule_flatten
poses_flat (32, 672, 64)
activations_flat (32, 672)
into capsule_fc_layer
activations  (32, 202)
poses  (32, 202, 64)
out of capsule_fc_layer
poses  (32, 202, 64)
activations  (32, 202)
capsule_3
cnnout (32, 66, 1, 128)
into capsules_init
activations  (32, 22, 1, 64)
poses  (32, 22, 1, 64, 64)
out of capsules_init
poses_init (32, 22, 1, 64, 64)
activations_init (32, 22, 1, 64)
into capsule_conv_layer
hk_offsets  (20, 3)
wk_offsets  (1, 1)
inputs_poses_patches  (32, 20, 1, 3, 1, 64, 64)
inputs_poses_patches (640, 192, 64)
i_activations_patches (32, 20, 1, 3, 1, 64)
i_activations_patches (640, 192)
u_hat_vecs (640, 32, 192, 64)
poses  (32, 20, 1, 32, 64)
activations  (32, 20, 1, 32)
out of capsule_conv_layer
poses_conv (32, 20, 1, 32, 64)
activations_conv (32, 20, 1, 32)
into capsule_flatten
activations  (32, 640)
poses  (32, 640, 64)
out of capsule_flatten
poses_flat (32, 640, 64)
activations_flat (32, 640)
into capsule_fc_layer
activations  (32, 202)
poses  (32, 202, 64)
out of capsule_fc_layer
poses  (32, 202, 64)
activations  (32, 202)
-------------------------------
poses  (32, 202, 64)
activations  (32, 202)
"""