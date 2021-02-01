# -*- coding:utf-8 -*-

from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from models.capsule.utils_cap import _conv2d_wrapper
from models.capsule.layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer


class Settings(object):
    def __init__(self):
        self.model_name = 'capsuleb'
        self.fact_len = 200
        self.filter_sizes = [2, 3, 4, 5, 7]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 256
        self.n_class = 202
        self.batch_size = 128
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

        self._global_step = tf.Variable(0, trainable=False)
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'), tf.device('/cpu:0'):
            self._X_inputs = tf.placeholder(tf.int32, [32, self.fact_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [32, self.n_class], name='y_input')
            print('self._X_inputs.shape', self._X_inputs.shape)  # (32, 200)
            print('self._y_inputs.shape', self._y_inputs.shape)  # (32, 202)

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('capsule'):
            poses, activations = self._inference(self._X_inputs)
            bs, hw = activations.get_shape().as_list()
            # self._y_pred = tf.sigmoid(self.logits)
            # exit(0)

        with tf.variable_scope('fc-bn-layer'):
            W_fc = self.weight_variable([hw, hw], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(activations, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[hw], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

        with tf.variable_scope('out_layer'):
            W_out = self.weight_variable([hw, self.n_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self.logits = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')  # 每个类别的分数 scores
            self._y_pred = tf.sigmoid(self.logits)

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
    #
    def capsuleb(self, inputs):
        poses_list = []
        for _, ngram in enumerate([3]):
            with tf.variable_scope('capsule_' + str(ngram)):
                print('capsule_' + str(ngram))
                cnnout = _conv2d_wrapper(
                    inputs, shape=[ngram, self.embedding_size, 1, 128], strides=[1, 3, 1, 1], padding='VALID',
                    add_bias=True, activation_fn=tf.nn.relu, name='conv1'
                )
                print('cnnout', cnnout.shape)
                # (32, 50, 1, 32)

                tf.logging.info('output shape: {}'.format(cnnout.get_shape()))
                poses_init, activations_init = capsules_init(cnnout, shape=[1, 1, 128, 64], strides=[1, 3, 1, 1],
                                                             padding='VALID', pose_shape=64, add_bias=True, name='primary')
                print('poses_init', poses_init.shape)
                # (32, 17, 1, 16, 16)
                print('activations_init', activations_init.get_shape())
                # (32, 17, 1, 16)
                poses_conv, activations_conv = capsule_conv_layer(poses_init, activations_init, shape=[3, 1, 64, 32],
                                                                  strides=[1, 1, 1, 1], iterations=2, name='conv2')
                print('poses_conv', poses_conv.shape)  # (32, 15, 1, 16, 16)
                print('activations_conv', activations_conv.shape)  # (32, 15, 1, 16)
                poses_flat, activations_flat = capsule_flatten(poses_conv, activations_conv)
                print('poses_flat', poses_flat.shape)  # (32, 240, 16)
                print('activations_flat', activations_flat.shape)  # (32, 240)

                poses, activations = capsule_fc_layer(poses_flat, activations_flat, self.n_class, 3, 'fc2')
                print('poses ', poses.shape)  # (32, 202, 16)
                print('activations ', activations.shape)  # (32, 202)
                poses_list.append(poses)
        print('-------------------------------')
        poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0)
        print('poses ', poses.shape)
        activations = K.sqrt(K.sum(K.square(poses), 2))

        print('activations ', activations.shape)
        return poses, activations

    # def capsulea(self, X):
    #     with tf.variable_scope('capsule_' + str(3)):
    #         print('capsule_' + str(3))
    #         self.cnnout = _conv2d_wrapper(X, shape=[3, self.embedding_size, 1, 256], strides=[1, 3, 1, 1], padding='VALID',
    #                                  add_bias=True, activation_fn=tf.nn.relu, name='conv1')
    #         print('cnnout', self.cnnout.shape)  # (32, 99, 1, 256)
    #         tf.logging.info('output shape: {}'.format(self.cnnout.get_shape()))
    #         poses, activations = capsules_init(self.cnnout, shape=[1, 1, 256, 512], strides=[1, 3, 1, 1],
    #                                                      padding='VALID', pose_shape=512, add_bias=True, name='primary')
    #         print('poses_init', poses.shape)  # (32, 99, 1, 16, 16)
    #         print('activations_init', activations.get_shape())  # (32, 99, 1, 16)
    #         poses, activations = capsule_conv_layer(poses, activations, shape=[3, 1, 512, 512], strides=[1, 3, 1, 1],
    #                                                           iterations=1, name='conv2')
    #         print('poses_conv', poses.shape)  # (32, 97, 1, 16, 16)
    #         print('activations_conv', activations.shape)  # (32, 97, 1, 16)
    #         poses, activations = capsule_flatten(poses, activations)
    #         print('poses_flat', poses.shape)  # (32, 1552, 16)
    #         print('activations_flat', activations.shape)  # (32, 1552)
    #
    #         poses, activations = capsule_fc_layer(poses, activations, self.n_class, 1, 'fc2')
    #         print('poses ', poses.shape)  # (32, 202, 16)
    #         print('activations ', activations.shape)  # (32, 202)
    #
    #     return poses, activations


    def capsule_model_A(self, X):
        with tf.variable_scope('capsule_' + str(3)):
            print('capsule_' + str(3))
            print('X ', X)
            cnnout = _conv2d_wrapper(X, shape=[3, self.embedding_size, 1, 32], strides=[1, 2, 1, 1], padding='VALID',
                                     add_bias=True, activation_fn=tf.nn.relu, name='conv1')
            print('cnnout', cnnout.shape)
            tf.logging.info('output shape: {}'.format(cnnout.get_shape()))
            poses, activations = capsules_init(cnnout, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                                                         padding='VALID', pose_shape=16, add_bias=True,
                                               name='primary')
            print('poses_init', poses.shape)
            print('activations_init', activations.get_shape())
            poses, activations_conv = capsule_conv_layer(poses, activations, shape=[3, 1, 16, 16],
                                                         strides=[1, 1, 1, 1], iterations=3, name='conv2')
            print('poses_conv', poses.shape)
            print('activations_conv', activations_conv.shape)
            poses, activations = capsule_flatten(poses, activations_conv)
            print('poses_flatten', poses.shape)
            print('activations_flat', activations.shape)

            poses, activations = capsule_fc_layer(poses, activations, self.n_class, 3, 'fc2')
            print('poses ', poses.shape)
            print('activations ', activations.shape)

        return poses, activations


    def capsule_model_B(self, X):
        print('X.shape', X.shape)  # (25, 200, 300, 1)
        poses_list = []
        for _, ngram in enumerate([3]):
            with tf.variable_scope('capsule_' + str(ngram)):
                print('capsule_' + str(ngram))
                cnnout = _conv2d_wrapper(
                    X, shape=[ngram, self.fact_len, 1, 32], strides=[1, 3, 1, 1], padding='VALID',
                    add_bias=True, activation_fn=tf.nn.relu, name='conv1'
                )
                print('cnnout', cnnout.shape)  # (25, 99, 1, 32)

                tf.logging.info('output shape: {}'.format(cnnout.get_shape()))
                poses_init, activations_init = capsules_init(cnnout, shape=[1, 1, 32, 16], strides=[1, 2, 1, 1],
                                                             padding='VALID', pose_shape=16, add_bias=True, name='primary')
                print('poses_init', poses_init.shape)  # (25, 99, 1, 16, 16)
                print('activations_init', activations_init.get_shape())  # (25, 99, 1, 16)
                poses_conv, activations_conv = capsule_conv_layer(poses_init, activations_init, shape=[3, 1, 16, 16],
                                                                  strides=[1, 1, 1, 1], iterations=3, name='conv2')
                print('poses_conv', poses_conv.shape)  # (25, 97, 1, 16, 16)
                print('activations_conv', activations_conv.shape)  # (25, 97, 1, 16)
                poses_flat, activations_flat = capsule_flatten(poses_conv, activations_conv)
                print('capsule_flatten', poses_flat.shape)  # (25, 1552, 16)
                print('activations_flat', activations_flat.shape)  # (25, 1552)

                poses, activations = capsule_fc_layer(poses_flat, activations_flat, self.n_class, 3, 'fc2')
                print('poses ', poses.shape)  # (25, 9, 16)
                print('activations ', activations.shape)  # (25, 9)
                poses_list.append(poses)
        print('-------------------------------')
        poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0)
        print('poses ', poses.shape)
        activations = K.sqrt(K.sum(K.square(poses), 2))
        print('activations ', activations.shape)
        return poses, activations


    def _inference(self, X_inputs):
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        inputs = tf.expand_dims(inputs, -1)
        # poses, activations = self.capsule_model_A(inputs)  # done
        poses, activations = self.capsule_model_B(inputs)
        return poses, activations

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