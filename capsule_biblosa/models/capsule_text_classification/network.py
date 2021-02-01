# coding: utf-8

from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from utils import _conv2d_wrapper
from layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer
import tensorflow.contrib.slim as slim


def baseline_model_cnn(X, num_classes):
    nets = _conv2d_wrapper(
        X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID', 
        add_bias=False, activation_fn=tf.nn.relu, name='conv1'
        )
    nets = slim.flatten(nets)
    tf.logging.info('flatten shape: {}'.format(nets.get_shape()))
    nets = slim.fully_connected(nets, 128, scope='relu_fc3', activation_fn=tf.nn.relu)
    tf.logging.info('fc shape: {}'.format(nets.get_shape()))

    activations = tf.sigmoid(slim.fully_connected(nets, num_classes, scope='final_layer', activation_fn=None))
    tf.logging.info('fc shape: {}'.format(activations.get_shape()))
    return tf.zeros([0]), activations


def baseline_model_kimcnn(X, max_sent, num_classes):
    pooled_outputs = []
    for i, filter_size in enumerate([3,4,5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):            
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")            
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")            
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = 100 * 3
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    activations = tf.sigmoid(slim.fully_connected(h_pool_flat, num_classes, scope='final_layer'))
    return tf.zeros([0]), activations


def capsule_model_B(X, num_classes):
    print('X.shape', X.shape)  # (25, 200, 300, 1)
    print('num_classes: ', num_classes.shape)
    poses_list = []
    for _, ngram in enumerate([3, 4, 5]):
        with tf.variable_scope('capsule_' + str(ngram)):
            print('capsule_' + str(ngram))
            cnnout = _conv2d_wrapper(
                X, shape=[ngram, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID', 
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
            print('cnnout', cnnout.shape)  # (25, 99, 1, 32)

            tf.logging.info('output shape: {}'.format(cnnout.get_shape()))
            poses_init, activations_init = capsules_init(cnnout, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                                 padding='VALID', pose_shape=16, add_bias=True, name='primary')
            print('poses_init', poses_init.shape)  # (25, 99, 1, 16, 16)
            print('activations_init', activations_init.get_shape())  # (25, 99, 1, 16)
            poses_conv, activations_conv = capsule_conv_layer(poses_init, activations_init, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1],
                                      iterations=3, name='conv2')
            print('poses_conv', poses_conv.shape)  # (25, 97, 1, 16, 16)
            print('activations_conv', activations_conv.shape)  # (25, 97, 1, 16)
            poses_flat, activations_flat = capsule_flatten(poses_conv, activations_conv)
            print('capsule_flatten', poses_flat.shape)  # (25, 1552, 16)
            print('activations_flat', activations_flat.shape)  # (25, 1552)

            poses, activations = capsule_fc_layer(poses_flat, activations_flat, num_classes, 3, 'fc2')
            print('poses ', poses.shape)  # (25, 9, 16)
            print('activations ', activations.shape)  # (25, 9)
            poses_list.append(poses)
    print('-------------------------------')
    poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0)
    print('poses ', poses.shape)
    activations = K.sqrt(K.sum(K.square(poses), 2))
    print('activations ', activations.shape)
    return poses, activations


def capsule_model_A(X, num_classes):
    with tf.variable_scope('capsule_' + str(3)):
        print('capsule_' + str(3))
        print('X ', X)
        cnnout = _conv2d_wrapper(X, shape=[3, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID',
                add_bias=True, activation_fn=tf.nn.relu, name='conv1')
        print('cnnout', cnnout.shape)
        tf.logging.info('output shape: {}'.format(cnnout.get_shape()))
        poses_init, activations_init = capsules_init(cnnout, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')
        print('poses_init', poses_init.shape)
        print('activations_init', activations_init.get_shape())
        poses_conv, activations_conv = capsule_conv_layer(poses_init, activations_init, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1],
                                                          iterations=3, name='conv2')
        print('poses_conv', poses_conv.shape)
        print('activations_conv', activations_conv.shape)
        poses_flat, activations_flat = capsule_flatten(poses_conv, activations_conv)
        print('capsule_flatten', poses_flat.shape)
        print('activations_flat', activations_flat.shape)

        poses, activations = capsule_fc_layer(poses_flat, activations_flat, num_classes, 3, 'fc2')
        print('poses ', poses.shape)
        print('activations ', activations.shape)

    return poses, activations

"""
train.shape (4926, 300)
train_label.shape (4926,)
test.shape (335, 300)
test_label.shape (335, 9)
dev.shape (548, 300)
dev_label.shape (548,)
w2v.shape (21055, 300)
<class 'list'>
max sent:  300
vocab size:  21055
vec size:  300
num_classes:  9
X.shape (25, 200, 300, 1)
num_classes:  ()
capsule_3

cnnout (25, 99, 1, 32)
poses_init (25, 99, 1, 16, 16)
activations_init (25, 99, 1, 16)

poses_conv (25, 97, 1, 16, 16)
activations_conv (25, 97, 1, 16)

capsule_flatten (25, 1552, 16)
activations_flat (25, 1552)

poses  (25, 9, 16)
activations  (25, 9)

capsule_4
cnnout (25, 99, 1, 32)
poses_init (25, 99, 1, 16, 16)
activations_init (25, 99, 1, 16)
poses_conv (25, 97, 1, 16, 16)
activations_conv (25, 97, 1, 16)
capsule_flatten (25, 1552, 16)
activations_flat (25, 1552)
poses  (25, 9, 16)
activations  (25, 9)
capsule_5
cnnout (25, 98, 1, 32)
poses_init (25, 98, 1, 16, 16)
activations_init (25, 98, 1, 16)
poses_conv (25, 96, 1, 16, 16)
activations_conv (25, 96, 1, 16)
capsule_flatten (25, 1536, 16)
activations_flat (25, 1536)
poses  (25, 9, 16)
activations  (25, 9)

poses  (25, 9, 16)
activations  (25, 9)
"""