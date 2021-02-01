from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.layers.python.layers import initializers

import tensorflow as tf

slim = tf.contrib.slim

epsilon = 1e-9

def _matmul_broadcast(x, y, name):
    """
    Compute x @ y, broadcasting over the first `N - 2` ranks.
    """
    with tf.variable_scope(name) as scope:
        return tf.reduce_sum(tf.nn.dropout(x[..., tf.newaxis] * y[..., tf.newaxis, :, :],1), axis=-2)


def _get_variable_wrapper(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None,
    caching_device=None, partitioner=None, validate_shape=True, custom_getter=None):
    """
    Wrapper over tf.get_variable().
    """

    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=trainable,
        collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape,
        custom_getter=custom_getter)
    return var


def _get_weights_wrapper( name, shape, dtype=tf.float32, initializer=initializers.xavier_initializer(), weights_decay_factor=None):
  """
  Wrapper over _get_variable_wrapper() to get weights, with weights decay factor in loss.
  """

  weights = _get_variable_wrapper(name=name, shape=shape, dtype=dtype, initializer=initializer)

  if weights_decay_factor is not None and weights_decay_factor > 0.0:
    weights_wd = tf.multiply(tf.nn.l2_loss(weights), weights_decay_factor, name=name + '/l2loss')
    tf.add_to_collection('losses', weights_wd)

  return weights


def _get_biases_wrapper(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0)):
  """Wrapper over _get_variable_wrapper() to get bias.
  """

  biases = _get_variable_wrapper(
    name=name, shape=shape, dtype=dtype, initializer=initializer
  )

  return biases

def batchnorm(Ylogits, offset, convolutional=False):
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


def _conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name, stddev=0.1):
  """
  Wrapper over tf.nn.conv2d().
  """

  with tf.variable_scope(name) as scope:
    kernel = _get_weights_wrapper(name='weights', shape=shape, weights_decay_factor=0.0,
                                  initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    output = tf.nn.conv2d(inputs, filter=kernel, strides=strides, padding=padding, name='conv')
    bnepsilon = 1e-5
    if add_bias:
      print('add_bias')
      biases = _get_biases_wrapper(name='biases', shape=[shape[-1]])
      mean, variance = tf.nn.moments(output, [0, 1, 2])
      output = tf.nn.batch_normalization(output, mean=mean, variance=variance, offset=biases,
                                                      scale=None, variance_epsilon=bnepsilon)
    if activation_fn is not None:
      output = activation_fn(output, name='activation')

  return output


def _separable_conv2d_wrapper(inputs, depthwise_shape, pointwise_shape, strides, padding, add_bias, activation_fn, name):
  """
  Wrapper over tf.nn.separable_conv2d().
  """
  
  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(name='depthwise_weights', shape=depthwise_shape, weights_decay_factor=0.0)
    pkernel = _get_weights_wrapper(name='pointwise_weights', shape=pointwise_shape, weights_decay_factor=0.0)
    output = tf.nn.separable_conv2d(input=inputs, depthwise_filter=dkernel, pointwise_filter=pkernel,
      strides=strides, padding=padding, name='conv')
    if add_bias:
      biases = _get_biases_wrapper(name='biases', shape=[pointwise_shape[-1]])
      output = tf.add(output, biases, name='biasAdd')
    if activation_fn is not None:
      output = activation_fn(output, name='activation')

  return output


def _depthwise_conv2d_wrapper(inputs, shape, strides, padding, add_bias, activation_fn, name):
  """
  Wrapper over tf.nn.depthwise_conv2d().
  """

  with tf.variable_scope(name) as scope:
    dkernel = _get_weights_wrapper(name='depthwise_weights', shape=shape, weights_decay_factor=0.0)
    output = tf.nn.depthwise_conv2d(inputs, filter=dkernel, strides=strides, padding=padding, name='conv')
    if add_bias:
      d_ = output.get_shape()[-1].value
      biases = _get_biases_wrapper(name='biases', shape=[d_])
      output = tf.add(output, biases, name='biasAdd')
    if activation_fn is not None:
      output = activation_fn(output, name='activation')

    return output
