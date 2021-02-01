# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Settings(object):
    def __init__(self):
        self.model_name = 'DPCNN'
        self.fact_len = 200
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.num_filters = 512
        self.fc_hidden_size = 1024
        self.hidden_size = 256
        self.kernel_size = 3
        self.num_classes = 202
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class Atten_TextCNN(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size
        self.num_classes = settings.num_classes
        self.filter_sizes = settings.filter_sizes
        self.kernel_size = settings.kernel_size
        self.num_filters = settings.num_filters
        self.n_filter_total = self.num_filters * len(self.filter_sizes)
        self.fc_hidden_size = settings.fc_hidden_size

        self.initializer = tf.contrib.layers.xavier_initializer()
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int32, [None, self.fact_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.num_classes], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('Atten_TextCNN'):
            self.inistiante_weight()
            output = self._inference()
            h, w = output.get_shape().as_list()
            print('output.get_shape()', output.get_shape())

        # with tf.variable_scope('fc-bn-layer'):
        #     # exit(0)
        #     W_fc = self.weight_variable([self.n_filter_total, self.fc_hidden_size], name='Weight_fc')
        #     tf.summary.histogram('W_fc', W_fc)
        #     h_fc = tf.matmul(output, W_fc, name='h_fc')
        #     beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
        #     tf.summary.histogram('beta_fc', beta_fc)
        #     fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
        #     self.update_emas.append(update_ema_fc)
        #     self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

        with tf.variable_scope('out_layer'):
            W_out = self.weight_variable([w, self.num_classes], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.num_classes], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self.logits = tf.nn.xw_plus_b(output, W_out, b_out, name='y_pred')  # 每个类别的分数 scores
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

    def inistiante_weight(self):
        with tf.name_scope('weights'):
            self.region_w = tf.get_variable("W_region", [self.kernel_size, self.embedding_size, 1,
                                                         self.num_filters], initializer=self.initializer,
                                            dtype=tf.float32)
            self.w_projection = tf.get_variable("W_projection", [self.num_filters, self.num_classes],
                                                initializer=self.initializer, dtype=tf.float32)
            self.b_projection = tf.get_variable('b_projection', [self.num_classes], initializer=self.initializer,
                                                dtype=tf.float32)

    def conv3(self, k, input_):
        conv3_w = tf.get_variable("W_conv%s" % k,
                                  [self.kernel_size, 1, self.num_filters, self.num_filters],
                                  initializer=self.initializer, dtype=tf.float32)
        conv = tf.nn.conv2d(input_, conv3_w, strides=[1, 1, 1, 1], padding='SAME')
        return conv

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self._X_inputs, name='look_up')
        self.embedded_words = tf.expand_dims(self.embedded_words, axis=-1)  # [None,seq,embedding,1]

        regoin_embedding = tf.nn.conv2d(self.embedded_words, self.region_w, strides=[1, 1, 1, 1],
                                        padding='VALID')  # [batch,seq-3+1,1,250]

        pre_activation = tf.nn.relu(regoin_embedding, name='preactivation')

        conv3 = self.conv3(0, pre_activation)  # [batch,seq-3+1,1,250]
        # batch norm
        conv3 = tf.layers.batch_normalization(conv3)

        conv3_pre_activation = tf.nn.relu(conv3, name='preactivation')
        conv3 = self.conv3(1, conv3_pre_activation)  # [batch,seq-3+1,1,250]
        # batch norm
        conv3 = tf.layers.batch_normalization(conv3)

        conv3 = conv3 + regoin_embedding  # [batch,seq-3+1,1,250]
        k = 1
        # print('conv3',conv3.get_shape().as_list())
        while conv3.get_shape().as_list()[1] >= 2:
            conv3, k = self._block(conv3, k)

        conv3 = tf.squeeze(conv3, [1, 2])  # [batch,250]
        print('conv3 ==>', conv3)
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        return conv3

    def _block(self, x, k):
        x = tf.pad(x, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])

        px = tf.nn.max_pool(x, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
        # conv
        k += 1
        x = tf.nn.relu(px)
        x = self.conv3(k, x)
        x = tf.layers.batch_normalization(x)

        # conv
        k += 1
        x = tf.nn.relu(x)
        x = self.conv3(k, x)
        x = tf.layers.batch_normalization(x)
        x = x + px
        return x, k

    def _inference(self):
        output = self.inference()
        return output


