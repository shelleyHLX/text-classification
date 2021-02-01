# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Settings(object):
    def __init__(self):
        self.model_name = 'Attention_TextCNN_ThreeTask'
        self.fact_len = 200
        self.filter_sizes = [2, 3, 4, 5, 7]
        self.n_filter = 256
        self.accu_fc_hidden_size = 1024
        self.law_fc_hidden_size = 1024
        self.death_fc_hidden_size = 256
        self.imp_fc_hidden_size = 256
        self.lif_fc_hidden_size = 256
        self.hidden_size = 256

        self.accu_class = 202
        self.law_class = 183
        self.death_class = 2
        self.imp_class = 1
        self.lif_class = 2

        self.accu_alpha = 0.3
        self.law_alpha = 0.3
        self.dea_alpha = 0.1
        self.imp_alpha = 0.2
        self.lif_alpha = 0.1

        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'

class Atten_TextCNN(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size

        self.accu_class = settings.accu_class
        self.law_class = settings.law_class
        self.death_class = settings.death_class
        self.imp_class = settings.imp_class
        self.lif_class = settings.lif_class
        #
        self.accu_alpha = settings.accu_alpha
        self.law_alpha = settings.law_alpha
        self.dea_alpha = settings.dea_alpha
        self.imp_alpha = settings.imp_alpha
        self.lif_alpha = settings.lif_alpha

        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.accu_fc_hidden_size = settings.accu_fc_hidden_size
        self.law_fc_hidden_size = settings.law_fc_hidden_size
        self.death_fc_hidden_size = settings.death_fc_hidden_size
        self.imp_fc_hidden_size = settings.imp_fc_hidden_size
        self.lif_fc_hidden_size = settings.lif_fc_hidden_size

        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            # acc, law, death, imp, lif
            self._X_inputs = tf.placeholder(tf.int32, [None, self.fact_len], name='X_input')
            self.acc = tf.placeholder(tf.float32, [None, self.accu_class], name='acc')
            self.law = tf.placeholder(tf.float32, [None, self.law_class], name='law')
            self.death = tf.placeholder(tf.float32, [None, self.death_class], name='death')
            self.imp = tf.placeholder(tf.float32, [None, self.imp_class], name='imp')
            self.lif = tf.placeholder(tf.float32, [None, self.lif_class], name='lif')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('Atten_TextCNN'):
            output = self._inference(self._X_inputs)

        # with tf.variable_scope('fc-bn-layer'):
        #     W_fc = self.weight_variable([self.n_filter_total, self.fc_hidden_size], name='Weight_fc')
        #     tf.summary.histogram('W_fc', W_fc)
        #     h_fc = tf.matmul(output, W_fc, name='h_fc')
        #     beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
        #     tf.summary.histogram('beta_fc', beta_fc)
        #     fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
        #     self.update_emas.append(update_ema_fc)
        #     self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

        with tf.variable_scope('accu_out_layer'):
            W_fc = self.weight_variable([self.n_filter_total, self.accu_fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.accu_fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

            W_out = self.weight_variable([self.accu_fc_hidden_size, self.accu_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.accu_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self.acc_prob = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')
            self.accu_pred =  tf.sigmoid(self.acc_prob) # 每个类别的分数 scores

        with tf.variable_scope('law_out_layer'):
            W_fc = self.weight_variable([self.n_filter_total, self.law_fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.law_fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

            W_out = self.weight_variable([self.law_fc_hidden_size, self.law_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.law_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self.law_prob = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')
            self.law_pred = tf.sigmoid(self.law_prob)

        with tf.variable_scope('death_out_layer'):
            W_fc = self.weight_variable([self.n_filter_total, self.death_fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.death_fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

            W_out = self.weight_variable([self.death_fc_hidden_size, self.death_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.death_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self.death_pred = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')

        with tf.variable_scope('imp_out_layer'):
            W_fc = self.weight_variable([self.n_filter_total, self.imp_fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.imp_fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

            W_out = self.weight_variable([self.imp_fc_hidden_size, self.imp_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.imp_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self.imp_pred = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')

        with tf.variable_scope('lif_out_layer'):
            W_fc = self.weight_variable([self.n_filter_total, self.lif_fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.lif_fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

            W_out = self.weight_variable([self.lif_fc_hidden_size, self.lif_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.lif_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self.lif_pred = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')
            print('self.lif_pred ', self.lif_pred.shape)

        with tf.name_scope('loss'):
            # classification
            self.accu_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.acc_prob, labels=self.acc))
            self.law_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.law_prob, labels=self.law))
            # imprisonment
            self.death_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.death_pred, labels=self.death))
            self.imp_loss = tf.reduce_mean(tf.square(self.imp_pred - self.imp))
            self.lif_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.lif_pred, labels=self.lif))
            # total loss
            self._loss = self.accu_alpha*self.accu_loss + self.law_alpha*self.law_loss + self.dea_alpha*self.death_loss + \
                         self.imp_alpha*self.imp_loss + self.lif_alpha*self.lif_loss

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

    def huber_loss(self, labels, predictions, delta=1.0):  # regression
        residual = tf.abs(predictions - labels)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.reduce_mean(tf.where(condition, small_res, large_res))

    def task_specific_attention(self, inputs, output_size,
                                initializer=layers.xavier_initializer(),
                                activation_fn=tf.tanh, scope=None):
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
        with tf.variable_scope(scope or 'attention') as scope:
            attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[output_size],
                                                       initializer=initializer, dtype=tf.float32)
            input_projection = layers.fully_connected(inputs, output_size,
                                                      activation_fn=activation_fn, scope=scope)
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector),
                                        axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            tf.summary.histogram('attention_weigths', attention_weights)
            weighted_projection = tf.multiply(inputs, attention_weights)
            return weighted_projection

    def TextCNN(self, X_inputs, n_step):
        inputs = tf.expand_dims(X_inputs, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filter]
                W_filter = self.weight_variable(shape=filter_shape, name='W_filter')
                beta = self.bias_variable(shape=[self.n_filter], name='beta_filter')
                tf.summary.histogram('beta', beta)
                conv = tf.nn.conv2d(inputs, W_filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv_bn, update_ema = self.batchnorm(conv, beta, convolutional=True)
                h = tf.nn.relu(conv_bn, name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, n_step - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                self.update_emas.append(update_ema)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total])
        return h_pool_flat

    def _inference(self, X_inputs):
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        output_att = self.task_specific_attention(inputs, self.embedding_size)
        textcnn_out = self.TextCNN(output_att, self.fact_len)
        return textcnn_out


