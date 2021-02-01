# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood
import keras.backend as K


class Settings(object):
    def __init__(self):
        self.model_name = 'Attention_TextCNN_loss_para'
        self.fact_len = 200
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.n_filter = 256
        self.fc_hidden_size = 2000
        self.hidden_size = 256
        self.n_class = 202
        self.loss_type = 'focal_loss'
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class Atten_TextCNN(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size
        self.n_class = settings.n_class
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.fc_hidden_size = settings.fc_hidden_size
        self.loss_type = settings.loss_type
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int32, [None, self.fact_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('Atten_TextCNN'):
            output = self._inference(self._X_inputs)

        with tf.variable_scope('fc-bn-layer'):
            W_fc = self.weight_variable([self.n_filter_total, self.fc_hidden_size], name='Weight_fc')
            tf.summary.histogram('W_fc', W_fc)
            h_fc = tf.matmul(output, W_fc, name='h_fc')
            beta_fc = tf.Variable(tf.constant(0.1, tf.float32, shape=[self.fc_hidden_size], name="beta_fc"))
            tf.summary.histogram('beta_fc', beta_fc)
            fc_bn, update_ema_fc = self.batchnorm(h_fc, beta_fc, convolutional=False)
            self.update_emas.append(update_ema_fc)
            self.fc_bn_relu = tf.nn.relu(fc_bn, name="relu")

        with tf.variable_scope('out_layer'):
            W_out = self.weight_variable([self.fc_hidden_size, self.n_class], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.n_class], name='bias_out')
            tf.summary.histogram('bias_out', b_out)
            self._y_pred = tf.nn.xw_plus_b(self.fc_bn_relu, W_out, b_out, name='y_pred')  # 每个类别的分数 scores
            self.labels = tf.sigmoid(self._y_pred)

        # with tf.name_scope('loss'):
        #     self._loss = tf.reduce_mean(
        #         tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))
        #     tf.summary.scalar('loss', self._loss)

        with tf.name_scope('loss'):
            self._loss = self.focal_loss(self.y_inputs, self.y_pred)

        self.saver = tf.train.Saver(max_to_keep=1)


    def focal_loss(self, labels, logits, gamma=2.0, alpha=0.25, normalize=True):
        labels = tf.where(labels > 0, tf.ones_like(labels), tf.zeros_like(labels))
        labels = tf.cast(labels, tf.float32)
        probs = tf.sigmoid(logits)
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        alpha_t = tf.ones_like(logits) * alpha
        alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
        probs_t = tf.where(labels > 0, probs, 1.0 - probs)
        # tf.where(input, a,b)，其中a，b均为尺寸一致的tensor，作用是将a中对应input中true的位置的元素值不变，其余元素进行替换，替换成b中对应位置的元素值
        focal_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)
        loss = focal_matrix * ce_loss

        loss = tf.reduce_sum(loss)
        if normalize:
            n_pos = tf.reduce_sum(labels)
            # total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
            # total_weights = tf.Print(total_weights, [n_pos, total_weights])
            #         loss = loss / total_weights
            def has_pos():
                return loss / tf.cast(n_pos, tf.float32)
            def no_pos():
                #total_weights = tf.stop_gradient(tf.reduce_sum(focal_matrix))
                #return loss / total_weights
                return loss
            loss = tf.cond(n_pos > 0, has_pos, no_pos)
        return loss


    def loss_func(self, logits, labels):
        # logits = logits[:, 0, 0, 0]
        # labels = tf.equal(labels, 1)
        labels = tf.cast(labels, tf.float32)

        if self.loss_type == 'focal_loss':
            loss = self.focal_loss(labels=labels, logits=logits,
                              gamma=2.0, alpha=0.25)
        elif self.loss_type == 'ce_loss':
            ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = labels, logits = logits)
            num_samples = tf.cast(tf.reduce_prod(tf.shape(labels)), tf.float32)
            loss = tf.reduce_sum(ce_loss) / num_samples
        elif self.loss_type == 'cls_balance':
            ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = labels, logits = logits)
            pos_weight = tf.cast(tf.equal(labels, 1), tf.float32)
            neg_weight = 1 - pos_weight

            n_pos = tf.reduce_sum(pos_weight)
            n_neg = tf.reduce_sum(neg_weight)

            def has_pos():
                return tf.reduce_sum(ce_loss * pos_weight) / n_pos
            def has_neg():
                return tf.reduce_sum(ce_loss * neg_weight) / n_neg
            def no():
                return tf.constant(0.0)
            pos_loss = tf.cond(n_pos > 0, has_pos, no)
            neg_loss = tf.cond(n_neg > 0, has_neg, no)
            loss = (pos_loss + neg_loss) / 2.0

        # elif self.loss_type == 'ohnm':
        #     ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #         labels = labels, logits = logits)
        #     pos_weight = tf.cast(tf.equal(labels, 1), tf.float32)
        #     neg_weight = 1 - pos_weight
        #     n_pos = tf.reduce_sum(pos_weight)
        #     n_neg = tf.reduce_sum(neg_weight)
        #
        #
        #     # find the most wrongly classified negative examples:
        #     n_selected = tf.minimum(n_pos * 3, n_neg)
        #     n_selected = tf.cast(tf.maximum(n_selected, 1), tf.int32)
        #
        #     neg_mask = tf.equal(labels, 0)
        #     hardness = tf.where(neg_mask, ce_loss, tf.zeros_like(ce_loss))
        #     vals, _ = tf.nn.top_k(neg_scores, k = n_selected)
        #     th = vals[-1]
        #     selected_neg_mask = tf.logical_and(hardness >= th, neg_mask)
        #     neg_weight = tf.cast(selected_neg_mask, tf.float32)
        #
        #     loss_weight = pos_weight + neg_weight
        #     loss = tf.reduce_sum(ce_loss * loss_weight) / tf.reduce_sum(loss_weight)
        elif self.loss_type == 'ohem':
            ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = labels, logits = logits)
            # find the most wrongly classified examples:
            num_examples = tf.reduce_prod(labels.shape)
            n_selected = tf.cast(num_examples / 2, tf.int32)
            vals, _ = tf.nn.top_k(ce_loss, k = n_selected)
            th = vals[-1]
            selected_mask = ce_loss >= th
            loss_weight = tf.cast(selected_mask, tf.float32)
            loss = tf.reduce_sum(ce_loss * loss_weight) / tf.reduce_sum(loss_weight)
        else:
            raise ValueError('Unknow loss_type:', self.loss_type)
        return loss


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


