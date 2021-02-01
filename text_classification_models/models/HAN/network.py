# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn


class Settings(object):
    def __init__(self):
        self.model_name = 'HAN'
        # self.fact_len = 200
        self.doc_len = 16
        self.sent_len = 16
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 512
        # gru
        self.gru_hidden_size = 128
        self.n_layer = 1

        self.n_class = 202
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class Atten_TextCNN(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        # self.fact_len = settings.fact_len
        self.doc_len = settings.doc_len
        self.sent_len = settings.sent_len
        self.hidden_size = settings.hidden_size
        self.n_class = settings.n_class
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.gru_hidden_size = settings.gru_hidden_size
        self.n_layer = settings.n_layer
        self.fc_hidden_size = settings.fc_hidden_size

        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int32, [None, self.doc_len*self.sent_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('Atten_TextCNN'):
            output = self.han_inference(self._X_inputs)
            bs, w = output.get_shape().as_list()
            exit(0)

        with tf.variable_scope('fc-bn-layer'):
            W_fc = self.weight_variable([w, self.fc_hidden_size], name='Weight_fc')
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

    def gru_cell(self):
        with tf.name_scope('gru_cell'):
            cell = rnn.GRUCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_gru(self, inputs, seg_num):
        """build the bi-GRU network. Return the encoder represented vector.
        n_step: 句子的词数量；或者文档的句子数。
        seg_num: 序列的数量，原本应该为 batch_size, 但是这里将 batch_size 个 doc展开成很多个句子。
        """
        cells_fw = [self.gru_cell() for _ in range(self.n_layer)]
        cells_bw = [self.gru_cell() for _ in range(self.n_layer)]
        initial_states_fw = [cell_fw.zero_state(seg_num, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(seg_num, tf.float32) for cell_bw in cells_bw]
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                            initial_states_fw = initial_states_fw, initial_states_bw = initial_states_bw,
                                                            dtype=tf.float32)
        # outputs: Output Tensor shaped: seg_num, max_time, layers_output]，其中layers_output=hidden_size * 2 在这里。
        return outputs  # (?, 200, hidden_size * 2)

    def task_specific_attention(self, inputs, output_size,
                                initializer=layers.xavier_initializer(),
                                activation_fn=tf.tanh, scope=None):
        """
        Performs task-specific attention reduction, using learned
        attention context vector (constant within task of interest).
        Args:
            inputs: Tensor of shape [batch_size, units, input_size]
                `input_size` must be static (known)
                `units` axis will be attended over (reduced from output)
                `batch_size` will be preserved
            output_size: Size of output's inner (feature) dimension
        Returns:
           outputs: Tensor of shape [batch_size, output_dim].
        """
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
        with tf.variable_scope(scope or 'attention') as scope:
            # u_w, attention 向量
            attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[output_size],
                                                       initializer=initializer, dtype=tf.float32)
            # 全连接层，把 h_i 转为 u_i ， shape= [batch_size, units, input_size] -> [batch_size, units, output_size]
            input_projection = layers.fully_connected(inputs, output_size,
                                                      activation_fn=activation_fn,
                                                      scope=scope)
            # 输出 [batch_size, units]
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector),
                                        axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            tf.summary.histogram('attention_weigths', attention_weights)
            weighted_projection = tf.multiply(inputs, attention_weights)
            outputs = tf.reduce_sum(weighted_projection, axis=1)
            return outputs  # 输出 [batch_size, hidden_size*2]

    def han_inference(self, X_inputs):
        """分层 attention 模型
        Args:
            X_inputs: tensor.shape=(batch_size, doc_len*sent_len)
        Returns:
            doc_attn_outputs: tensor.shape=(batch_size, hidden_size(*2 for bigru))
        """
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        # inputs.shape=[batch_size, doc_len*sent_len, embedding_size]
        sent_inputs = tf.reshape(inputs,[self.batch_size*self.doc_len, self.sent_len, self.embedding_size])
        # shape=(?, 40, 256)
        with tf.variable_scope('sentence_encoder'):  # 生成句向量
            sent_outputs = self.bi_gru(sent_inputs, seg_num=self.batch_size*self.doc_len)
            sent_attn_outputs = self.task_specific_attention(sent_outputs, self.hidden_size*2)
            # [batch_size*doc_len, hidden_size*2]
            with tf.variable_scope('dropout'):
                sent_attn_outputs = tf.nn.dropout(sent_attn_outputs, self.keep_prob)
        with tf.variable_scope('doc_encoder'):      # 生成文档向量
            doc_inputs = tf.reshape(sent_attn_outputs, [self.batch_size, self.doc_len, self.hidden_size*2])
            doc_outputs = self.bi_gru(doc_inputs, self.batch_size)
            # [batch_size, doc_len, hidden_size*2]
            doc_attn_outputs = self.task_specific_attention(doc_outputs, self.hidden_size*2)
            # [batch_size, hidden_size*2]
        return doc_attn_outputs    # [batch_size, hidden_size*2]




