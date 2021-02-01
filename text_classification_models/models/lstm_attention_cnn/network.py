# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import numpy as np

class Settings(object):
    def __init__(self):
        self.model_name = 'lstm_Attention_TextCNN'
        self.fact_len = 200
        self.filter_sizes = [2, 3, 4, 5, 7]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 256
        self.n_layer = 2
        self.n_class = 202
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class Atten_TextCNN(object):
    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size
        self.n_class = settings.n_class
        self.n_layer = settings.n_layer
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.fc_hidden_size = settings.fc_hidden_size

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

        with tf.name_scope('loss'):
            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))
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
    # ************************** RNN ****************************
    def gru_cell(self, hidden_size):
        print('============= gru_cell =============')
        with tf.name_scope('gru_cell'):
            cell = rnn.GRUCell(num_units=hidden_size, reuse=tf.get_variable_scope().reuse)
            # rnn.MultiRNNCell, rnn.LSTMCell, rnn.GRUCell, rnn.RNNCell, rnn.GRUCell
            # print('type(cell): ', type(cell))
        return rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)

    def bi_gru(self, inputs):
        print('================== bi_gru ==================')
        """build the Bi-GRU network. 返回个所有层的隐含状态。"""
        cells_fw = [self.gru_cell(128) for _ in range(self.n_layer)]
        cells_bw = [self.gru_cell(128) for _ in range(self.n_layer)]
        print('inputs: ', inputs.shape)  # (?, 200, 256)
        # print('cells_fw: ', type(cells_fw))
        # print('cells_bw: ', type(cells_bw))
        # print('celss_fw: ', np.asarray(cells_fw).shape)
        initial_states_fw = [cell_fw.zero_state(self.batch_size, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(self.batch_size, tf.float32) for cell_bw in cells_bw]
        # print('initial_states_bw: ', np.asarray(initial_states_bw).shape)
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                            initial_states_fw=initial_states_fw,
                                                            initial_states_bw=initial_states_bw,
                                                            dtype=tf.float32)
        print('outputs', outputs.shape)  # (batch_size, 200, 256)
        outputs = tf.concat(outputs, axis=2)
        print('outputs: ', outputs.shape)  # (?, 200, self.hidden_size * 2),定义cell时的size;  2*cell_size=embedding size
        return outputs

    # LSTM
    def BiLSTM(self, X, hidden_size = 256):
        cells_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        cells_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, X, dtype=tf.float32)
        outputs = tf.concat(outputs, 2)
        return outputs

    def BiLSTM_Multilayer(self, X):
        print('BiLSTM_Multilayer')
        print('X', X.shape)
        cells_fws = [tf.contrib.rnn.LSTMCell(256) for _ in range(3)]
        cells_fws_wrap = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(cells_fws),
                                                       output_keep_prob=self.keep_prob)
        cells_bws = [tf.contrib.rnn.LSTMCell(256) for _ in range(3)]
        cells_bws_warp = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(cells_bws),
                                                       output_keep_prob=self.keep_prob)
        inputs = tf.transpose(X, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, 256])
        inputs = tf.split(inputs, 200, 0)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cells_fws_wrap, cells_bws_warp, inputs, dtype=tf.float32)
        # print('outputs ', outputs.shape)
        outputs = tf.convert_to_tensor(outputs)  # (200, batch_size, 512)
        outputs = tf.transpose(outputs, [1, 0, 2])
        print('outputs ', outputs.shape)  # (batch_size, 200, 512)
        print('BiLSTM_Multilayer')
        return outputs


    def lstm_me(self, inputs):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.embedding_size, forget_bias=0.0)
        # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.LSTMCell, tf.nn.rnn_cell.GRUCell, tf.nn.rnn_cell.MultiRNNCell
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        num_layers = 2
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(num_layers)])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        outputs_lstm, _ = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state)
        print('outputs: ', outputs_lstm.shape)  # 定义cell时的size (32, 500, 100) batch_size, words, self.size
        return outputs_lstm

    # **************************** RNN *****************************

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
        output_lstm = self.bi_gru(inputs)
        output_att = self.task_specific_attention(output_lstm, self.embedding_size)
        textcnn_out = self.TextCNN(output_att, self.fact_len)
        return textcnn_out


