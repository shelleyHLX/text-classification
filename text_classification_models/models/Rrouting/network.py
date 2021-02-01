# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from models.routing.Capsule import Capusule

class Settings(object):
    def __init__(self):
        self.model_name = 'Rrout'
        self.fact_len = 16*16
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.num_filters = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 256
        self.kernel_size = 3
        self.seq_encoder = 'bigru'
        self.attn_mode = 'Rrout'
        self.dropout = 0.5
        self.out_caps_num = 3  #
        self.rout_iter = 2  # 3

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

        self.attn_mode = settings.attn_mode
        self.seq_encoder = settings.seq_encoder
        self.dropout = settings.dropout
        self.out_caps_num = settings.out_caps_num
        self.rout_iter = settings.rout_iter

        self.initializer = tf.contrib.layers.xavier_initializer()
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')

        with tf.name_scope('Inputs'):
            self._X_inputs = tf.placeholder(tf.int32, [None, 16, 16], name='X_input')
            self.sNum = tf.placeholder(shape=(None,), dtype=tf.int32, name='ph_sNum')
            # shape(b_sz, sNum) [[6, 3, 5, 3], [22, 44, 22],] 句子词数
            self.wNum = tf.placeholder(shape=(None, 16), dtype=tf.int32, name='ph_wNum')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.num_classes], name='y_input')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('Atten_TextCNN'):
            output = self._inference()
            print('output ', output)
            bs, w = output.get_shape().as_list()

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
            W_out = self.weight_variable([self.fc_hidden_size, self.num_classes], name='Weight_out')
            tf.summary.histogram('Weight_out', W_out)
            b_out = self.bias_variable([self.num_classes], name='bias_out')
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

    def biLSTM(self, in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):
        with tf.variable_scope(scope or 'biLSTM'):
            cell_fwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen, dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')
            x_out = tf.concat(x_out, axis=2)
            if dropout is not None:
                x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
        return x_out

    def biGRU(self, in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):

        with tf.variable_scope(scope or 'biGRU'):
            cell_fwd = tf.nn.rnn_cell.GRUCell(h_sz)
            cell_bwd = tf.nn.rnn_cell.GRUCell(h_sz)
            x_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fwd, cell_bwd, in_x, xLen, dtype=tf.float32, swap_memory=True,
                                                       scope='birnn')
            x_out = tf.concat(x_out, axis=2)
            if dropout is not None:
                x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
        return x_out

    def mkMask(self, input_tensor, maxLen):
        shape_of_input = tf.shape(input_tensor)
        shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

        oneDtensor = tf.reshape(input_tensor, shape=(-1,))
        flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
        return tf.reshape(flat_mask, shape_of_output)

    def masked_softmax(self, inp, seqLen):
        seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)
        if len(inp.get_shape()) != len(seqLen.get_shape())+1:
            raise ValueError('rank of seqLen should be %d, but have the rank %d.\n'
                             % (len(inp.get_shape())-1, len(seqLen.get_shape())))
        mask = self.mkMask(seqLen, tf.shape(inp)[-1])
        masked_inp = tf.where(mask, inp, tf.ones_like(inp) * (-np.Inf))
        ret = tf.nn.softmax(masked_inp)
        return ret

    def task_specific_attention(self, in_x, xLen, out_sz, activation_fn=tf.tanh,
                                dropout=None, is_train=False, scope=None):
        '''

        :param in_x: shape(b_sz, tstp, dim)
        :param xLen: shape(b_sz,)
        :param out_sz: scalar
        :param activation_fn: activation
        :param dropout:
        :param is_train:
        :param scope:
        :return:
        '''

        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None

        with tf.variable_scope(scope or 'attention') as scope:
            context_vector = tf.get_variable(name='context_vector', shape=[out_sz],
                                             dtype=tf.float32)
            in_x_mlp = tf.layers.dense(in_x, out_sz, activation=activation_fn, name='mlp')

            attn = tf.tensordot(in_x_mlp, context_vector, axes=[[2], [0]])  # shape(b_sz, tstp)
            attn_normed = self.masked_softmax(attn, xLen)

            attn_normed = tf.expand_dims(attn_normed, axis=-1)
            attn_ctx = tf.matmul(in_x_mlp, attn_normed, transpose_a=True)  # shape(b_sz, dim, 1)
            attn_ctx = tf.squeeze(attn_ctx, axis=[2])   # shape(b_sz, dim)
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx

    def routing_masked(self, in_x, xLen, out_sz, out_caps_num, iter=3,
                       dropout=None, is_train=False, scope=None):
        '''

        :param in_x: shape(b_sz, tstp, dim)
        :param xLen: shape(b_sz,)
        :param out_sz: scalar
        :param dropout:
        :param is_train:
        :param scope:
        :return:
        '''
        print('============ routing_masked'*2)
        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
        b_sz = tf.shape(in_x)[0]
        with tf.variable_scope(scope or 'routing'):
            attn_ctx = Capusule(out_caps_num, out_sz, iter)(in_x, xLen)   # shape(b_sz, out_caps_num, out_sz)
            print('attn_ctx', attn_ctx)  # (?, 5, 400)  (?, 5, 400)
            attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num*out_sz])
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx

    def reverse_routing_masked(self, in_x, xLen, out_sz, out_caps_num, iter=3,
                               dropout=None, is_train=False, scope=None):
        '''

        :param in_x: shape(b_sz, tstp, dim)
        :param xLen: shape(b_sz,)
        :param out_sz: scalar
        :param dropout:
        :param is_train:
        :param scope:
        :return:
        '''

        assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
        b_sz = tf.shape(in_x)[0]
        with tf.variable_scope(scope or 'routing'):
            '''shape(b_sz, out_caps_num, out_sz)'''
            attn_ctx = Capusule(out_caps_num, out_sz, iter)(in_x, xLen, reverse_routing=True)
            attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num * out_sz])
            if dropout is not None:
                attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
        return attn_ctx

    def hierachical_attention(self, in_x, sNum, wNum, scope=None):
        '''

        :param in_x: shape(b_sz, ststp, wtstp, emb_sz)
        :param sNum: shape(b_sz, )
        :param wNum: shape(b_sz, ststp)
        :param scope:
        :return:
        '''
        print('====hierachical_attention====')
        b_sz, ststp, wtstp, _ = tf.unstack(tf.shape(in_x))
        # print('b_sz, ststp, wtstp, _ ', b_sz, ststp, wtstp, _ )
        emb_sz = int(in_x.get_shape()[-1])
        with tf.variable_scope(scope or 'hierachical_attention'):
            flatten_in_x = tf.reshape(in_x, [b_sz*ststp, wtstp, emb_sz])
            print('flatten_in_x ', flatten_in_x)  # (?, ?, 300)
            flatten_wNum = tf.reshape(wNum, [b_sz * ststp])
            print('flatten_wNum ', flatten_wNum)  # (?,)
            flatten_attn_ctx = None
            with tf.variable_scope('sentence_enc'):
                if self.seq_encoder == 'bigru':
                    flatten_birnn_x = self.biGRU(flatten_in_x, flatten_wNum,
                                                 self.hidden_size, scope='biGRU')
                    print('flatten_birnn_x ', flatten_birnn_x)  # (?, ?, 400)
                elif self.seq_encoder == 'bilstm':
                    flatten_birnn_x = self.biLSTM(flatten_in_x, flatten_wNum,
                                                  self.hidden_size, scope='biLSTM')
                else:
                    raise ValueError('no such encoder %s'%self.seq_encoder)

                '''shape(b_sz*sNum, dim)'''
                # if self.attn_mode == 'avg':
                #     flatten_attn_ctx = reduce_avg(flatten_birnn_x, flatten_wNum, dim=1)

                if self.attn_mode == 'attn':
                    flatten_attn_ctx = self.task_specific_attention(flatten_birnn_x, flatten_wNum,
                                                                    int(flatten_birnn_x.get_shape()[-1]),
                                                                    dropout=self.dropout,
                                                                    is_train=self.is_train, scope='attention')
                    print('flatten_attn_ctx ', flatten_attn_ctx)  # (?, 400)
                elif self.attn_mode == 'rout':
                    print('====rout'*10)
                    print('flatten_birnn_x ', flatten_birnn_x)  # (?, ?, 400)
                    print('flatten_wNum ', flatten_wNum)  # (?,)
                    flatten_attn_ctx = self.routing_masked(flatten_birnn_x, flatten_wNum,
                                                           int(flatten_birnn_x.get_shape()[-1]),
                                                           self.out_caps_num, iter=self.rout_iter,
                                                           dropout=self.dropout,
                                                           is_train=self.is_train, scope='rout')
                    print('flatten_attn_ctx', flatten_attn_ctx)  # (?, 2000)
                elif self.attn_mode == 'Rrout':
                    flatten_attn_ctx = self.reverse_routing_masked(
                        flatten_birnn_x, flatten_wNum,  int(flatten_birnn_x.get_shape()[-1]), self.out_caps_num,
                        iter=self.rout_iter, dropout=self.dropout, is_train=self.is_train, scope='Rrout')
                else:
                    raise ValueError('no such attn mode %s' % self.attn_mode)
            snt_dim = int(flatten_attn_ctx.get_shape()[-1])
            print('snt_dim ', snt_dim)  # 2000
            snt_reps = tf.reshape(flatten_attn_ctx, shape=[b_sz, ststp, snt_dim])
            print('snt_reps ', snt_reps)  # (?, ?, 2000)

            with tf.variable_scope('doc_enc'):
                if self.seq_encoder == 'bigru':
                    birnn_snt = self.biGRU(snt_reps, sNum, self.hidden_size, scope='biGRU')
                    print('birnn_snt ', birnn_snt)  # (?, ?, 400)
                elif self.seq_encoder == 'bilstm':
                    birnn_snt = self.biLSTM(snt_reps, sNum, self.hidden_size, scope='biLSTM')
                else:
                    raise ValueError('no such encoder %s'%self.seq_encoder)

                '''shape(b_sz, dim)'''
                # if self.config.attn_mode == 'avg':
                #     doc_rep = reduce_avg(birnn_snt, sNum, dim=1)
                # elif self.config.attn_mode == 'max':
                #     doc_rep = tf.reduce_max(birnn_snt, axis=1)
                if self.attn_mode == 'attn':
                    doc_rep = self.task_specific_attention(birnn_snt, sNum,
                                                           int(birnn_snt.get_shape()[-1]),
                                                           dropout=self.dropout,
                                                           is_train=self.is_train, scope='attention')
                    print('doc_rep ', doc_rep)  # (?, 400)
                elif self.attn_mode == 'rout':
                    doc_rep = self.routing_masked(
                        birnn_snt, sNum, int(birnn_snt.get_shape()[-1]), self.out_caps_num, iter=self.rout_iter,
                        dropout=self.dropout, is_train=self.is_train, scope='attention')
                    print('doc_rep ', doc_rep)  # (?, 2000)
                elif self.attn_mode == 'Rrout':
                    doc_rep = self.reverse_routing_masked(birnn_snt, sNum,
                                                          int(birnn_snt.get_shape()[-1]),
                                                          self.out_caps_num,
                                                          iter=self.rout_iter,
                                                          dropout=self.dropout,
                                                          is_train=self.is_train, scope='attention')
                else:
                    raise ValueError('no such attn mode %s' % self.attn_mode)
                return doc_rep


    def _inference(self):
        inputs = tf.nn.embedding_lookup(self.embedding, self._X_inputs)
        output = self.hierachical_attention(inputs, self.sNum, self.wNum,)
        return output


