# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

class Settings(object):
    def __init__(self):
        self.model_name = 'TextCNN_two_embedding'
        self.fact_len = 200
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 256
        self.stride_length = 3
        self.n_class = 202
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class Atten_TextCNN(object):
    def __init__(self, W_embedding, W_embedding2, settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size
        self.n_class = settings.n_class
        self.filter_sizes = settings.filter_sizes
        self.stride_length = settings.stride_length
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.fc_hidden_size = settings.fc_hidden_size
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
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
            self.embedding2 = tf.get_variable(name='embedding2', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding2), trainable=True)
        self.embedding_size = 100

        with tf.variable_scope('Atten_TextCNN'):
            output = self._inference(self._X_inputs)
            bs, w = output.get_shape().as_list()
            # exit(0)

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

    def conv_layers_return_2d_two_embedding(self, input_x, name_scope, reuse_flag=False):  # great 81.3

        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier
        input_x:[batch_size,sequence_length,embed_size,2]
        """
        # 1.=====>get emebedding of words in the sentence
        #sentence_embeddings_expanded = tf.expand_dims(input_x,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        print("going to start:conv_layers_return_2d_two_embedding. input_x:",input_x)
        sentence_embeddings_expanded=input_x
        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embedding_size, 2, self.n_filter],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1],padding="VALID",
                                    name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_train, scope='cnn1')
                print(i, "conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.n_filter])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                print('h_----------', h)
                bs, fl, eb, fs = h.get_shape().as_list()
                h = tf.reshape(h, [-1, fl, fs, 1])  # shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size,[filter_size, self.n_filter, 1, self.n_filter],
                                          initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="VALID",name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_train, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.n_filter])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                bs, fl, eb, fn = h.get_shape().as_list()
                # 3. Max-pooling
                pooling_max = tf.nn.max_pool(h, ksize=[1, fl, 1, 1],
                                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_max)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length - filter_size + 1,1,num_filters]
        # concat
        h = tf.concat(pooled_outputs, axis=3)  # [batch_size,num_total_filters]
        h = tf.reshape(h, shape=[-1, self.n_filter*len(self.filter_sizes)])
        print("h.concat:", h)

        # with tf.name_scope("dropout"):
        #     h = tf.nn.dropout(h,keep_prob=self.keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]

    def _inference(self, X_inputs):
        input_x1 = tf.nn.embedding_lookup(self.embedding, self.X_inputs) #[batch_size,total_sequence_length,embed_size]
        input_x2 = tf.nn.embedding_lookup(self.embedding2, self.X_inputs) #[batch_size,total_sequence_length,embed_size]
        input=tf.stack([input_x1,input_x2],axis=-1) #[batch_size,total_sequence_length,embed_size,2]
        #input_x = tf.layers.dense(input_x, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        inputs = self.conv_layers_return_2d_two_embedding(input, self.fact_len)
        # exit(0)
        return inputs


