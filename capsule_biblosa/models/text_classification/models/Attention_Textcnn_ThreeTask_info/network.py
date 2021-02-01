# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Settings(object):
    def __init__(self):
        self.model_name = 'TextCNN_info_loss'
        self.fact_len = 200
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.n_filter = 256
        self.accu_fc_hidden_size = 1024
        self.article_fc_hidden_size = 1024
        self.death_fc_hidden_size = 256
        self.imp_fc_hidden_size = 256
        self.lif_fc_hidden_size = 256
        self.hidden_size = 256

        self.accu_class = 202
        self.article_class = 183
        self.death_class = 2
        self.imp_class = 1
        self.lif_class = 2

        self.accu_alpha = 0.3
        self.article_alpha = 0.3
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
        self.article_class = settings.article_class
        self.death_class = settings.death_class
        self.imp_class = settings.imp_class
        self.lif_class = settings.lif_class
        #
        self.accu_alpha = settings.accu_alpha
        self.article_alpha = settings.article_alpha
        self.dea_alpha = settings.dea_alpha
        self.imp_alpha = settings.imp_alpha
        self.lif_alpha = settings.lif_alpha

        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.accu_fc_hidden_size = settings.accu_fc_hidden_size
        self.article_fc_hidden_size = settings.article_fc_hidden_size
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
            self.acc_y = tf.placeholder(tf.float32, [None, self.accu_class], name='acc')
            self.article_y = tf.placeholder(tf.float32, [None, self.article_class], name='law')
            self.death_y = tf.placeholder(tf.float32, [None, self.death_class], name='death')
            self.imp_y = tf.placeholder(tf.float32, [None, self.imp_class], name='imp')
            self.lif_y = tf.placeholder(tf.float32, [None, self.lif_class], name='lif')

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('Atten_TextCNN'):
            output = self._inference(self._X_inputs)

        self.logits_accusation,self.logits_article,self.logits_deathpenalty,self.logits_lifeimprisonment,self.logits_imprisonment = \
            self.project_tasks(output)

        self.accu_pred = tf.nn.sigmoid(self.logits_accusation)
        self.law_pred = tf.nn.sigmoid(self.logits_article)
        self.death_pred = tf.nn.sigmoid(self.logits_deathpenalty)
        self.lif_pred = tf.nn.sigmoid(self.logits_lifeimprisonment)
        self.imp_pred = self.logits_imprisonment

        self._loss = self.loss_fun()
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

    def loss_fun(self,l2_lambda=0.0001*3, epislon=0.000001):
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))

        # loss1: accusation
        # input_y_accusation_onehot=tf.one_hot(self.input_y_accusation,self.accusation_num_classes)
        losses_accusation= tf.nn.sigmoid_cross_entropy_with_logits(labels=self.acc_y,logits=self.logits_accusation)
        # [batch_size,num_classes]
        self.loss_accusation = tf.reduce_mean((tf.reduce_sum(losses_accusation,axis=1)))
        # shape=(?,)-->(). loss for all data in the batch-->single loss

        # loss2:relevant article
        # input_y_article_onehot=tf.one_hot(self.input_y_article,self.article_num_classes)
        losses_article= tf.nn.sigmoid_cross_entropy_with_logits(labels=self.article_y,logits=self.logits_article)
        # [batch_size,num_classes]
        self.loss_article = tf.reduce_mean((tf.reduce_sum(losses_article, axis=1)))
        # shape=(?,)-->(). loss for all data in the batch-->single loss

        # loss3:death penalty
        losses_deathpenalty = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.death_y,logits=self.logits_deathpenalty)
        self.loss_deathpenalty = tf.reduce_mean((tf.reduce_sum(losses_deathpenalty, axis=1)))
        # shape=(?,)-->(). loss for all data in the batch-->single loss

        # loss4:life imprisonment
        losses_lifeimprisonment = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.lif_y,logits=self.logits_lifeimprisonment)
        self.loss_lifeimprisonment = tf.reduce_mean((tf.reduce_sum(losses_lifeimprisonment, axis=1)))
        # shape=(?,)-->(). loss for all data in the batch-->single loss

        # loss5: imprisonment: how many year in prison.
        self.loss_imprisonment =tf.reduce_mean(tf.divide(tf.pow((self.logits_imprisonment-self.imp_y),2),1000.0)) #1000.0TODO
        # self.loss_imprisonment = self.huber_loss(labels=self.imp_y, predictions=self.logits_imprisonment)
        # self.loss_imprisonment = self.mse_loss(labels=self.imp_y, predictions=self.logits_imprisonment)
        # print("sigmoid_cross_entropy_with_logits.losses:", losses_accusation)  # shape=(?, 1999).
        # self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

        """0.25, 0.25, 0.15, 0.15, 0.2"""
        self.weights_accusation = 0.3  # tf.nn.sigmoid(tf.cast(self.global_step / 1000.0, dtype=tf.float32)) / 3.0    # 0--1/3
        self.weights_article = 0.3  # tf.nn.sigmoid(tf.cast(self.global_step / 1000.0, dtype=tf.float32)) / 3.0       # 0--1/3
        self.weights_deathpenalty = 0.05  # tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 9.0   #0--1/9
        self.weights_lifeimprisonment = 0.05  # tf.nn.sigmoid(tf.cast(self.global_step / 1000.0, dtype=tf.float32)) / 9.0 #0--1/9
        self.weights_imprisonment=0.3  # 1-(self.weights_accusation+self.weights_article+self.weights_deathpenalty+self.weights_lifeimprisonment) #0-1/9
        loss = self.weights_accusation*self.loss_accusation+self.weights_article*self.loss_article+self.weights_deathpenalty*self.loss_deathpenalty + \
               self.weights_lifeimprisonment*self.loss_lifeimprisonment+self.weights_imprisonment*self.loss_imprisonment  # +self.l2_loss
        # loss=self.loss_accusation
        return loss

    def project_tasks(self,h):
        """
        :param h: shared features
        :return: logits
        transoform each sub task using one-layer MLP ,then get logits.
        get some insights from densely connected layers from recently development
        """
        # 1.accusation: FC-->dropout-->classifier
        h_accusation = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        # h_accusation = tf.nn.dropout(h_accusation,keep_prob=self.keep_prob) # TODO ADD 2018.07.02
        logits_accusation = tf.layers.dense(h_accusation, self.accu_class,use_bias=True)  # shape:[None,self.num_classes]

        # 2.relevant article: concated features-->FC-->dropout-->classifier
        h_article_concated=tf.concat([h,h_accusation],axis=-1) #TODO [batch,?,hidden_size*2] ADD 2018.07.02
        h_article = tf.layers.dense(h_article_concated, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        # h_article = tf.nn.dropout(h_article,keep_prob=self.keep_prob) # TODO ADD 2018.07.02
        logits_article = tf.layers.dense(h_article, self.article_class,use_bias=True)  # shape:[None,self.num_classes]

        # 3.death penalty: concated features-->FC-->dropout-->classifier
        # h_deathpenalty_concated=tf.concat([h], axis=-1)  # TODO [batch,?,hidden_size*3] ADD 2018.07.02
        # h_deathpenalty_concated=tf.concat([h], axis=-1)
        h_deathpenalty = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        # h_deathpenalty = tf.nn.dropout(h_deathpenalty,keep_prob=self.keep_prob)  # TODO ADD 2018.07.02
        logits_deathpenalty = tf.layers.dense(h_deathpenalty,self.death_class,use_bias=True)
        # shape:[None,self.num_classes] #

        #4.life imprisonment: concated features-->FC-->dropout-->classifier
        # h_lifeimprsion_concated=tf.concat([h],axis=-1)
        h_lifeimprisonment = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        # h_lifeimprisonment = tf.nn.dropout(h_lifeimprisonment,keep_prob=self.keep_prob) # TODO ADD 2018.07.02
        logits_lifeimprisonment = tf.layers.dense(h_lifeimprisonment, self.lif_class,use_bias=True)  # shape:[None,self.num_classes]

        #5.imprisonment: concated features-->FC-->dropout-->classifier
        # h_imprisonment_concated=tf.concat([h,h_accusation,h_article,h_deathpenalty,h_lifeimprisonment],axis=-1)
        h_imprisonment_concated=tf.concat([h,h_deathpenalty,h_lifeimprisonment],axis=-1)
        logits_imprisonment = tf.layers.dense(h_imprisonment_concated, self.hidden_size, activation=tf.nn.relu, use_bias=True)
        logits_imprisonment = tf.nn.dropout(logits_imprisonment,keep_prob=self.keep_prob) # TODO ADD 2018.07.02
        logits_imprisonment = tf.layers.dense(logits_imprisonment, self.imp_class,use_bias=True)  # imprisonment is a continuous value, no need to use activation function
        logits_imprisonment = tf.reshape(logits_imprisonment, [-1]) #[batch_size]
        return logits_accusation, logits_article, logits_deathpenalty, logits_lifeimprisonment, logits_imprisonment


    def huber_loss(self, labels, predictions, delta=1.0):  # regression
        residual = tf.abs(predictions - labels)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.reduce_mean(tf.where(condition, small_res, large_res))

    def mse_loss(self, labels, predictions):
        return tf.reduce_mean(tf.square(labels - predictions))


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
        # output_att = self.task_specific_attention(inputs, self.embedding_size)
        textcnn_out = self.TextCNN(inputs, self.fact_len)
        return textcnn_out


