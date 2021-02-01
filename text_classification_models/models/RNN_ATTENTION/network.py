# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

class Settings(object):
    def __init__(self):
        self.model_name = 'RNN_Attention'
        self.fact_len = 200
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 256
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
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size
        self.n_class = settings.n_class
        self.filter_sizes = settings.filter_sizes
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.gru_hidden_size = settings.gru_hidden_size
        self.n_layer = settings.n_layer
        self.fc_hidden_size = settings.fc_hidden_size
        self.initializer=tf.random_normal_initializer(stddev=0.1)

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
            exit(0)

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
        print('============= gru_cell =============')
        with tf.name_scope('gru_cell'):
            cell = rnn.GRUCell(self.gru_hidden_size, reuse=tf.get_variable_scope().reuse)
            print('type(cell): ', type(cell))
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_gru(self, inputs):
        print('================== bi_gru ==================')
        """build the Bi-GRU network. 返回个所有层的隐含状态。"""
        cells_fw = [self.gru_cell() for _ in range(self.n_layer)]
        cells_bw = [self.gru_cell() for _ in range(self.n_layer)]
        print('inputs: ', inputs.shape)  # (?, 200, 256)
        print('cells_fw: ', type(cells_fw))
        print('cells_bw: ', type(cells_bw))
        initial_states_fw = [cell_fw.zero_state(self.batch_size, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(self.batch_size, tf.float32) for cell_bw in cells_bw]
        # print('initial_states_bw: ', np.asarray(initial_states_bw).shape)
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                            initial_states_fw=initial_states_fw,
                                                            initial_states_bw=initial_states_bw,
                                                            dtype=tf.float32)
        print('outputs: ', outputs.shape)  # (?, 200, self.hidden_size * 2)

        return outputs

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

    def attention(self,input_sequences,attention_level,reuse_flag=False):
        """
        :param input_sequence: [batch_size,seq_length,num_units]
        :param attention_level: word or sentence level
        :return: [batch_size,hidden_size]
        """
        num_units=input_sequences.get_shape().as_list()[-1] * 4  # get last dimension embedding_size
        with tf.variable_scope("attention_" + str(attention_level),reuse=reuse_flag):
            v_attention = tf.get_variable("u_attention" + attention_level, shape=[num_units],initializer=self.initializer)
            #1.one-layer MLP
            u=tf.layers.dense(input_sequences,num_units,activation=tf.nn.tanh,use_bias=True) #[batch_size,seq_legnth,num_units].no-linear
            #2.compute weight by compute simility of u and attention vector v
            score=tf.multiply(u,v_attention) #[batch_size,seq_length,num_units]
            weight=tf.reduce_sum(score,axis=2,keep_dims=True) #[batch_size,seq_length,1]
            #weight=tf.nn.softmax(weight,axis=1) #[batch_size,seq_length,1] #TODO temp removed since it make performance worse 2018.05.29
            #3.weight sum
            attention_representation=tf.reduce_sum(tf.multiply(u,weight),axis=1) #[batch_size,num_units]. TODO here we not use original input_sequences but transformed version of input: u.
        return attention_representation

    def attention_multihop(self,input_sequences,attention_level,reuse_flag=False):
        """
        perform multi-hop attention, instead of only one hop. but final demsion is same as before.
        :param input_sequences:[batch_size,sequence_length,num_units]
        :param attention_level:
        :param reuse_flag:
        :return: attention_representation:[batch_size,sequence_length,num_units]
        """
        num_hops=4
        num_units = input_sequences.get_shape().as_list()[-1]  # get last dimension
        attention_rep_list=[]
        for i in range(num_hops):
            with tf.variable_scope("attention_"+str(i) +'_'+ str(attention_level), reuse=reuse_flag):
                v_attention = tf.get_variable("u_attention" + attention_level, shape=[num_units], initializer=self.initializer)
                # 1.one-layer MLP
                u = tf.layers.dense(input_sequences, num_units, activation=tf.nn.tanh,use_bias=True)  # [batch_size,seq_legnth,num_units].no-linear
                # 2.compute weight by compute simility of u and attention vector v
                score = tf.multiply(u, v_attention)  # [batch_size,seq_length,num_units]
                weight = tf.reduce_sum(score, axis=2, keep_dims=True)  # [batch_size,seq_length,1]
                # weight=tf.nn.softmax(weight,axis=1) #[batch_size,seq_length,1] #TODO temp removed since it make performance worse 2018.05.29
                # 3.weight sum
                attention_rep = tf.reduce_sum(tf.multiply(u, weight),axis=1)  # [batch_size,num_units]. TODO here we not use original input_sequences but transformed version of input: u.
                attention_rep_list.append(attention_rep)

        attention_representation=tf.concat(attention_rep_list,axis=-1) #[
        return attention_representation

    def attention_multiply(self,input_sequences,attention_level,reuse_flag=False): #TODO need update
        """
        :param input_sequence: [batch_size,seq_length,num_units]
        :param attention_level: word or sentence level
        :return: [batch_size,hidden_size]
        """
        num_units=input_sequences.get_shape().as_list()[-1] * 4  # get last dimension
        with tf.variable_scope("attention_" + str(attention_level),reuse=reuse_flag):
            v_attention = tf.get_variable("u_attention" + attention_level, shape=[num_units],initializer=self.initializer)
            #1.one-layer MLP
            u=tf.layers.dense(input_sequences,num_units,activation=tf.nn.tanh,use_bias=True) #[batch_size,seq_legnth,num_units].no-linear
            #2.compute weight by compute simility of u and attention vector v
            score=tf.multiply(u,v_attention) #[batch_size,seq_length,num_units]. TODO NEED ADD multiply SCALE V_a.
            score=tf.reduce_sum(score,axis=2,keep_dims=True) #todo keepdims-->keep_dims /tf.sqrt(tf.cast(num_units,tf.float32)) #[batch_size,seq_length,1]
            weight=tf.nn.softmax(score,dim=1) #[batch_size,seq_length,1]
            #3.weight sum
            attention_representation=tf.reduce_sum(tf.multiply(input_sequences,weight),axis=1) #[batch_size,num_units]
        return attention_representation


    def attention_additive_batch(self,input_sequences_original, attention_level,reuse_flag=False):
        #TODO check: paper 'Neural Machine Transation By Jointly Learning To Align and Translate'

        """ additive attention(support batch of input with sequences)
        :param input_sequence: [batch_size,seq_length,num_units]
        :param attention_level: word or sentence level
        :return: #[batch_size,sequence_length]
        """
        # [batch_size,seq_length,num_units*2].
        input_sequences=tf.transpose(input_sequences_original,perm=[0,2,1]) #[batch_size,num_units,sequence_length]<---[batch_size,seq_length,num_units].
        _, num_units, sequence_lenghth = input_sequences.get_shape().as_list()
        print("###attention_additive_batch.input_sequences:",input_sequences,";attention_level:",attention_level,"num_units:", num_units, ";sequence_lenghth:", sequence_lenghth)
        with tf.variable_scope("attention_" + str(attention_level), reuse=reuse_flag):
            # 1.create or get learnable variables
            attention_vector = tf.get_variable("attention_vector_" + attention_level,shape=[num_units, 1],initializer=self.initializer)
            W = tf.get_variable("W" + attention_level,shape=[1, num_units, num_units],initializer=self.initializer)
            U = tf.get_variable("U" + attention_level, shape=[num_units, num_units],initializer=self.initializer)
            v = tf.get_variable("v" + attention_level, shape=[1, 1, num_units],initializer=self.initializer)

            # 2.get part1 and part2 of additive attention
            W = tf.tile(W, (self.batch_size, 1, 1))  # [batch_size,num_units,num_units]
            part1 = tf.matmul(W,input_sequences)  # [batch_size,num_units,sequence_length]<----([batch_size,num_units,num_units],[batch_size,num_units,sequence_length])
            part2 = tf.expand_dims(tf.matmul(U, attention_vector),axis=0)  # [1,num_units,1]<---[num_units,1]<-----([num_units,num_units],[num_units,1])

            # 3.activation
            activation = tf.nn.tanh(part1 + part2)  # [batch_size,num_units,sequence_length]

            # 4. get attention score by using matmul
            v = tf.tile(v, (self.batch_size, 1, 1))  # [batch_size,1,num_units]
            score = tf.matmul(v,activation)  # [batch_size,1,sequence_length]<------([batch_size,1,num_units],[batch_size,num_units,sequence_length])
            score = tf.squeeze(score)  # [batch_size,sequence_length]

            # 5. normalize using softmax
            weights=tf.nn.softmax(score,dim=1) #[batch_size,sequence_length]

            # 6. weighted sum
            weights=tf.expand_dims(weights,axis=-1) #[batch_size,sequence_length,1]
            result=tf.multiply(input_sequences_original,weights) #[batch_size,squence_length,num_units]
            result=tf.reduce_sum(result,axis=1) #[batch_size,num_units]
        return result  # [batch_size,num_units]

    def attention_additive(self,input_sequence,attention_level,reuse_flag=False): #check: paper 'Neural Machine Transation By Jointly Learning To Align and Translate'

        """
        :param input_sequence: [num_units,1]
        :param attention_level: word or sentence level
        :return: [batch_size,hidden_size]
        """
        attention_representation=None

        num_units=input_sequence.get_shape().as_list()[-1] * 4 #get last dimension
        with tf.variable_scope("attention_" + str(attention_level),reuse=reuse_flag):
            attention_vector = tf.get_variable("attention_vector_" + attention_level, shape=[num_units,1],initializer=self.initializer)
            W=tf.get_variable("W" + attention_level, shape=[num_units,num_units],initializer=self.initializer)
            U=tf.get_variable("U" + attention_level, shape=[num_units,num_units],initializer=self.initializer)

            v = tf.get_variable("v" + attention_level, shape=[1,num_units],initializer=self.initializer)
            part1=tf.matmul(W,input_sequence)   #[num_units,1]<----([num_units,num_units],[num_units,1])
            part2=tf.matmul(U,attention_vector) #[num_units,1]<-----([num_units,num_units],[num_units,1])
            activation=tf.nn.tanh(part1+part2)  #[num_units,1]
            result=tf.matmul(v,activation) #  [1,1]<------([1,num_units],[num_units,1])
            result=tf.reshape(result,()) #scalar
        return result


    def _inference(self, X_inputs):
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        inputs = self.bi_gru(inputs)
        out = self.task_specific_attention(inputs, self.fact_len)
        return out


