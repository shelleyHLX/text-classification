# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

import tensorflow.contrib.layers as layers
import numpy as np

class Settings(object):
    def __init__(self):
        self.model_name = 'entity_network'
        self.fact_len = 200
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 100
        self.stride_length = 3

        self.use_bi_lstm=False
        self.story_length = 1
        self.block_size=20

        self.n_class = 202
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class Atten_TextCNN(object):
    def __init__(self, W_embedding,settings):
        self.model_name = settings.model_name
        self.fact_len = settings.fact_len
        self.hidden_size = settings.hidden_size
        self.n_class = settings.n_class
        self.filter_sizes = settings.filter_sizes
        self.stride_length = settings.stride_length
        self.n_filter = settings.n_filter
        self.n_filter_total = self.n_filter * len(self.filter_sizes)
        self.fc_hidden_size = settings.fc_hidden_size

        self.use_bi_lstm = settings.use_bi_lstm
        self.story_length = settings.story_length
        self.block_size = settings.block_size
        self.dimension=self.hidden_size*2 if self.use_bi_lstm else self.hidden_size
        #if use bi-lstm, set dimension value, so it can be used later for parameter.


        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        self.update_emas = list()
        self._tst = tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            # self._X_inputs = tf.placeholder(tf.int32, [None, self.fact_len], name='X_input')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')
            self.story=tf.placeholder(tf.int32,[None,self.story_length,self.fact_len],name="story")
            self.query = tf.placeholder(tf.int32, [None, self.fact_len], name="question")

        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
            # self.embedding2 = tf.get_variable(name='embedding', shape=W_embedding.shape,
            #                                  initializer=tf.constant_initializer(W_embedding2), trainable=True)
        self.embedding_size = 100

        with tf.variable_scope('Atten_TextCNN'):
            self.logits = self._inference()  # [None, self.label_size]. main computation graph is here.
            print('logits: ', self.logits)

            self._y_pred = tf.sigmoid(self.logits)
            print(self.y_pred)
            exit(0)


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

    # @property
    # def X_inputs(self):
    #     return self._X_inputs

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
    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope("output_module"):
            self.H=tf.get_variable("H",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.R = tf.get_variable("R", shape=[self.dimension, self.n_class], initializer=self.initializer)
            self.y_bias=tf.get_variable("y_bias",shape=[self.n_class])
            self.b_projected = tf.get_variable("b_projection", shape=[self.n_class])
            self.h_u_bias=tf.get_variable("h_u_bias",shape=[self.dimension])

        with tf.variable_scope("dynamic_memory"):
            self.U=tf.get_variable("U",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.V=tf.get_variable("V",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.W=tf.get_variable("W",shape=[self.dimension,self.dimension],initializer=self.initializer)
            self.h_bias=tf.get_variable("h_bias",shape=[self.dimension])
            self.h2_bias = tf.get_variable("h2_bias", shape=[self.dimension])

        with tf.variable_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],initializer=self.initializer)  # TODO o.k to use batch_size in first demension?
    #



    def embedding_with_mask(self):
        # 1.1 embedding for story and query
        story_embedding = tf.nn.embedding_lookup(self.embedding,self.story)  # [batch_size,story_length,sequence_length,embed_size]
        query_embedding=tf.nn.embedding_lookup(self.embedding,self.query)    # [batch_size,sequence_length,embed_size]
        # 1.2 mask for story and query
        story_mask=tf.get_variable("story_mask",[self.fact_len,1],initializer=tf.constant_initializer(1.0))
        query_mask=tf.get_variable("query_mask",[self.fact_len,1],initializer=tf.constant_initializer(1.0))
        # 1.3 multiply of embedding and mask for story and query
        self.story_embedding=tf.multiply(story_embedding,story_mask)  # [batch_size,story_length,sequence_length,embed_size]
        self.query_embedding=tf.multiply(query_embedding,query_mask)  # [batch_size,sequence_length,embed_size]


    def input_encoder_bi_lstm(self):
        """use bi-directional lstm to encode query_embedding:[batch_size,sequence_length,embed_size]
                                         and story_embedding:[batch_size,story_length,sequence_length,embed_size]
        output:query_embedding:[batch_size,hidden_size*2]  story_embedding:[batch_size,self.story_length,self.hidden_size*2]
        """
        #1. encode query: bi-lstm layer
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
        if self.keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.keep_prob)
        query_hidden_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.query_embedding,dtype=tf.float32,scope="query_rnn")  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        query_hidden_output = tf.concat(query_hidden_output, axis=2) #[batch_size,sequence_length,hidden_size*2]
        self.query_embedding=tf.reduce_sum(query_hidden_output,axis=1) #[batch_size,hidden_size*2]
        print("input_encoder_bi_lstm.self.query_embedding:",self.query_embedding)

        #2. encode story
        # self.story_embedding:[batch_size,story_length,sequence_length,embed_size]
        self.story_embedding=tf.reshape(self.story_embedding,shape=(-1,self.story_length*self.fact_len,self.embedding_size)) #[self.story_length*self.sequence_length,self.embed_size]
        lstm_fw_cell_story = rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
        lstm_bw_cell_story = rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
        if self.keep_prob is not None:
            lstm_fw_cell_story = rnn.DropoutWrapper(lstm_fw_cell_story, output_keep_prob=self.keep_prob)
            lstm_bw_cell_story = rnn.DropoutWrapper(lstm_bw_cell_story, output_keep_prob=self.keep_prob)
        story_hidden_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_story, lstm_bw_cell_story,
                                                                 self.story_embedding,dtype=tf.float32,scope="story_rnn")
        story_hidden_output=tf.concat(story_hidden_output,axis=2) #[batch_size,story_length*sequence_length,hidden_size*2]
        story_hidden_output=tf.reshape(story_hidden_output,shape=(-1,self.story_length,self.fact_len,self.hidden_size*2))
        self.story_embedding = tf.reduce_sum(story_hidden_output, axis=2)  # [batch_size,self.story_length,self.hidden_size*2]

    def input_encoder_bow(self):
        # 1.4 use bag of words to encoder story and query
        self.story_embedding=tf.reduce_sum(self.story_embedding,axis=2) #[batch_size,story_length,embed_size]
        self.query_embedding=tf.reduce_sum(self.query_embedding,axis=1)  #[batch_size,embed_size]

    def rnn_story(self):
        """
        run rnn for story to get last hidden state
        input is:  story:                 [batch_size,story_length,embed_size]
        :return:   last hidden state.     [batch_size,embed_size]
        """
        # 1.split input to get lists.
        input_split=tf.split(self.story_embedding,self.story_length,axis=1) #a list.length is:story_length.each element is:[batch_size,1,embed_size]
        input_list=[tf.squeeze(x,axis=1) for x in input_split]           #a list.length is:story_length.each element is:[batch_size,embed_size]
        # 2.init keys(w_all) and values(h_all) of memory
        h_all=tf.get_variable("hidden_states",shape=[self.block_size,self.dimension],initializer=self.initializer)# [block_size,hidden_size]
        w_all=tf.get_variable("keys",          shape=[self.block_size,self.dimension],initializer=self.initializer)# [block_size,hidden_size]
        # 3.expand keys and values to prepare operation of rnn
        w_all_expand=tf.tile(tf.expand_dims(w_all,axis=0),[self.batch_size,1,1]) #[batch_size,block_size,hidden_size]
        h_all_expand=tf.tile(tf.expand_dims(h_all,axis=0),[self.batch_size,1,1]) #[batch_size,block_size,hidden_size]
        # 4. run rnn using input with cell.
        for i,input in enumerate(input_list):
            h_all_expand=self.cell(input,h_all_expand,w_all_expand,i) #w_all:[batch_size,block_size,hidden_size]; h_all:[batch_size,block_size,hidden_size]
        return h_all_expand #[batch_size,block_size,hidden_size]


    def cell(self,s_t,h_all,w_all,i):
        """
        parallel implementation of single time step for compute of input with memory
        :param s_t:   [batch_size,hidden_size].vector representation of current input(is a sentence).notice:hidden_size=embedding_size
        :param w_all: [batch_size,block_size,hidden_size]
        :param h_all: [batch_size,block_size,hidden_size]
        :return: new hidden state: [batch_size,block_size,hidden_size]
        """
        # 1.gate
        s_t_expand=tf.expand_dims(s_t, axis=1)       #[batch_size,1,hidden_size]
        g=tf.nn.sigmoid(tf.multiply(s_t_expand,h_all)+tf.multiply(s_t_expand,w_all))#shape:[batch_size,block_size,hidden_size]

        # 2.candidate hidden state
        #below' shape:[batch_size*block_size,hidden_size]
        h_candidate_part1=tf.matmul(tf.reshape(h_all,shape=(-1,self.dimension)), self.U) + tf.matmul(tf.reshape(w_all,shape=(-1,self.dimension)), self.V)+self.h_bias
        print("======>h_candidate_part1:",h_candidate_part1) #(160, 100)
        h_candidate_part1=tf.reshape(h_candidate_part1,shape=(self.batch_size,self.block_size,self.dimension)) #[batch_size,block_size,hidden_size]
        h_candidate_part2=tf.expand_dims(tf.matmul(s_t,self.W)+self.h2_bias,axis=1)              #shape:[batch_size,1,hidden_size]
        h_candidate=self.activation(h_candidate_part1+h_candidate_part2,scope="h_candidate"+str(i))   #shape:[batch_size,block_size,hidden_size]

        # 3.update hidden state
        h_all=h_all+tf.multiply(g,h_candidate) #shape:[batch_size,block_size,hidden_size]

        # 4.normalized hidden state
        h_all=tf.nn.l2_normalize(h_all,-1) #shape:[batch_size,block_size,hidden_size]
        return h_all  #shape:[batch_size,block_size,hidden_size]

    def activation(self,features, scope=None):  # scope=None
        with tf.variable_scope(scope, 'PReLU', initializer=self.initializer):
            alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
            pos = tf.nn.relu(features)
            neg = alpha * (features - tf.abs(features)) * 0.5
            return pos + neg

    def output_module(self):
        """
        1.use attention mechanism between query and hidden states, to get weighted sum of hidden state. 2.non-linearity of query and hidden state to get label.
        input: query_embedding:[batch_size,embed_size], hidden state:[batch_size,block_size,hidden_size] of memory
        :return:y: predicted label.[]
        """
        # 1.use attention mechanism between query and hidden states, to get weighted sum of hidden state.
        # 1.1 get possibility distribution (of similiarity)
        p=tf.nn.softmax(tf.multiply(tf.expand_dims(self.query_embedding,axis=1),self.hidden_state)) #shape:[batch_size,block_size,hidden_size]<---query_embedding_expand:[batch_size,1,hidden_size]; hidden_state:[batch_size,block_size,hidden_size]
        # 1.2 get weighted sum of hidden state
        u=tf.reduce_sum(tf.multiply(p,self.hidden_state),axis=1) #shape:[batch_size,hidden_size]<----------([batch_size,block_size,hidden_size],[batch_size,block_size,hidden_size])

        # 2.non-linearity of query and hidden state to get label
        H_u_matmul=tf.matmul(u,self.H)+self.h_u_bias #shape:[batch_size,hidden_size]<----([batch_size,hidden_size],[hidden_size,hidden_size])
        activation=self.activation(self.query_embedding + H_u_matmul,scope="query_add_hidden")           #shape:[batch_size,hidden_size]
        activation = tf.nn.dropout(activation,keep_prob=self.keep_prob) #shape:[batch_size,hidden_size]
        y=tf.matmul(activation,self.R)+self.y_bias #shape:[batch_size,vocab_size]<-----([batch_size,hidden_size],[hidden_size,vocab_size])
        return y #shape:[batch_size,vocab_size]


    def _inference(self):
        """main computation graph here: 1.input encoder 2.dynamic emeory 3.output layer """
        # 1.input encoder
        self.embedding_with_mask()
        if self.use_bi_lstm:
            self.input_encoder_bi_lstm()
        else:
            self.input_encoder_bow()
        # 2. dynamic emeory
        self.hidden_state=self.rnn_story() #[batch_size,block_size,hidden_size]. get hidden state after process the story

        # 3.output layer
        self.logits=self.output_module() #[batch_size,vocab_size]

