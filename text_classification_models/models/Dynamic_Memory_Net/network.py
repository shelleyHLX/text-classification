# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn

import tensorflow.contrib.layers as layers
import numpy as np

class Settings(object):
    def __init__(self):
        self.model_name = 'Dynamic_Memory_Net'
        self.fact_len = 200
        self.filter_sizes = [1, 2, 3, 4, 5]
        self.n_filter = 256
        self.fc_hidden_size = 1024
        self.hidden_size = 1024
        self.stride_length = 3

        self.story_length = 1
        self.num_pass = 2
        self.use_gated_gru = True
        self.decode_with_sequences=False

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

        self.story_length = settings.story_length
        self.num_pass = settings.num_pass
        self.use_gated_gru = settings.use_gated_gru
        self.decode_with_sequences=settings.decode_with_sequences

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
        self.embedding_size = W_embedding.shape[1]

        with tf.variable_scope('Atten_TextCNN'):
            self.instantiate_weights()
            self._inference()
            self._y_pred = tf.sigmoid(self.logits)
            print(self.y_pred)
            # exit(0)

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

    def input_module(self):
        """encode raw texts into vector representation"""
        story_embedding=tf.nn.embedding_lookup(self.embedding,self.story)  # [batch_size,story_length,sequence_length,embed_size]
        story_embedding=tf.reshape(story_embedding,(self.batch_size,self.story_length,self.fact_len*self.embedding_size))
        hidden_state=tf.ones((self.batch_size,self.hidden_size),dtype=tf.float32)
        cell = rnn.GRUCell(self.hidden_size)
        self.story_embedding, hidden_state=tf.nn.dynamic_rnn(cell,story_embedding,dtype=tf.float32,scope="input_module")

    def question_module(self):
        """
        input:tokens of query:[batch_size,sequence_length]
        :return: representation of question:[batch_size,hidden_size]
        """
        query_embedding = tf.nn.embedding_lookup(self.embedding, self.query)  # [batch_size,sequence_length,embed_size]
        cell=rnn.GRUCell(self.hidden_size)
        _,self.query_embedding=tf.nn.dynamic_rnn(cell,query_embedding,dtype=tf.float32,scope="question_module") #query_embedding:[batch_size,hidden_size]

    def x1Wx2_parallel(self,x1,x2,scope):
        """
        :param x1: [batch_size,story_length,hidden_size]
        :param x2: [batch_size,1,hidden_size]
        :param scope: a string
        :return:  [batch_size,story_length,hidden_size]
        """
        with tf.variable_scope(scope):
            print('x1', x1)
            bs, sl, hs = x1.get_shape().as_list()
            x1=tf.reshape(x1,shape=(-1, sl*hs)) #[batch_size,story_length*hidden_size]
            print('x1', x1)
            x1_w=tf.layers.dense(x1,self.story_length*self.hidden_size,use_bias=False) #[self.hidden_size, story_length*self.hidden_size]
            x1_w_expand=tf.expand_dims(x1_w,axis=2)     #[batch_size,story_length*self.hidden_size,1]
            x1_w_x2=tf.matmul(x1_w_expand,x2)           #[batch_size,story_length*self.hidden_size,hidden_size]
            x1_w_x2=tf.reshape(x1_w_x2,shape=(self.batch_size,self.story_length,self.hidden_size,self.hidden_size))
            x1_w_x2=tf.reduce_sum(x1_w_x2,axis=3)      #[batch_size,story_length,hidden_size]
            return x1_w_x2

    def attention_mechanism_parallel(self,c_full,m,q,i):
        """ parallel implemtation of gate function given a list of candidate sentence, a query, and previous memory.
        Input:
           c_full: candidate fact. shape:[batch_size,story_length,hidden_size]
           m: previous memory. shape:[batch_size,hidden_size]
           q: question. shape:[batch_size,hidden_size]
        Output: a scalar score (in batch). shape:[batch_size,story_length]
        """
        q=tf.expand_dims(q,axis=1) #[batch_size,1,hidden_size]
        m=tf.expand_dims(m,axis=1) #[batch_size,1,hidden_size]

        # 1.define a large feature vector that captures a variety of similarities between input,memory and question vector: z(c,m,q)
        c_q_elementwise=tf.multiply(c_full,q)          #[batch_size,story_length,hidden_size]
        c_m_elementwise=tf.multiply(c_full,m)          #[batch_size,story_length,hidden_size]
        c_q_minus=tf.abs(tf.subtract(c_full,q))        #[batch_size,story_length,hidden_size]
        c_m_minus=tf.abs(tf.subtract(c_full,m))        #[batch_size,story_length,hidden_size]
        # c_transpose Wq
        c_w_q=self.x1Wx2_parallel(c_full,q,"c_w_q"+str(i))   #[batch_size,story_length,hidden_size]
        c_w_m=self.x1Wx2_parallel(c_full,m,"c_w_m"+str(i))   #[batch_size,story_length,hidden_size]
        # c_transposeWm
        q_tile=tf.tile(q,[1,self.story_length,1])     #[batch_size,story_length,hidden_size]
        m_tile=tf.tile(m,[1,self.story_length,1])     #[batch_size,story_length,hidden_size]
        z=tf.concat([c_full,m_tile,q_tile,c_q_elementwise,c_m_elementwise,c_q_minus,c_m_minus,c_w_q,c_w_m],2) #[batch_size,story_length,hidden_size*9]
        # 2. two layer feed foward
        g=tf.layers.dense(z,self.hidden_size*3,activation=tf.nn.tanh)  #[batch_size,story_length,hidden_size*3]
        g=tf.layers.dense(g,1,activation=tf.nn.sigmoid)                #[batch_size,story_length,1]
        g=tf.squeeze(g,axis=2)                                         #[batch_size,story_length]
        return g

    #:param s_t: vector representation of current input(is a sentence). shape:[batch_size,sequence_length,embed_size]
    #:param h: value(hidden state).shape:[hidden_size]
    #:param w: key.shape:[hidden_size]
    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope("gru_cell"):
            # self.W_z = tf.get_variable("W_z", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            # self.U_z = tf.get_variable("U_z", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            # self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # # GRU parameters:reset gate related
            # self.W_r = tf.get_variable("W_r", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            # self.U_r = tf.get_variable("U_r", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            # self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])
            #
            # self.W_h = tf.get_variable("W_h", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            # self.U_h = tf.get_variable("U_h", shape=[self.embedding_size, self.hidden_size], initializer=self.initializer)
            # self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])
            self.W_z = tf.get_variable("W_z", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.hidden_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])


    def gru_cell(self, Xt, h_t_minus_1,variable_scope):
        """
        single step of gru
        :param Xt: Xt:[batch_size,hidden_size]
        :param h_t_minus_1:[batch_size,hidden_size]
        :return:[batch_size,hidden_size]
        """
        with tf.variable_scope(variable_scope):
            # 1.update gate: decides how much past information is kept and how much new information is added.
            print(Xt)
            print(self.W_z)
            print(h_t_minus_1)
            print(self.U_z)
            z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,self.U_z) + self.b_z)  # z_t:[batch_size,self.hidden_size]
            # 2.reset gate: controls how much the past state contributes to the candidate state.
            r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,self.U_r) + self.b_r)  # r_t:[batch_size,self.hidden_size]
            # 3.compute candiate state h_t~
            h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size,self.hidden_size]
            # 4.compute new state: a linear combine of pervious hidden state and the current new state h_t~
            h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size,hidden_size]
        return h_t

    def gated_gru(self,c_current,h_previous,g_current):
        """
        gated gru to get updated hidden state
        :param  c_current: [batch_size,embedding_size]
        :param  h_previous:[batch_size,hidden_size]
        :param  g_current: [batch_size,1]
        :return h_current: [batch_size,hidden_size]
        """
        # 1.compute candidate hidden state using GRU.
        h_candidate=self.gru_cell(c_current, h_previous,"gru_candidate_sentence") #[batch_size,hidden_size]
        # 2.combine candidate hidden state and previous hidden state using weight(a gate) to get updated hidden state.
        h_current=tf.multiply(g_current,h_candidate)+tf.multiply(1-g_current,h_previous) #[batch_size,hidden_size]
        return h_current

    def episodic_memory_module(self):#input(story):[batch_size,story_length,hidden_size]
        """
        episodic memory module
        1.combine features
        1.attention mechansim using gate function.take fact representation c,question q,previous memory m_previous
        2.use gated-gru to update hidden state
        3.set last hidden state as episode result
        4.use gru to update final memory using episode result

        input: story(from input module):[batch_size,story_length,hidden_size]
        output: last hidden state:[batch_size,hidden_size]
        """
        candidate_inputs=tf.split(self.story_embedding,self.story_length,axis=1) # a list. length is: story_length. each element is:[batch_size,1,embedding_size]
        candidate_list=[tf.squeeze(x,axis=1) for x in candidate_inputs]          # a list. length is: story_length. each element is:[batch_size  ,embedding_size]
        m_current=self.query_embedding
        h_current = tf.zeros((self.batch_size, self.hidden_size))
        for pass_number in range(self.num_pass):#for each candidate sentence in the list,do loop.
            # 1. attention mechansim.take fact representation c,question q,previous memory m_previous
            g = self.attention_mechanism_parallel(self.story_embedding, m_current,self.query_embedding,pass_number)  # [batch_size,story_length]
            # 2.below is Memory Update Mechanism
            if self.use_gated_gru: #use gated gru to update episode. this is default method.
                g = tf.split(g, self.story_length,axis=1)  # a list. length is: sequence_length. each element is:[batch_size,1]
                # 2.1 use gated-gru to update hidden state
                for i,c_current in enumerate(candidate_list):
                    print('c_current-------------c_current', c_current)
                    g_current=g[i] #[batch_size,1]
                    h_current=self.gated_gru(c_current,h_current,g_current) #h_current:[batch_size,hidden_size]. g[i] represent score( a scalar) for current candidate sentence:c_current.
                # 2.2 assign last hidden state to e(episodic)
                e_i=h_current #[batch_size,hidden_size]
            else: #use weighted sum to get episode(e.g. used in question answering)
                p_gate=tf.nn.softmax(g,dim=1)                #[batch_size,story_length]. compute weight
                p_gate=tf.expand_dims(p_gate,axis=2)         #[batch_size,story_length,1]
                e_i=tf.multiply(p_gate,self.story_embedding) #[batch_size,story_length,hidden_size]
                e_i=tf.reduce_sum(e_i,axis=1)                #[batch_size,story_length]
            #3. use gru to update episodic memory m_i
            m_current=self.gru_cell(e_i, m_current,"gru_episodic_memory") #[batch_size,hidden_size]
        self.m_T=m_current #[batch_size,hidden_size]

    def answer_module(self):
        """ Answer Module:generate an answer from the final memory vector.
        Input:
            hidden state from episodic memory module:[batch_size,hidden_size]
            question:[batch_size, embedding_size]
        """
        steps=self.fact_len if self.decode_with_sequences else 1 #decoder for a list of tokens with sequence. e.g."x1 x2 x3 x4..."
        a=self.m_T #init hidden state
        y_pred=tf.zeros((self.batch_size,self.hidden_size)) #TODO usually we will init this as a special token '<GO>', you can change this line by pass embedding of '<GO>' from outside.
        logits_list=[]
        logits_return=None
        for i in range(steps):
            cell = rnn.GRUCell(self.hidden_size)
            y_previous_q=tf.concat([y_pred,self.query_embedding],axis=1) #[batch_hidden_size*2]
            _, a = cell( y_previous_q,a)
            logits=tf.layers.dense(a,units=self.n_class) #[batch_size,vocab_size]
            logits_list.append(logits)
        if self.decode_with_sequences:#need to get sequences.
            logits_return = tf.stack(logits_list, axis=1)  # [batch_size,sequence_length,num_classes]
        else:#only need to get an answer, not sequences
            logits_return = logits_list[0]  #[batcj_size,num_classes]

        return logits_return

    def _inference(self):
        """main computation graph here: a.Input Module,b.Question Module,c.Episodic Memory Module,d.Answer Module """
        # 1.Input Module
        self.input_module() #[batch_size,story_length,hidden_size
        # 2.question module
        self.question_module() #[batch_size,hidden_size]
        # 3.episodic memory module
        self.episodic_memory_module() #[batch_size,hidden_size]
        # 4. answer module
        self.logits=self.answer_module() #[batch_size,vocab_size]



