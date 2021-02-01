# -*- coding:utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn
from models.seq2seq_attention.seq2seq_net import rnn_decoder_with_attention, extract_argmax_and_embed

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
        self.decoder_sent_length = 6
        self.n_class = 202
        self.summary_path = '../../summary/' + self.model_name + '/'
        self.ckpt_path = '../../ckpt/' + self.model_name + '/'
        self.log_path = '../../log/' + self.model_name + '/'


class seq2seq_attention_model:
    def __init__(self, W_ebedding, settings, ):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = settings.n_class
        self.sequence_length = settings.fact_len

        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.decoder_sent_length=settings.decoder_sent_length
        self.hidden_size = settings.hidden_size
        self.l2_lambda=0.0001

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")                 #x
        self.decoder_input = tf.placeholder(tf.int32, [None, self.decoder_sent_length],name="decoder_input")  #y, but shift
        self.input_y_label = tf.placeholder(tf.int32, [None, self.decoder_sent_length], name="input_y_label") #y, but shift
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.batch_size = tf.placeholder(tf.int32, [None], name='batch size')

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        self.instantiate_weights()
        self.logits = self.inference() #logits shape:[batch_size,decoder_sent_length,self.num_classes]

        self.loss = self.loss_seq2seq()

    def inference(self):
        """main computation graph here:
        #1.Word embedding. 2.Encoder with GRU 3.Decoder using GRU(optional with attention)."""
        ###################################################################################################################################
        # 1.embedding of words
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)  #[None, self.sequence_length, self.embed_size]
        # 2.encoder with GRU
        # 2.1 forward gru
        hidden_state_forward_list = self.gru_forward(self.embedded_words,self.gru_cell)  # a list,length is sentence_length, each element is [batch_size,hidden_size]
        # 2.2 backward gru
        hidden_state_backward_list = self.gru_forward(self.embedded_words,self.gru_cell,reverse=True)  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        # 2.3 concat forward hidden state and backward hidden state. hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        thought_vector_list=[tf.concat([h_forward,h_backward],axis=1) for h_forward,h_backward in zip(hidden_state_forward_list,hidden_state_backward_list)]#list,len:sent_len,e:[batch_size,hidden_size*2]

        # 3.Decoder using GRU with attention
        thought_vector=tf.stack(thought_vector_list,axis=1) #shape:[batch_size,sentence_length,hidden_size*2]
        #initial_state=tf.reduce_sum(thought_vector,axis=1) #[batch_size,hidden_size*2] #TODO NEED TO TEST WHICH ONE IS BETTER: SUM UP OR USE LAST HIDDEN STATE==>similiarity.
        initial_state=tf.nn.tanh(tf.matmul(hidden_state_backward_list[0],self.W_initial_state)+self.b_initial_state) #initial_state:[batch_size,hidden_size*2]. TODO this is follow paper's way.
        cell=self.gru_cell_decoder #this is a special cell. because it beside previous hidden state, current input, it also has a context vecotor, which represent attention result.

        output_projection=(self.W_projection,self.b_projection) #W_projection:[self.hidden_size * 2, self.num_classes]; b_projection:[self.num_classes]
        loop_function = extract_argmax_and_embed(self.Embedding_label,output_projection) if not self.is_train else None #loop function will be used only at testing, not training.
        attention_states=thought_vector #[None, self.sequence_length, self.embed_size]
        decoder_input_embedded=tf.nn.embedding_lookup(self.Embedding_label,self.decoder_input) #[batch_size,self.decoder_sent_length,embed_size]
        decoder_input_splitted = tf.split(decoder_input_embedded, self.decoder_sent_length,axis=1)  # it is a list,length is decoder_sent_length, each element is [batch_size,1,embed_size]
        decoder_input_squeezed = [tf.squeeze(x, axis=1) for x in decoder_input_splitted]  # it is a list,length is decoder_sent_length, each element is [batch_size,embed_size]

        #rnn_decoder_with_attention(decoder_inputs, initial_state, cell, loop_function,attention_states,scope=None):
        #input1:decoder_inputs:target, shift by one. for example.the target is:"X Y Z",then decoder_inputs should be:"START X Y Z" A list of 2D Tensors [batch_size x input_size].
        #input2:initial_state: 2D Tensor with shape  [batch_size x cell.state_size].
        #input3:attention_states:represent X. 3D Tensor [batch_size x attn_length x attn_size].
        #output:?
        outputs, final_state=rnn_decoder_with_attention(decoder_input_squeezed, initial_state, cell, loop_function, attention_states, scope=None) # A list.length:decoder_sent_length.each element is:[batch_size x output_size]
        decoder_output=tf.stack(outputs,axis=1) #decoder_output:[batch_size,decoder_sent_length,hidden_size*2]
        decoder_output=tf.reshape(decoder_output,shape=(-1,self.hidden_size*2)) #decoder_output:[batch_size*decoder_sent_length,hidden_size*2]

        with tf.name_scope("dropout"):
            decoder_output = tf.nn.dropout(decoder_output,keep_prob=self.dropout_keep_prob)  # shape:[None,hidden_size*4]
        # 4. get logits
        with tf.name_scope("output"):
            logits = tf.matmul(decoder_output, self.W_projection) + self.b_projection  # logits shape:[batch_size*decoder_sent_length,self.num_classes]==tf.matmul([batch_size*decoder_sent_length,hidden_size*2],[hidden_size*2,self.num_classes])
            logits=tf.reshape(logits,shape=(self.batch_size,self.decoder_sent_length,self.num_classes)) #logits shape:[batch_size,decoder_sent_length,self.num_classes]
        ###################################################################################################################################
        return logits

    def loss_seq2seq(self):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_label, logits=self.logits)
            #losses:[batch_size,self.decoder_sent_length]
            loss_batch=tf.reduce_sum(losses,axis=1)/self.decoder_sent_length #loss_batch:[batch_size]
            loss=tf.reduce_mean(loss_batch)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss + l2_losses
            return loss

    def gru_cell(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size,embed_size]
        :param h_t_minus_1:[batch_size,embed_size]
        :return:
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,self.U_z) + self.b_z)  # z_t:[batch_size,self.hidden_size]
        # 2.reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,self.U_r) + self.b_r)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t

    def gru_cell_decoder(self, Xt, h_t_minus_1,context_vector):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size,embed_size]
        :param h_t_minus_1:[batch_size,embed_size]
        :param context_vector. [batch_size,embed_size].this represent the result from attention( weighted sum of input during current decoding step)
        :return:
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z_decoder) + tf.matmul(h_t_minus_1,self.U_z_decoder) +tf.matmul(context_vector,self.C_z_decoder)+self.b_z_decoder)  # z_t:[batch_size,self.hidden_size]
        # 2.reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r_decoder) + tf.matmul(h_t_minus_1,self.U_r_decoder) +tf.matmul(context_vector,self.C_r_decoder)+self.b_r_decoder)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h_decoder) +r_t * (tf.matmul(h_t_minus_1, self.U_h_decoder)) +tf.matmul(context_vector, self.C_h_decoder)+ self.b_h_decoder)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t,h_t

    # forward gru for first level: word levels
    def gru_forward(self, embedded_words,gru_cell, reverse=False):
        """
        :param embedded_words:[None,sequence_length, self.embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length,axis=1)  # it is a list,length is sentence_length, each element is [batch_size,1,embed_size]
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]  # it is a list,length is sentence_length, each element is [batch_size,embed_size]
        h_t = tf.ones((self.batch_size,self.hidden_size))
        h_t_list = []
        if reverse:
            embedded_words_squeeze.reverse()
        for time_step, Xt in enumerate(embedded_words_squeeze):  # Xt: [batch_size,embed_size]
            h_t = gru_cell(Xt,h_t) #h_t:[batch_size,embed_size]<------Xt:[batch_size,embed_size];h_t:[batch_size,embed_size]
            h_t_list.append(h_t)
        if reverse:
            h_t_list.reverse()
        return h_t_list  # a list,length is sentence_length, each element is [batch_size,hidden_size]

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("decoder_init_state"):
            self.W_initial_state = tf.get_variable("W_initial_state", shape=[self.hidden_size, self.hidden_size*2], initializer=self.initializer)
            self.b_initial_state = tf.get_variable("b_initial_state", shape=[self.hidden_size*2])
        with tf.name_scope("embedding_projection"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size*2],dtype=tf.float32) #,initializer=self.initializer
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size*2, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        # GRU parameters:update gate related
        with tf.name_scope("gru_weights_encoder"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("gru_weights_decoder"):
            self.W_z_decoder = tf.get_variable("W_z_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.U_z_decoder = tf.get_variable("U_z_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.C_z_decoder = tf.get_variable("C_z_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer) #TODO
            self.b_z_decoder = tf.get_variable("b_z_decoder", shape=[self.hidden_size*2])
            # GRU parameters:reset gate related
            self.W_r_decoder = tf.get_variable("W_r_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.U_r_decoder = tf.get_variable("U_r_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.C_r_decoder = tf.get_variable("C_r_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer) #TODO
            self.b_r_decoder = tf.get_variable("b_r_decoder", shape=[self.hidden_size*2])

            self.W_h_decoder = tf.get_variable("W_h_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)
            self.U_h_decoder = tf.get_variable("U_h_decoder", shape=[self.embed_size*2, self.hidden_size*2], initializer=self.initializer)   #TODO
            self.C_h_decoder = tf.get_variable("C_h_decoder", shape=[self.embed_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.b_h_decoder = tf.get_variable("b_h_decoder", shape=[self.hidden_size*2])

        with tf.name_scope("full_connected"):
            self.W_fc=tf.get_variable("W_fc",shape=[self.hidden_size*2,self.hidden_size])
            self.a_fc=tf.get_variable("a_fc",shape=[self.hidden_size])


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




