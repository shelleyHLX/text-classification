# coding: utf-8

from models.biblosa.configs import cfg
import tensorflow as tf
from abc import ABCMeta, abstractmethod


class ModelTemplate(metaclass=ABCMeta):  # token_num, char_num, max_lens --- tds, cds, tl,
    def __init__(self, W_embedding, scope):
        self.scope = scope
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.W_embedding = W_embedding

        # ---- place holder -----
        self.token_seq = tf.placeholder(tf.int32, [None, cfg.fact_len], name='token_seq')

        self.gold_label = tf.placeholder(tf.float64, [None, cfg.n_class], name='gold_label')  # bs
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        self.saver = tf.train.Saver(max_to_keep=1)

        # ------------ other ---------
        self.token_mask = tf.cast(self.token_seq, tf.bool)
        # self.token_len = tf.reduce_sum(tf.cast(self.token_mask, tf.int32), -1)
        self.tensor_dict = {}

    @abstractmethod
    def build_network(self):
        pass

    def build_loss(self):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gold_label, logits=self.logits))
        return loss

    def build_accuracy(self):
        return tf.sigmoid(self.logits)

    def update_tensor_add_ema_and_opt(self):
        self.logits = self.build_network()
        self.loss = self.build_loss()
        self.y_pred = self.build_accuracy()
