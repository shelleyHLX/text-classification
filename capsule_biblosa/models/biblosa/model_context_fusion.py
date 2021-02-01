
from models.biblosa.configs import cfg
from models.biblosa.utils_biblosa.record_log import _logger
import tensorflow as tf

from models.biblosa.model_template import ModelTemplate
from models.biblosa.nn_utils.nn import linear
from models.biblosa.nn_utils.baselines.interface import sentence_encoding_models


class ModelContextFusion(ModelTemplate):
    def __init__(self, W_embedding, scope):
        super(ModelContextFusion, self).__init__(W_embedding, scope)
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        with tf.variable_scope('emb'):
            emb = tf.nn.embedding_lookup(self.W_embedding, self.token_seq)  # bs,sl1,tel

        with tf.variable_scope('sent_encoding'):
            rep = sentence_encoding_models(
                emb, self.token_mask, cfg.context_fusion_method, 'relu',
                'ct_based_sent2vec', cfg.wd, self.is_train, cfg.dropout, block_len=cfg.block_len)
        print('emb ', emb.shape)  # (?, 200, 256)
        print('rep ', rep.shape)  # (?, 512)
        # exit(0)
        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(linear([rep], cfg.hidden_units_num, True, scope='pre_logits_linear',
                                           wd=cfg.wd, input_keep_prob=cfg.dropout,
                                           is_train=self.is_train))  # bs, hn
            logits = linear([pre_logits], cfg.n_class, False, scope='get_output',
                            wd=cfg.wd, input_keep_prob=cfg.dropout, is_train=self.is_train)  # bs, 5
        _logger.done()
        return logits