#-*-coding:utf-8-*-
import keras
import os
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras_bert import load_trained_model_from_checkpoint
from keras.callbacks import Callback
import sys
from data_pre import get_sen_train, get_sen_test, get_tag_dict
from data_generator_bert import data_generator_bert

sys.path.append('../../utils')
from utils import evaluator as evale
from tqdm import tqdm
import time



train_data = get_sen_train()
test_data = get_sen_test()
tag2id, _ = get_tag_dict()
save_path = 'save/'
model_path = save_path + 'base_best.h5'
log_path = save_path + 'base_log.log'

if not os.path.exists(save_path):
    os.makedirs(save_path)
train_generator = data_generator_bert(train_data, tag2id, batch_size=16)
test_generator = data_generator_bert(test_data, tag2id, batch_size=16)

log_write = open(log_path, 'w')

config_path = '/home/lb/FYB2019/chinese_L-12_H-768_A-12/bert_config.json'
cpkt_path = '/home/lb/FYB2019/chinese_L-12_H-768_A-12/bert_model.ckpt'

# config_path = '/home/lb/FYB2019/bert_wwm/bert_wwm_ext/bert_config.json'
# cpkt_path = '/home/lb/FYB2019/bert_wwm/bert_wwm_ext/bert_model.ckpt'

bert_model = load_trained_model_from_checkpoint(config_path, cpkt_path)
for l in bert_model.layers:
    l.trainable = True

indices = Input(shape=(None,))
segments = Input(shape=(None,))
x = bert_model([indices, segments])

x = Lambda(lambda x: x[:, -1, :])(x)
x = Dense(4096, activation=None)(x)
p = Dense(202, activation='sigmoid')(x)
model = Model([indices, segments], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
)
model.summary()


class Evaluate(Callback):
    def __init__(self, model):
        self.best = 0.
        self.model = model

    def on_epoch_end(self, epoch, logs):
        f1, TP, FP, FN = self.evaluate()
        print('now:', f1)
        if f1 > self.best:
            self.best = f1
            model.save_weights(model_path)
        print('best:', self.best)

    def evaluate(self):
        predict_labels_all, marked_labels_all = [], []
        for i in tqdm(range(test_generator.__len__())):
            X_batch, y_batch = test_generator.__get_testbatch__(i)
            predict_labels = model.predict(X_batch)
            predict_labels_all.append(predict_labels)
            marked_labels_all.append(y_batch)
        marked_labels_all = np.concatenate(marked_labels_all, 0)
        predict_labels_all = np.concatenate(predict_labels_all, 0)
        # print(predict_labels_all)
        f1_micro, f1_macro, score12, TP, FP, FN = evale(marked_labels_all, predict_labels_all, True, False)
        print('f1-micro:%f, f1-macro:%f, score12:%f\n' % (f1_micro, f1_macro, score12))
        log_write.write('f1-micro:%f, f1-macro:%f, score12:%f\n' % (f1_micro, f1_macro, score12))
        return score12, TP, FP, FN


evaluator = Evaluate(model)
model.fit_generator(train_generator.__iter__(),
                    steps_per_epoch=train_generator.__len__(),
                    epochs=10,
                    callbacks=[evaluator]
                    )
log_write.close()