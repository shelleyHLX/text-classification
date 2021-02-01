#-*-coding:utf-8-*-
import keras
import os
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras_bert import build_model_from_config
from keras.callbacks import Callback
import sys
from data_pre import get_sen_final, get_tag_dict
from data_generator_bert import data_generator_bert

sys.path.append('../../utils')
from utils import evaluator as evale
from tqdm import tqdm
import time

batch_size = 8
model_path = 'save/base_best.h5'
config_path = 'bert/bert_config.json'

tag2id, _ = get_tag_dict()
test_data = get_sen_final()
test_generator = data_generator_bert(test_data, tag2id, batch_size=16)

bert_model, _ = build_model_from_config(config_path)
indices = Input(shape=(None,))
segments = Input(shape=(None,))
x = bert_model([indices, segments])

x = Lambda(lambda x: x[:, -1, :])(x)
x = Dense(4096, activation=None)(x)
p = Dense(202, activation='sigmoid')(x)
model = Model([indices, segments], p)

model.summary()

model.load_weights(model_path)
dict_path = 'bert/vocab.txt'


def evaluate():
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
    return score12, TP, FP, FN

f1, TP, FP, FN = evaluate()
print(f1)