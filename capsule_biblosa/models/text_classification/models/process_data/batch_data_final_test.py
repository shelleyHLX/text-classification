# coding: utf-8

from __future__ import division
from __future__ import print_function

import numpy as np
from multiprocessing import Pool
import os
import pickle
from data_helper import train_batch

# w
batch_test_path = '../data/first_stage/test/'

if not os.path.exists(batch_test_path):
    os.makedirs(batch_test_path)

save_path = '../data/'

# r
file_word2id_te = 'fact2word_first_stage_test.npy'
y_file_test = 'first_stage_test_accu.npy'

batch_size = 128

embedding_size = 256
embedding_file_name = save_path + 'sr_word2id_' + str(256) + '.pkl'
with open(embedding_file_name, 'rb') as inp:
    sr_word2id = pickle.load(inp)
dict_word2id = dict()

for i in range(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]

def get_idword(word):
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]

def get_id4words(words):
    ids = list(map(get_idword, words))
    return ids


with open(save_path + 'accu_id.pkl', 'rb') as inp:
    dic_accu_id = pickle.load(inp)


def get_idaccu(accu):
    """获取 accu 所对应的 id."""
    if accu not in dic_accu_id:
        return 1
    else:
        return dic_accu_id[accu]

def get_id4accus(accus):
    """把 accus 转为 对应的 id"""
    ids = list(map(get_idaccu, accus))  # 获取id
    return ids


def pad_X200_same(words, max_len=200):
    words_len = len(words)
    words = words[0:words_len-1]
    words_len = len(words)

    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[(words_len - max_len):]
    if words_len < max_len:
        num = int(max_len / words_len)
        words_num = words * num
        words_num = np.asarray(words_num)
        end_idx = max_len - words_len * num - 1
        words_last = words[-1:-end_idx-2:-1]
        words_last = np.asarray(words_last)
        return np.hstack([words_last, words_num])


def valid_test_get_batch(wd_fact2id_path, y_path, batch_path):
    print('loading facts and ys.',
          save_path + wd_fact2id_path,
          save_path + y_path)
    facts = np.load(save_path + wd_fact2id_path)
    print(facts[0:2])
    p = Pool()
    word2id = np.asarray(list(p.map(get_id4words, facts)))

    y = np.load(save_path + y_path)
    print(y[0:10])
    y_id = np.asarray(list(p.map(get_id4accus, y)))
    print(y_id[0:50])

    p = Pool()
    X = np.asarray(list(p.map(pad_X200_same, word2id)), dtype=np.int64)
    print(X[0:2])

    train_batch(X, y_id, batch_path, batch_size)


if __name__ == '__main__':
    valid_test_get_batch(file_word2id_te, y_file_test, batch_test_path)

