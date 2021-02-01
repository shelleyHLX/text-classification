# coding: utf-8

import numpy as np
import codecs
from multiprocessing import Pool
import pickle

batch_size = 128
embedding_size = 256

# read_words = '../data_raw_final/first_stage/fact2word_te.npy'
# read_ys = '../data_raw_final/first_stage/y_te.npy'
# batch_path = '../data_raw_final/first_stage/test_s496/'
# batch_path = '../data_raw_final/first_stage/test_s200/'
# batch_path = '../data_raw_final/first_stage/test_0200/'

read_words = './fact2word_te.npy'
read_ys = './y_te.npy'
# batch_path = '../data_raw_final/restData/rest_s200/'  # 修改 batch id
# batch_path = './SF200B300_wv256_bs128/'  # 修改 batch id
batch_path = './test_p0496_wv256_bs128/'  # 修改 batch id
# batch_path = './s200_wv256_bs64/'  # 修改 batch id

###########################################################
# read_words_te = './fact2word_test.npy'
# read_ys_te = './y_test.npy'
#
# read_words_tr = './fact2word_train.npy'
# read_ys_tr = './y_train.npy'
#
# read_words_va = './fact2word_valid.npy'
# read_ys_va = './y_valid.npy'
#
# batch_path_te = './SF200B300_wv256_bs128/test/'
# batch_path_tr = './SF200B300_wv256_bs128/train/'
# batch_path_va = './SF200B300_wv256_bs128/valid/'
###########################################################

accu_id_dict = dict()
i = 0
rf = codecs.open('./accu.txt', 'r', 'utf-8')
for line in rf.readlines():
    line = line.replace('\n', '').replace('\r', '')
    accu_id_dict[line] = i
    i += 1
print(accu_id_dict)

def get_idaccu(accu):
    """获取 accu 所对应的 id."""
    if accu not in accu_id_dict:
        return 1
    else:
        return accu_id_dict[accu]

def get_id4accus(accus):
    """把 accus 转为 对应的 id"""
    ids = list(map(get_idaccu, accus))  # 获取id
    return ids
###########################################################
law_id_dict = dict()
i = 0
rf = codecs.open('./law.txt', 'r', 'utf-8')
for line in rf.readlines():
    line = int(line.strip('\n'))
    law_id_dict[line] = i
    i += 1
print(law_id_dict)
def get_idlaw(law):
    """获取 law 所对应的 id."""
    if law not in law_id_dict:
        print('law ont in dic law id')
        exit(0)
    else:
        return law_id_dict[law]

def get_id4laws(laws):
    """把 laws 转为 对应的 id"""
    ids = list(map(get_idlaw, laws))  # 获取id
    return ids
###########################################################
embedding_file_name = '../../data/' + 'sr_word2id_' + str(embedding_size) + '.pkl'
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

###########################################################
###########################################################
###########################################################

true_dict = {True:1, False:0}
def get_impid(imp):
    return true_dict[imp]

def pad_X200_same(words, max_len=200):
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


def pad_X500_same(words, max_len=500):
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


def pad_X496_same(words, max_len=496):
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


def pad_X496_0(words, max_len=496):
    words_len = len(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[(words_len - max_len):]
    if words_len < max_len:
        words_last = np.zeros(shape=max_len - words_len, dtype=np.int32)
        return np.hstack([words, words_last])


def pad_X500_S_F200B200(words, max_len=400):
    words_len = len(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        F = words[0:200]
        B = words[words_len-200:words_len]
        return np.hstack([F, B])
    if words_len < max_len:
        num = int(max_len / words_len)
        words_num = words * num
        words_num = np.asarray(words_num)
        end_idx = max_len - words_len * num - 1
        words_last = words[-1:-end_idx-2:-1]
        words_last = np.asarray(words_last)
        return np.hstack([words_last, words_num])


def id_pad_length_over_sample(file_words, file_ys, batch_path, batch_num):
    p = Pool()
    words = np.load(file_words)
    accuwords, relewords, death, impr, life = np.load(file_ys)
    rel_id = np.asarray(list(p.map(get_id4laws, relewords)))
    acc_id = np.asarray(list(p.map(get_id4accus, accuwords)))
    word2id = np.asarray(list(p.map(get_id4words, words)))
    death_id = np.asarray(list(p.map(get_impid, death)))
    life_id = np.asarray(list(p.map(get_impid, life)))
    del words, accuwords, relewords, death, life
    print(rel_id[0])
    print(acc_id[0])
    print(word2id[0])
    print(death_id[0])
    print(life_id[0])
    sample_num = len(death_id)
    lengths = []
    for sample in word2id:
        lengt = len(sample)
        if lengt>16*16:
            lengths.append(16*16)
        else:
            lengths.append(lengt)

    lengths = np.asarray(lengths)
    new_index = np.random.permutation(sample_num)
    word2id = word2id[new_index]
    rel_id = rel_id[new_index]
    acc_id = acc_id[new_index]
    death_id = death_id[new_index]
    impr = impr[new_index]
    life_id = life_id[new_index]
    lengths = lengths[new_index]
    for start in list(range(0, sample_num, batch_size)):
        print(batch_num)
        end = min(start + batch_size, sample_num)
        batch_name = batch_path + str(batch_num) + '.npz'
        X_batch = word2id[start:end]
        # X_batch = np.asarray(list(p.map(pad_X496_same, X_batch)), dtype=np.int64)
        X_batch = np.asarray(list(p.map(pad_X496_0, X_batch)), dtype=np.int64)
        # X_batch = np.asarray(list(p.map(pad_X500_S_F200B200, X_batch)), dtype=np.int64)
        if batch_num == 0:
            print(len(X_batch[0]))
        acc_batch = acc_id[start:end]
        law_batch = rel_id[start:end]
        death_batch = death_id[start:end]
        imp_batch = impr[start:end]
        lif_batch = life_id[start:end]
        lengths_batch = lengths[start:end]
        np.savez(batch_name, X=X_batch, length=lengths_batch,
                 acc=acc_batch, law=law_batch, death=death_batch, imp=imp_batch, lif=lif_batch)
        batch_num += 1


if __name__ == '__main__':
    id_pad_length_over_sample(read_words, read_ys, batch_path, batch_num=0)
    # id_pad_length_over_sample(read_words, read_ys, batch_path, batch_num=0)
    ####################################################################
    # id_pad_length_over_sample(read_words_te, read_ys_te, batch_path_te, batch_num=0)
    # id_pad_length_over_sample(read_words_va, read_ys_va, batch_path_va, batch_num=0)
    # id_pad_length_over_sample(read_words_tr, read_ys_tr, batch_path_tr, batch_num=0)

"""
bs 128 batch rest: 5846
bs 128 batch train: 12978

bs 64 batch rest: 
bs 64 batch train: 25956
"""
