# coding: utf-8

import numpy as np
import pickle
import codecs

save_path = '../../data/'

embedding_file_name = save_path + 'sr_word2id_' + str(256) + '.pkl'
with open(embedding_file_name, 'rb') as inp:
    sr_word2id = pickle.load(inp)
dict_id2word = dict()

for i in range(len(sr_word2id)):
    dict_id2word[sr_word2id.values[i]] =  sr_word2id.index[i]

def get_idword(id):
    if id not in dict_id2word:
        return 'UNK'
    else:
        return dict_id2word[id]

# ========================================================
with open(save_path + 'id_accu.pkl', 'rb') as inp:
    id_accu = pickle.load(inp)

# print(wrong_ids)
# print(len(wrong_ids))

def id2word():
    wrong_ids = np.load('wrong_ids.npy')
    with open(save_path + 'idfact_dict_te.pkl', 'rb') as inpp:
        idfact_dict = pickle.load(inpp)
    fact2word = np.load(save_path + 'fact2word_te.npy')
    wrong_fact2words = []
    wrong_facts = []
    for wr in wrong_ids:
        wrong_fact2words.append(fact2word[wr])
        wrong_facts.append(idfact_dict[wr])
    w_file = codecs.open('wrong_fact_labels.txt', 'a', 'utf-8')
    # acc
    wrong_mar_labels = np.load('wrongmark_labelsid.npy')
    wrong_pre_labels = np.load('wrongpre_labelsid.npy')
    accus_mar = []
    accus_pre = []
    for i in range(len(wrong_mar_labels)):
        mar_label = wrong_mar_labels[i]
        pre_label = wrong_pre_labels[i]
        accu_mar = []
        accu_pre = []
        for i in mar_label:
            accu = id_accu[i]
            accu_mar.append(accu)
        for i in pre_label:
            accu = id_accu[i]
            accu_pre.append(accu)
        accus_mar.append(accu_mar)
        accus_pre.append(accu_pre)

    for i in range(len(wrong_ids)):
        fact2wordstr = ' '.join(wrong_fact2words[i])
        w_file.write(str(wrong_ids[i]) + '\n')
        w_file.write(fact2wordstr + '\n')
        w_file.write(wrong_facts[i] + '\n')
        accu_m_str = ' '.join(accus_mar[i])
        accu_p_str = ' '.join(accus_pre[i])
        w_file.write('marked label: ' + accu_m_str + '\t' + 'predicted label: ' + accu_p_str + '\n')
    w_file.close()


if __name__ == '__main__':
    id2word()

