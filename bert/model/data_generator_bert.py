#-*-coding:utf-8-*-
import random
import sys
sys.path.append('../utils')
from utils import seq_padding
import numpy as np
from keras_bert import Tokenizer
import codecs

dict_path = '/home/lb/FYB2019/chinese_L-12_H-768_A-12/vocab.txt'


def load_dict(file):
    char_dict = {}
    with codecs.open(file) as f:
        for i, line in enumerate(f.readlines()):
            char_dict[line.strip()] = i
    return char_dict


bert_dict = load_dict(dict_path)
tokenizer = Tokenizer(bert_dict)  # 空格被自动删去　注意了


class data_generator_bert(object):
    def __init__(self, data, tag2id, pad_size=500, batch_size=8):
        self.data = data
        self.batch_size = batch_size
        self.tag2id = tag2id
        self.pad_size = pad_size

    def __len__(self):
        return len(self.data)//self.batch_size
    def __iter__(self):
        train_data = self.data
        while True:
            idxs = [i for i in range(len(train_data))]
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                #print(i)
                d = train_data[i]
                text = d[0][:self.pad_size]
                x1, x2 = tokenizer.encode(first=text)
                X1.append(x1)
                X2.append(x2)
                label = [0] * 202
                for p in d[1]:
                    label[self.tag2id[p]] = 1
                Y.append(label)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = np.array(seq_padding(X1))
                    X2 = np.array(seq_padding(X2))
                    Y = np.array(seq_padding(Y))
                    yield [X1, X2], Y
                    X1, X2, Y = [], [], []
    def __get_testbatch__(self,  num):
        start_idx = num*self.batch_size
        end_idx = [start_idx+self.batch_size, len(self.data)][start_idx+self.batch_size > len(self.data)]
        idxs = [start_idx + i for i in range(end_idx - start_idx)]
        X1, X2, Y = [], [], []
        for i in idxs:
            d = self.data[i]
            text = d[0][:self.pad_size]
            x1, x2 = tokenizer.encode(first=text)
            X1.append(x1)
            X2.append(x2)
            label = [0] * 202
            for p in d[1]:
                label[self.tag2id[p]] = 1
            Y.append(label)
        #print(X1)
        X1 = np.array(seq_padding(X1))
        X2 = np.array(seq_padding(X2))
        Y = np.array(seq_padding(Y))
        return [X1, X2], Y
