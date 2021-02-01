#-*-coding:utf-8-*-
import logging
import numpy as np


def load_dict(name):
    tag2id, id2tag = {}
    with open(name) as f:
        for i, line in f.readlines():
            tag2id[line.strip()] = i
            id2tag[i] = line.strip()
    return tag2id, id2tag


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def evaluator(y_true, y_pred, need_num = False, sig = True):
    if sig:
        y_pred = sigmoid(y_pred)
    y_pred = y_pred > 0.5
    num_class = np.size(y_true, 1)
    TP, FP, FN = [0]*num_class, [0]*num_class, [0]*num_class
    for i in range(np.size(y_true, 0)):
        #print(y_pred)
        #print(y_true)
        #print("————————————————")
        for j in range(num_class):
            if y_true[i,j]==1 and y_pred[i,j]==1:
                TP[j] += 1
            elif y_true[i,j]==0 and y_pred[i,j]==1:
                FP[j] += 1
            elif y_true[i,j]==1 and y_pred[i,j]==0:
                FN[j] += 1
    P_micro = (sum(TP)+0.001)/(sum(TP)+sum(FP)+0.001)
    R_micro = (sum(TP)+0.001)/(sum(TP)+sum(FN)+0.001)
    F1_micro = 2*P_micro*R_micro/(P_micro+R_micro)
    f1 = []
    for i in range(num_class):
        f1.append((2*TP[i]+0.001)/(2*TP[i]+FP[i]+FN[i]+0.001))
    F1_macro = sum(f1)/num_class
    if need_num:
        return F1_micro, F1_macro, (F1_micro+F1_macro)/2, TP, FP, FN
    else:
        return F1_micro, F1_macro, (F1_micro+F1_macro)/2


def seq_padding(X, maxlen=None,padnum=0):
    if maxlen:
        return [x + [padnum] * (maxlen - len(x)) if len(x)<maxlen else x[:maxlen] for x in X]
    else:
        L = [len(x) for x in X]
        ML = max(L)
        return [x + [padnum] * (ML - len(x)) for x in X]
