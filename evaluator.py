# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np
import codecs

def sigmoid(X):
    sig = [1.0 / float(1.0 + np.exp(-x)) for x in X]
    return sig

def to_categorical_one_sample(cl, n_class):
    y = np.zeros(n_class, dtype=np.int32)

    for i in range(len(cl)):
        y[cl[i]] = 1
    return y


def cail_imprisonment_evaluator(dea_pre, dea_mar, imp_pre, imp_mar, lif_pre, lif_mar):
    print('imprisonment')
    samples = len(dea_pre)
    print('num of sampls: ', samples)
    # print('imp_pre: ', np.array(imp_pre))
    impris_pred = []
    impris_mark = []
    # print('dea ', np.asarray(dea_pre).shape)
    # print('imp ', np.asarray(imp_pre).shape)
    # print('lif ', np.asarray(lif_pre).shape)
    for i in range(samples):
        # predict
        # print(i)
        death_pred = np.argmax(dea_pre[i])
        life_pred = np.argmax(lif_pre[i])
        # print('lif_pre[i] ', lif_pre[i])
        # print('life')
        if death_pred == 1:
            imprisonment = -2
        elif life_pred == 1:
            imprisonment = -1
        else:
            # print(imp_pre[i])
            imprisonment = round(imp_pre[i])
        impris_pred.append(imprisonment)
        # print('mark')
        # mark
        if dea_mar[i]:
            im_mark = 1
        elif lif_mar[i]:
            im_mark = 1
        else:
            im_mark = imp_mar[i]
        impris_mark.append(im_mark)
    # predict labels category
    # print('num of samples: ', samples)
    sc_sum = 0.0
    for i in range(samples):
        pred = impris_pred[i]
        mark = impris_mark[i]
        v = abs(np.log(mark + 1) - np.log(pred + 1))
        if v <= 0.2:
            sc = 1.0
        elif v <= 0.4:
            sc = 0.8
        elif v <= 0.6:
            sc = 0.6
        elif v <= 0.8:
            sc = 0.4
        elif v <= 1.0:
            sc = 0.2
        else:
            sc = 0.0
        sc_sum = sc_sum + sc * 1.0
    score = sc_sum / samples * 1.0

    return score


def cail_evaluator(predict_labels_list, marked_labels_list):
    # predict labels category
    predict_labels_category = []
    samples = len(predict_labels_list)
    print('num of samples: ', samples)
    # pred = codecs.open('pred' + str(samples) + '.txt', 'a', 'utf-8')
    # mark = codecs.open('mark' + str(samples) + '.txt', 'a', 'utf-8')
    for samp in range(samples):  # number of samples
        predict_category = [1 if i > 0.5 else 0 for i in predict_labels_list[samp]]
        predict_labels_category.append(predict_category)
    # pred.close()
    # marked labels category
    marked_labels_category = []
    num_class = len(predict_labels_category[0])
    print('num of classes: ', num_class)
    for i in range(samples):
        marked_category = to_categorical_one_sample(marked_labels_list[i], num_class)
        marked_labels_category.append(marked_category)
    # mark.close()
    tp_list = []
    fp_list = []
    fn_list = []
    f1_list = []

    for i in range(num_class):  # 类别个数
        # print(i)
        tp = 0.0  # predict=1, truth=1
        fp = 0.0  # predict=1, truth=0
        fn = 0.0  # predict=0, truth=1
        # 样本个数
        pre = [p[i] for p in predict_labels_category]
        mar = [p[i] for p in marked_labels_category]
        pre = np.asarray(pre)
        mar = np.asarray(mar)

        for i in range(len(pre)):
            if pre[i] == 1 and mar[i] == 1:
                tp += 1
            elif pre[i] == 1 and mar[i] == 0:
                fp += 1
            elif pre[i] == 0 and mar[i] == 1:
                fn += 1
        precision = 0.0
        if tp + fp > 0:
            precision = tp / (tp + fp)
        recall = 0.0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        f1_list.append(f1)

    # micro level
    f1_micro = 0.0
    if sum(tp_list) + sum(fp_list) > 0:
        f1_micro = sum(tp_list) / (sum(tp_list) + sum(fp_list))
    # macro level
    f1_macro = sum(f1_list) / len(f1_list)
    score12 = (f1_macro + f1_micro) / 2.0

    return f1_micro, f1_macro, score12


def cail_evaluator_single_label(predict_labels_list, marked_labels_list):
    # predict labels category
    predict_labels_category = []
    samples = len(predict_labels_list)
    print('num of samples: ', samples)
    for i in range(samples):  # number of samples
        predict_norm = sigmoid(predict_labels_list[i])

        max_sam = max(predict_norm)
        predict_category = [1 if i == max_sam else 0 for i in predict_norm]
        predict_labels_category.append(predict_category)

    marked_labels_category = []
    num_class = len(predict_labels_category[0])
    print('num of classes: ', num_class)
    for i in range(samples):
        marked_category = to_categorical_one_sample(marked_labels_list[i], num_class)
        marked_labels_category.append(marked_category)
    tp_list = []
    fp_list = []
    fn_list = []
    f1_list = []

    for i in range(num_class):  # 类别个数
        # print(i)
        tp = 0.0  # predict=1, truth=1
        fp = 0.0  # predict=1, truth=0
        fn = 0.0  # predict=0, truth=1
        # 样本个数
        pre = [p[i] for p in predict_labels_category]
        mar = [p[i] for p in marked_labels_category]
        pre = np.asarray(pre)
        mar = np.asarray(mar)

        for i in range(len(pre)):
            if pre[i] == 1 and mar[i] == 1:
                tp += 1
            elif pre[i] == 1 and mar[i] == 0:
                fp += 1
            elif pre[i] == 0 and mar[i] == 1:
                fn += 1
        precision = 0.0
        if tp + fp > 0:
            precision = tp / (tp + fp)
        recall = 0.0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        f1_list.append(f1)
        # print('tp: %s, fp: %s, fn:%s, f1:%s' %(tp, fp, fn, f1))

    # micro level
    f1_micro = 0.0
    if sum(tp_list) + sum(fp_list) > 0:
        f1_micro = sum(tp_list) / (sum(tp_list) + sum(fp_list))
    # macro level
    f1_macro = sum(f1_list) / len(f1_list)
    score12 = (f1_macro + f1_micro) / 2.0

    return f1_micro, f1_macro, score12


if __name__ == '__main__':
    a = [4, 3, 7, 33]
    ma = max(a)
    print(ma)


