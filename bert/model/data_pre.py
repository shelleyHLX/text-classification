#-*-coding:utf-8-*-
import json


def get_sen_train():
    data = []
    with open('../cail2018_all/first_stage/train.json', 'r')as f:
        for line in f.readlines():
            facts = json.loads(line)
            data.append([facts['fact'], facts['meta']['accusation']])
    return data


def get_sen_test():
    data = []
    with open('../cail2018_all/first_stage/test.json', 'r')as f:
        for line in f.readlines():
            facts = json.loads(line)
            data.append([facts['fact'], facts['meta']['accusation']])
    return data


def get_sen_final():
    data = []
    with open('../cail2018_all/final_test.json', 'r')as f:
        for line in f.readlines():
            facts = json.loads(line)
            data.append([facts['fact'], facts['meta']['accusation']])
    return data


def get_tag_dict():
    tag2id, id2tag = {}, {}
    with open('../cail2018_all/meta/accu.txt') as f:
        for i, line in enumerate(f.readlines()):
            tag2id[line.strip()] = i
            id2tag[i] = line.strip()
    return tag2id, id2tag