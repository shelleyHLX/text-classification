# coding: utf-8

import codecs
import json
import pickle
import numpy as np
from multiprocessing import Pool

save_path = './'
y_file_test = 'y_test.npy'
y_file_valid = 'y_valid.npy'
y_file_train = 'y_train.npy'

raw_file_test = "data_test.json"
raw_file_valid = 'data_valid.json'
raw_file_train = 'bigdata_delete_smalltestvalid.json'

file_fact_test_dict = 'idfact_dict_test.pkl'
file_fact_valid_dict = 'idfact_dict_valid.pkl'
file_fact_train_dict = 'idfact_dict_train.pkl'


def read_data(raw_file, file_y, file_fact_dict):
    i = 0
    facts_dict = {}

    f_lines = codecs.open(raw_file, 'r', 'utf-8').readlines()
    accusation_all = []
    relevant_articles_all = []
    death_penalty_all = []
    imprisonment_all = []
    life_imprisonment_all = []
    print(len(f_lines))  # 748203

    for line in f_lines:
        if i % 1000 == 0:
            print(i)
        # print(type(line))  # str
        case = json.loads(line)
        fact_str = case['fact']
        criminals = case['meta']['criminals']
        fact_str = fact_str.replace(criminals[0], ',')
        facts_dict[i] = fact_str

        # accu
        accusation = case['meta']['accusation']
        accusation_all.append(accusation)
        # arti
        relevant_articles = []
        for articles in case['meta']['relevant_articles']:
            relevant_articles.append(int(articles))
        relevant_articles_all.append(relevant_articles)
        death_penalty = case['meta']['term_of_imprisonment']['death_penalty']
        death_penalty_all.append(death_penalty)
        imprisonment = case['meta']['term_of_imprisonment']['imprisonment']
        imprisonment_all.append(imprisonment)
        life_imprisonment = case['meta']['term_of_imprisonment']['life_imprisonment']
        life_imprisonment_all.append(life_imprisonment)

        i += 1
        # if i == 50:
        #     break
    # p = Pool()

    # accu
    accusation_all = np.asarray(accusation_all)
    # accu_id = np.asarray(list(p.map(get_id4accus, accusation_all)))
    print('accusation_all', accusation_all[0:5])

    # relevant_articles
    relevant_articles_all = np.asarray(relevant_articles_all)
    # rel_id = np.asarray(list(p.map(get_id4laws, relevant_articles_all)))
    print('relevant_articles_all ', relevant_articles_all[0:5])
    # death_penalty
    print('death_penalty_all', death_penalty_all[0:5])
    # imprisonment
    print('imprisonment_all', imprisonment_all[0:5])
    # life_imprisonment
    print('life_imprisonment_all', life_imprisonment_all[0:5])
    # save all
    labels = [accusation_all, relevant_articles_all, death_penalty_all, imprisonment_all, life_imprisonment_all]
    np.save(save_path+file_y, labels)
    print('save to ', save_path+file_y)

    print('save facts_dict', save_path + file_fact_dict)
    with open(save_path + file_fact_dict, 'wb') as outp:
        pickle.dump(facts_dict, outp)


if __name__ == '__main__':
    # read_data(raw_file_valid, y_file_valid, file_fact_valid_dict)
    # read_data(raw_file_test, y_file_test, file_fact_test_dict)
    read_data(raw_file_train, y_file_train, file_fact_train_dict)

"""

"""