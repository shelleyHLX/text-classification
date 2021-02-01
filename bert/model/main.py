#-*-coding:utf-8-*-
import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from keras_bert import build_model_from_config
from keras.callbacks import Callback
from keras_bert import Tokenizer
import numpy as np
import sys
import json

batch_size = 8
model_path = 'save/'
config_path = 'bert/bert_config.json'

bert_model, _ = build_model_from_config(config_path)
indices = Input(shape=(None,))
segments = Input(shape=(None,))
x = bert_model([indices, segments])
print(x.shape)
x = Lambda(lambda x: x[:,-1,:])(x)
x = Dense(2048, activation=None)(x)
p = Dense(20, activation='sigmoid')(x)
model = Model([indices, segments], p)
model.summary()
dict_path = 'bert/vocab.txt'

def load_dict(file):
    char_dict = {}
    with open(file,'r',encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            char_dict[line.strip()] = i
    return char_dict

bert_dict = load_dict(dict_path)
tokenizer = Tokenizer(bert_dict)

data_types = ['divorce', 'labor', 'loan']

    
def get_tag_dict(file):
    tag2id, id2tag = {}, {}
    with open(file,'r',encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            tag2id[line.strip()] = i
            id2tag[i] = line.strip()
    return tag2id, id2tag
    
def seq_padding(X, maxlen=None):
    if maxlen:
        return [x + [0] * (maxlen - len(x)) if len(x)<maxlen else x[:maxlen] for x in X]
    else:
        L = [len(x) for x in X]
        ML = max(L)
        return [x + [0] * (ML - len(x)) for x in X]
        


for data_type in data_types:
    input_tag2id = 'input/'+data_type+'/tags.txt'
    #input_file = '../../data_small/data_raw/'+data_type+'/data_small_selected.json'
    #output_file = 'output/'+data_type+'_output.json'
    input_file = '/input/' + data_type +'/input.json'
    output_file = '/ouput/' + data_type +'/ouput.json'
    model_save = model_path+data_type+'_best.ckpt'

    tag2id, id2tag = get_tag_dict(input_tag2id)
    model.load_weights(model_save)

    output = open(output_file,'w',encoding='utf-8')
    with open(input_file,'r',encoding='utf-8')as f:
        for lines in f.readlines():
            line = json.loads(lines)
            X1, X2, text, result_doc = [], [], [], []
            for sentence in line:
                x1,x2 = tokenizer.encode(first=sentence['sentence'][:350])
                X1.append(x1)
                X2.append(x2)
                text.append(sentence['sentence'])
                if len(X1)==batch_size or sentence==line[-1]:
                    X1 = np.array(seq_padding(X1))
                    X2 = np.array(seq_padding(X2))
                    y = model.predict([X1,X2])
                    for i in range(len(y)):
                        label = []
                        for j in range(20):
                            if y[i,j]>0.5:
                                label.append(id2tag.get(j).strip())
                        result_doc.append({'labels': label, 'sentence': text[i]})
                    X1,X2,text = [],[],[]
            output.write(json.dumps(result_doc, ensure_ascii=False))
            output.write('\n')
    output.close()
