
import numpy as np
import codecs

def read_word2vector(filename):
    read_file = open(filename)
    read_file.readline()
    word2vec_dict = {}
    for line in read_file.readlines():
        split = line.split()
        word = split[0]
        vector = split[1:]
        # if len(vector) != 256:
        #     print(line)
        #     continue
        word2vec_dict[word]=vector
    read_file.close()
    return word2vec_dict


if __name__ == '__main__':
    word2vec_dict = read_word2vector('word_embedding_100.txt')
    glove_dict = read_word2vector('glove_100_format.txt')
    all_dict = {}
    for key, value in word2vec_dict.items():
        if key in glove_dict:
            glove_vec = glove_dict[key]
        else:
            print('not in glove')
            glove_vec = np.random.randn(1, 100)
            glove_vec = [str(i) for i in glove_vec]
        all_dict[key] = value+glove_vec
    write_file = codecs.open('word2vec100_glove100.txt', 'a', 'utf-8')
    dim = 100 + 100
    write_file.write(str(len(all_dict.keys())) + ' ' + str(dim) + '\n')
    for key, value in all_dict.items():
        vector_str = ' '.join(value)
        line = key + ' ' + vector_str
        write_file.write(line + '\n')
    write_file.close()




