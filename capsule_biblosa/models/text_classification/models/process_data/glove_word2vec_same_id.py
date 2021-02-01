
import numpy as np
import codecs

# def read_word2vector(filename):
#     read_file = open(filename)
#     read_file.readline()
#     word2vec_dict = {}
#     for line in read_file.readlines():
#         split = line.split()
#         word = split[0]
#         vector = split[1:]
#         # if len(vector) != 256:
#         #     print(line)
#         #     continue
#         word2vec_dict[word]=vector
#     read_file.close()
#     return word2vec_dict

def read_embed(filename):
    read_file = open(filename)
    read_file.readline()
    words = []
    vectors = []
    embed_dict = {}
    for line in read_file.readlines():
        split = line.split()
        words.append(split[0])
        vectors.append(split[1:])
        embed_dict[split[0]] = split[1:]
    return words, vectors, embed_dict



if __name__ == '__main__':
    w2v_w, w2v_v, w2v_dict = read_embed('word_embedding_100.txt')
    glove_w, glove_v, glove_dict = read_embed('glove_100_format.txt')
    print('------------------')
    glove_new_vectors = []
    glove_new_words = []
    write_glove = codecs.open('glove_same_id.txt', "a", 'utf-8')
    write_glove.write(str(len(w2v_w)) + ' ' + str(100) + '\n')
    for i in range(len(w2v_w)):
        if i % 1000==0:
            print(i)
        word = w2v_w[i]
        if word in glove_w:
            vector = glove_dict[word]
        else:
            vector = np.random.randn(1, 100)
            vector = [str(i) for i in vector]
        vector_str = ' '.join(vector)
        line = word + ' ' + vector_str
        write_glove.write(line + '\n')
    write_glove.close()


