
import tensorflow as tf
import numpy as np
from models.Attention_Textcnn_ELMo.bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

def get_batch(data_path, batch_id):
    new_batch = np.load(data_path + str(batch_id) + '.npz')
    X_batch = new_batch['X']
    acc = new_batch['acc']
    # law = new_batch['law']
    # death = new_batch['death']
    # imp = new_batch['imp']
    # lif = new_batch['lif']
    return [X_batch, acc,]  # law, death, imp, lif]


def get_elmo(X_batch, bilm, i):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        elmo_context_vector = sess.run(
            elmo_context_input['weighted_op'],
            feed_dict={context_token_ids: X_batch}
        )
    return elmo_context_vector


if __name__ == '__main__':
    vocab_file = './bilm_trained/vocab_elmo.txt'
    options_file = './bilm_trained/options.json'  # try/options.json
    weight_file = './bilm_trained//weights.hdf5'
    token_embedding_file = './bilm_trained/8361_vocab_embedding.hdf5'
    data_path = '../../data/s200_wv256_bs128/train/'

    i = 0
    [X_batch, y_batch] = get_batch(data_path, '1')
    print(X_batch.shape)

    bilm = BidirectionalLanguageModel(options_file, weight_file, use_character_inputs=False,
                                      embedding_weight_file=token_embedding_file)
    context_token_ids = tf.placeholder('int32', shape=(None, None))
    context_embeddings_op = bilm(context_token_ids)
    elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
    elmo_context_vector = get_elmo(X_batch, bilm, i)
    i = i + 1
    print('elmo1', elmo_context_vector.shape)

    [X_batch, y_batch] = get_batch(data_path, '2')
    print(X_batch.shape)
    elmo_context_vector = get_elmo(X_batch, bilm, i)
    print('elmo2', elmo_context_vector.shape)


"""
各自的graph调用
"""
