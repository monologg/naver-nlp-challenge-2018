import numpy as np
import pickle
import os

word2vec_dir = 'word2vec'


def load_word_matrix(file_name, vocab, emb_dim, trainable):
    print("load pretrained word matrix from {}".format(file_name))
    # load pickle file
    with open(os.path.join(word2vec_dir, file_name), 'rb') as f:
        data = pickle.load(f)
    word_model = data['word']
    if trainable:
        word_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), emb_dim))
    else:
        word_matrix = np.zeros((len(vocab), emb_dim))
    not_found_cnt = 0
    for word, i in vocab.items():
        try:
            vector = word_model[word]
            word_matrix[i] = np.asarray(vector)
        except:
            not_found_cnt += 1
    print('{} words not in word vector'.format(not_found_cnt))
    # for PAD token, assign zero vector
    word_matrix[0] = np.full(shape=emb_dim, fill_value=0.0, dtype=np.float32)
    print('word_matrix.shape:', word_matrix.shape)
    return word_matrix


def load_char_matrix(file_name, vocab, emb_dim, trainable):
    print("load pretrained char matrix from {}".format(file_name))
    # load pickle file
    with open(os.path.join(word2vec_dir, file_name), 'rb') as f:
        data = pickle.load(f)
    word_model = data['character']
    if trainable:
        char_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), emb_dim))
    else:
        char_matrix = np.zeros((len(vocab), emb_dim))
    not_found_cnt = 0
    for word, i in vocab.items():
        try:
            vector = word_model[word]
            char_matrix[i] = np.asarray(vector)
        except:
            not_found_cnt += 1
    print('{} chars not in char vector'.format(not_found_cnt))
    # for PAD token, assign zero vector
    char_matrix[0] = np.full(shape=emb_dim, fill_value=0.0, dtype=np.float32)
    print('char_matrix.shape:', char_matrix.shape)
    return char_matrix


if __name__ == '__main__':
    with open('necessary.pkl', 'rb') as f:
        data = pickle.load(f)

    load_word_matrix('pretrained_dim_300.pkl', data['word'], 300, False)
    load_char_matrix('pretrained_dim_300.pkl', data['character'], 300, False)
