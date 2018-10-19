from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
from keras.preprocessing import sequence

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

path_to_model = '/tmp/web_0_300_20.bin'


# TODO: check mappings more carefully
def tag_mappings(tag):
    if tag is None:
        return 'X'
    if tag in ['NOUN']:
        return 'NOUN'
    if tag in ['ADJF', 'ADJS']:
        return 'ADJ'
    # errors with intj
    if tag in ['VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'INTJ']:
        return 'VERB'
    if tag in ['ADVB']:
        return 'ADV'
    if tag in ['CONJ']:
        return 'CCONJ'
    return 'X'


def add_pos_tags(docs):
    for i in range(len(docs)):
        docs[i] = ' '.join([x + '_' + tag_mappings(morph.parse(x)[0].tag.POS) for x in docs[i].split()])
    return docs


def add_pos_tag(word):
    return word + '_' + tag_mappings(morph.parse(word)[0].tag.POS)


# class Input

# def prepare_validation(validation, max_features=100000, max_len=100, emb_dim=300):

# TODO: convert to class
def prepare_input(x_train, y_train, x_test, y_test, max_features=100000, max_len=100, emb_dim=300):
    # x_data = add_pos_tags(x_data)
    tokenizer = Tokenizer(num_words=max_features)
    x_train = [sent[0] for sent in x_train]
    x_test = [sent[0] for sent in x_test]

    tokenizer.fit_on_texts(x_train + x_test)
    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_test = tokenizer.texts_to_sequences(x_test)
    word_index = tokenizer.word_index
    print('Amount of unique tokens %s' % len(word_index))
    x_train = pad_sequences(sequences_train, maxlen=max_len)
    x_test = pad_sequences(sequences_test, maxlen=max_len)
    y_train = np.asarray(y_train)  # check
    y_test = np.asarray(y_test)
    # computing the index mapping
    emb_ind = {}
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    print('Starting embedding matrix preparation...')
    for word, i in word_index.items():
        try:
            emb_vect = model.wv[add_pos_tag(word)]
            embedding_matrix[i] = emb_vect
        except:
            continue

    # TODO: change this
    path_to_validation = '/mnt/shdstorage/tmp/validation.csv'
    data = pd.read_csv(path_to_validation)
    texts = data.text.tolist()
    validation = tokenizer.texts_to_sequences(texts)
    validation = pad_sequences(validation, maxlen=max_len)

    return x_train, y_train, x_test, y_test, embedding_matrix, validation, data['label']


# X_train = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/X_train.csv', header=None).values.tolist()
# X_test = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/X_test.csv', header=None).values.tolist()
# y_train = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_train.csv', header=None).values.tolist()
# y_train = [y[0] for y in y_train]
# y_test = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_test.csv', header=None).values.tolist()
# y_test = [y[0] for y in y_test]
#
# X_train, y_train, X_test, y_test, embedding_matrix, validation, validation_y = prepare_input(X_train, y_train, X_test,
#                                                                                              y_test)


# docs = ['лиса идти лес', 'есть пить чай']
# print(add_pos_tags(docs))
#
# path_to_validation = '/mnt/shdstorage/tmp/validation.csv'
# data = pd.read_csv(path_to_validation)
# x = data['text'].tolist()
# print(x)
# y = data['label'].tolist()
# prepare_input(x, y)
#
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
