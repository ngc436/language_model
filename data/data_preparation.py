import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# TODO: remove wrong tokenizers (old version)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
from gensim.models import FastText
import pickle
import time

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

path_to_model = '/tmp/web_0_300_20.bin'
path_to_fasttext_emb = '/tmp/wiki.ru.bin'
path_to_fasttext_emb_2 = '/tmp/ft_native_300_ru_wiki_lenta_lemmatize.bin'
path_to_fasttext_unlem = '/tmp/ft_native_300_ru_wiki_lenta_lower_case.bin'


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

def _train_model(x_train, x_test, max_features=100000, emb_dim=300,
                 emb_type='fasttext_2', x_train_name=None):
    tokenizer = Tokenizer(num_words=max_features + 1, oov_token='oov')
    tokenizer.fit_on_texts(x_train + x_test)
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= max_features}
    tokenizer.word_index[tokenizer.oov_token] = max_features + 1
    word_index = tokenizer.word_index
    with open('/home/gmaster/projects/negRevClassif/data/embeddings/tokenizer_%s_%s_%s.pickle' % (
    emb_type, x_train_name, max_features), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Amount of unique tokens %s' % len(word_index))
    if emb_type == 'w2v':
        model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    if emb_type == 'fasttext':
        model = FastText.load_fasttext_format(path_to_fasttext_emb)
    if emb_type == 'fasttext_2':
        model = FastText.load_fasttext_format(path_to_fasttext_emb_2)
    if emb_type == 'fasttext_unlem':
        model = FastText.load_fasttext_format(path_to_fasttext_unlem)
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    print(len(word_index))
    print('Starting embedding matrix preparation...')
    set_of_undefined = {}
    if emb_type == 'w2v':
        for word, i in word_index.items():
            try:
                emb_vect = model.wv[add_pos_tag(word)].astype(np.float32)
                embedding_matrix[i] = emb_vect
            # out of vocabulary exception
            except:
                if word not in set_of_undefined:
                    set_of_undefined[word] = np.random.random(emb_dim)
                embedding_matrix[i] = set_of_undefined[word]

    else:
        for word, i in word_index.items():
            try:
                emb_vect = model.wv[word]
                embedding_matrix[i] = emb_vect.astype(np.float32)
            # out of vocabulary exception
            except:
                print(word)
    np.save('/home/gmaster/projects/negRevClassif/data/embeddings/%s_%s_%s.npy' % (emb_type, x_train_name, max_features),
            embedding_matrix)
    return embedding_matrix


# TODO: convert to class
# TODO: save w2v embeddings
def prepare_input(x_train, y_train, x_test, y_test, max_features=100000, max_len=100, emb_dim=300,
                  verification_name=None, emb_type='fasttext_2', x_train_name=None, path_to_goal_sample=None):
    x_train = [sent[0] for sent in x_train]
    x_test = [sent[0] for sent in x_test]

    try:
        embedding_matrix = np.load(
            '/home/gmaster/projects/negRevClassif/data/embeddings/%s_%s_%s.npy' % (emb_type, x_train_name, max_features))
    # not found exception
    except:
        print('Embedding model does not exist. Initialization...')
        embedding_matrix = _train_model(x_train=x_train, x_test=x_test, max_features=max_features, emb_dim=emb_dim,
                                        emb_type=emb_type, x_train_name=x_train_name)

    with open('/home/gmaster/projects/negRevClassif/data/embeddings/tokenizer_%s_%s_%s.pickle' % (
    emb_type, x_train_name, max_features), 'rb') as handle:
        tokenizer = pickle.load(handle)

    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_test = tokenizer.texts_to_sequences(x_test)
    x_train = pad_sequences(sequences_train, maxlen=max_len)
    x_test = pad_sequences(sequences_test, maxlen=max_len)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # oov
    path_to_verification = verification_name
    data = pd.read_csv(path_to_verification)
    texts = data.processed_text.tolist()  # CHANGE HERE
    verification = tokenizer.texts_to_sequences(texts)
    verification = pad_sequences(verification, maxlen=max_len)

    # prepare goal sample (columns: processed, text)
    if path_to_goal_sample:
        goal = pd.read_csv(path_to_goal_sample)
        texts = goal.processed_text_10.tolist()
        sequences_goal = tokenizer.texts_to_sequences(texts)
        x_goal = pad_sequences(sequences_goal, maxlen=max_len)
        return x_train, y_train, x_test, y_test, embedding_matrix, verification, x_goal

    return x_train, y_train, x_test, y_test, embedding_matrix, verification


def oov_processing():
    # if word exists at least in 10% of documents leave it
    # in other case, substitute to oov
    raise NotImplementedError


def prepare_sequence(text):
    # TODO: remove hardcore
    text = [text]
    with open(
            '/home/gmaster/projects/negRevClassif/data/embeddings/tokenizer_%s_%s_%s.pickle' % ('fasttext_2', 'X_train_5', 291837),
            'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(text)
    x = pad_sequences(sequences, maxlen=100)
    # y_test = np.asarray(y_test)
    return x


class Preprocessor():
    pass

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
# # x = data['text'].tolist()
# print(data)
# y = data['label'].tolist()
# prepare_input(x, y)
#
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
