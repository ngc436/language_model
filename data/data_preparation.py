import os.path
import sys
import keras

# TODO: initialize unknown words with average from all embeddings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# TODO: remove wrong tokenizers (old version)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import FastText
import pickle

import requests
import re

import pymorphy2
import pymystem3

morph = pymorphy2.MorphAnalyzer()

# path_to_model = '/tmp/web_0_300_20.bin'
# TODO: find out what's wrong with tayga corpora
path_to_w2v = '/home/gmaster/projects/negRevClassif/data/embeddings/ruwikiruscorpora_upos_skipgram_300_2_2018.vec'
path_to_fasttext_emb = '/tmp/wiki.ru.bin'
path_to_fasttext_emb_2 = '/home/gmaster/projects/negRevClassif/data/embeddings/ft_native_300_ru_wiki_lenta_lemmatize.bin'
path_to_fasttext_unlem = '/tmp/ft_native_300_ru_wiki_lenta_lower_case.bin'
# add w2v embeddings
upt_url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'
m = pymystem3.mystem.Mystem(mystem_bin='/home/gmaster/mystem')

# ==================== ADD UPT MAPPING ========================

mapping = {}
r = requests.get(upt_url, stream=True)
for pair in r.text.split('\n'):
    pair = re.sub('\s+', ' ', pair, flags=re.U).split(' ')
    if len(pair) > 1:
        mapping[pair[0]] = pair[1]


# udpipe_model_url = 'http://rusvectores.org/static/models/udpipe_syntagrus.model'
# udpipe_filename = udpipe_model_url.split('/')[-1]
# if not os.path.isfile(udpipe_filename):
#     print('UDPipe model not found. Downloading...', file=sys.stderr)
# wget.download(udpipe_model_url)
#
# print('Loading the model...', file=sys.stderr)
# model = Model.load(udpipe_filename)

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


def add_universal_tag(word):
    processed = m.analyze(word)
    tagged = []
    for w in processed:
        try:
            lemma = w["analysis"][0]["lex"].lower().strip()
            pos = w["analysis"][0]["gr"].split(',')[0]
            pos = pos.split('=')[0].strip()
            if pos in mapping:
                tagged.append(lemma + '_' + mapping[pos])  # здесь мы конвертируем тэги
            else:
                tagged.append(lemma + '_X')
        except KeyError:
            continue
    return tagged


def embedding(emb_type):
    if emb_type == 'w2v':
        model = KeyedVectors.load_word2vec_format(path_to_w2v)
    if emb_type == 'fasttext':
        model = FastText.load_fasttext_format(path_to_fasttext_emb)
    if emb_type == 'fasttext_2':
        print('loading fasttext embedding...')
        model = FastText.load_fasttext_format(path_to_fasttext_emb_2)
        print('Done!')
    if emb_type == 'fasttext_unlem':
        model = FastText.load_fasttext_format(path_to_fasttext_unlem)
    return model


class Processor:

    def __init__(self, max_features, emb_type, max_len, emb_dim=300):
        self.tokenizer = None
        self.max_features = max_features
        self.emb_type = emb_type
        self.model = None
        self.emb_dim = emb_dim
        self.embedding_matrix = None
        self.x_train_name = None
        self.max_len = max_len

    def prepare_embedding_matrix(self, word_index, x_train_name):
        print('Starting embedding matrix preparation...')
        embedding_matrix = np.zeros((self.max_features, self.emb_dim))
        if self.emb_type == 'w2v':
            for word, i in word_index.items():
                try:
                    emb_vect = self.model.wv[add_universal_tag(word)].astype(np.float32)
                    embedding_matrix[i] = emb_vect
                # out of vocabulary exception
                except:
                    print(word)
        else:
            for word, i in word_index.items():
                try:
                    emb_vect = self.model.wv[word]
                    embedding_matrix[i] = emb_vect.astype(np.float32)
                # out of vocabulary exception
                except:
                    print(word)
        np.save('/home/gmaster/projects/negRevClassif/data/embeddings/%s_%s_%s.npy' % (
            self.emb_type, x_train_name, self.max_features), embedding_matrix)

        return embedding_matrix

    def fit_processor(self, x_train, x_test, x_train_name, other=None):
        self.x_train_name = x_train_name
        try:
            self.embedding_matrix = np.load(
                '/home/gmaster/projects/negRevClassif/data/embeddings/%s_%s_%s.npy' % (
                    self.emb_type, x_train_name, self.max_features))
            with open('/home/gmaster/projects/negRevClassif/data/embeddings/tokenizer_%s_%s_%s.pickle' % (
                    self.emb_type, x_train_name, self.max_features), 'rb') as handle:
                self.tokenizer = pickle.load(handle)

        # not found exception
        except:  # to check
            print('No model found...initialization...')
            x_train = [sent[0] for sent in x_train]
            x_test = [sent[0] for sent in x_test]
            self.tokenizer = Tokenizer(num_words=self.max_features + 1, oov_token='oov')
            if not other:
                self.tokenizer.fit_on_texts(x_train + x_test)
            else:
                if isinstance(other[0], list):
                    other = [sent[0] for sent in other]
                self.tokenizer.fit_on_texts(x_train + x_test + other)
            # hopefully this staff helps to avoid issues with oov (NOT SURE needs to be checked)
            self.tokenizer.word_index = {e: i for e, i in self.tokenizer.word_index.items() if i <= self.max_features}
            self.tokenizer.word_index[self.tokenizer.oov_token] = self.max_features + 1
            word_index = self.tokenizer.word_index
            with open('/home/gmaster/projects/negRevClassif/data/embeddings/tokenizer_%s_%s_%s.pickle' % (
                    self.emb_type, x_train_name, self.max_features), 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # ======================== write tokenizer to file ===================================

            print('Amount of unique tokens %s' % len(word_index))

            self.model = embedding(self.emb_type)
            self.embedding_matrix = self.prepare_embedding_matrix(word_index, x_train_name)

    def prepare_input(self, x, y=None):
        # prepare x data
        if isinstance(x[0], list):
            x = [sent[0] for sent in x]
        sequences_x = self.tokenizer.texts_to_sequences(x)
        x = pad_sequences(sequences_x, maxlen=self.max_len)
        # prepare labels
        if y:
            if isinstance(y[0], list):
                y = [y[0] for y in y]
            y = np.asarray(y)
            return x, y
        return x

    def prepare_sequence(self, text):
        text = [text]
        sequences = self.tokenizer.texts_to_sequences(text)
        x = pad_sequences(sequences, maxlen=self.max_len)
        return x

    def prepare_custom_embedding(self,  vocabulary, x_train_name='custom'):
        self.max_features = len(vocabulary)
        try:
            self.embedding_matrix = np.load('/home/gmaster/projects/negRevClassif/data/embeddings/%s_%s_%s.npy' % (
                    self.emb_type, x_train_name, self.max_features))
        except:
            print('Starting embedding matrix preparation...')
            self.model = embedding(self.emb_type)
            embedding_matrix = np.zeros((len(vocabulary), self.emb_dim))
            if self.emb_type == 'w2v':
                for i, word in enumerate(vocabulary):
                    try:
                        emb_vect = self.model.wv[add_universal_tag(word)].astype(np.float32)
                        embedding_matrix[i] = emb_vect
                    # out of vocabulary exception
                    except:
                        print(word)
            else:
                for i, word in enumerate(vocabulary):
                    try:
                        emb_vect = self.model.wv[word]
                        embedding_matrix[i] = emb_vect.astype(np.float32)
                    # out of vocabulary exception
                    except:
                        print(word)

            self.embedding_matrix = embedding_matrix
            np.save('/home/gmaster/projects/negRevClassif/data/embeddings/%s_%s_%s.npy' % (
                self.emb_type, x_train_name, self.max_features), embedding_matrix)


class DataGenerator(keras.utils.Sequence):

    def __init__(self, ids_list, dim=(32, 32, 32), batch_size=32):
        self.dim = dim
        self.ids_list = ids_list
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.ids_list) / self.batch_size))

    def __data_generation(self, index):
        max_iter = min(self.batch_size, len(self.x) - self.batch_size * index)
