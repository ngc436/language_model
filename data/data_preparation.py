from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from gensim.models import  KeyedVectors
import pandas as pd
from keras.preprocessing import sequence

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

path_to_model = '/tmp/web_0_300_20.bin'

def tag_mappings(tag):
    if tag is None:
        print('VIUVIU')
        return 'X'
    if tag in ['NOUN']:
        return 'NOUN'
    if tag in ['ADJF','ADJS']:
        return 'ADJ'
    # errors with intj
    if tag in ['VERB','INFN','PRTF','PRTS','GRND','INTJ']:
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


def prepare_input(x_data, y_data, max_features=100000, max_len=100, emb_dim=300, fname='valid_emb'):
    # x_data = add_pos_tags(x_data)
    tokenizer = Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(x_data)
    sequences = tokenizer.texts_to_sequences(x_data)
    print(sequences)
    word_index = tokenizer.word_index
    print('Amount of unique torens %s' % len(word_index))
    data = pad_sequences(sequences, maxlen=max_len)
    labels = np.asarray(y_data) # check
    # computing the index mapping
    emb_ind = {}
    model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    for word, i in word_index.items():
        try:
            emb_vect = model.wv[add_pos_tag(word)]
            embedding_matrix[i] = emb_vect
        except:
            continue
    np.save(embedding_matrix,'/tmp/%s'%fname)
    np.load('/tmp/%s'%fname)



docs = ['лиса идти лес', 'есть пить чай']
print(add_pos_tags(docs))

path_to_validation = '/mnt/shdstorage/tmp/validation.csv'
data = pd.read_csv(path_to_validation)
x = data['text'].tolist()
print(x)
y = data['label'].tolist()
prepare_input(x,y)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
