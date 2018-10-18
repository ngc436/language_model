from language_model import QRNN

import numpy as np

np.random.seed(3255)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, SpatialDropout1D
from keras.regularizers import l2
from keras.constraints import maxnorm

def

class QRNN_model:

    def __init__(self, max_features=100000, max_len=90,embedding_dim=300, batch_size=64,
                 emb_matrix=None):
        self.max_features = max_features
        self.max_len = max_len
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.emb_matrix = emb_matrix
        self.model = self.init_model()

    def init_model(self):
        print('Building model...')
        model = Sequential()
        # len(word_index) + 1
        if self.emb_matrix:
            model.add(Embedding(self.max_features, self.embedding_dim, weights=[self.emb_matrix]))
        model.add(SpatialDropout1D(0.2))
        model.add(QRNN(128, window_size=3, dropout=0.2,
                       kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
                       kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def fit(self):


