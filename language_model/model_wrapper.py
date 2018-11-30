# TODO: finish model wrapper

from language_model import QRNN
import language_model
from data import Processor

import numpy as np
import pandas as pd

np.random.seed(3255)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, SpatialDropout1D
from keras.regularizers import l2
from keras.constraints import maxnorm

x_train_set_name = '/mnt/shdstorage/for_classification/X_train_4.csv'
x_train_name = x_train_set_name.split('/')[-1].split('.')[0]
x_test_set_name = '/mnt/shdstorage/for_classification/X_test_4.csv'
y_train_labels = '/mnt/shdstorage/for_classification/y_train_4.csv'
y_test_labels = '/mnt/shdstorage/for_classification/y_test_4.csv'

class BaseModel:

    def __init__(self, max_features=100000, max_len=90, embedding_dim=300, batch_size=64,
                 emb_matrix=None):

        # common for all models
        self.batch_size = batch_size
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.emb_matrix = emb_matrix
        self.model_type = None
        self.X_train = None
        self.y_train = None

    def init_model(self):
        raise NotImplementedError

    def prepare_data(self):

        X_train = pd.read_csv(x_train_set_name, header=None).values.tolist()
        X_test = pd.read_csv(x_test_set_name, header=None).values.tolist()
        y_train = pd.read_csv(y_train_labels, header=None).values.tolist()
        y_train = [y[0] for y in y_train]
        y_test = pd.read_csv(y_test_labels, header=None).values.tolist()
        y_test = [y[0] for y in y_test]



    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def run_pipeline(self, model_type):
        # data preparation

        self.model_type = model_type




class BiQrnnModel(BaseModel):

    def __init__(self, max_features=100000, max_len=90, embedding_dim=300, batch_size=64,
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
        print()



class BiLstmModel(BaseModel):

    def __init__(self):
        super().__init__()
