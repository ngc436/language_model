from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from language_model import QRNN

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, SpatialDropout1D
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.datasets import imdb

import pandas as pd
from data import prepare_input

# data = pd.read_csv('/mnt/shdstorage/tmp/validation.csv')
# print(data)

# 89208 tokens

batch_size = 64
max_features = 89209
max_len = 100
emb_dim = 300

X_train = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/X_train.csv', header=None).values.tolist()
X_test = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/X_test.csv', header=None).values.tolist()
y_train = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_train.csv', header=None).values.tolist()
y_train = [y[0] for y in y_train]
y_test = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_test.csv', header=None).values.tolist()
y_test = [y[0] for y in y_test]

X_train, y_train, X_test, y_test, embedding_matrix, validation, validation_y = prepare_input(X_train, y_train, X_test,
                                                                                             y_test)

# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
# print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
print('y_test:', y_test[:100])

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, emb_dim, weights=[embedding_matrix]))
model.add(SpatialDropout1D(0.2))
model.add(QRNN(emb_dim, window_size=3, dropout=0.2,
               kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
               kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Loading data...')

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print(model.predict_classes(validation))
path_to_validation = '/mnt/shdstorage/tmp/validation.csv'
data = pd.read_csv(path_to_validation)
print(data)
