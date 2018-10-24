# TODO:

from __future__ import print_function
import numpy as np

from vis_tools import *

np.random.seed(1337)  # for reproducibility

from language_model import QRNN
from metrics import calculate_all_metrics

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Embedding, SpatialDropout1D, Bidirectional
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.datasets import imdb

import pandas as pd
from data import prepare_input

# data = pd.read_csv('/mnt/shdstorage/tmp/validation.csv')
# print(data)

# 89208 tokens

batch_size = 320
max_features = 273046  # 172567
max_len = 100
emb_dim = 300

X_train = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/X_train_2.csv', header=None).values.tolist()
X_test = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/X_test_2.csv', header=None).values.tolist()
y_train = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_train_2.csv', header=None).values.tolist()
y_train = [y[0] for y in y_train]
y_test = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_test_2.csv', header=None).values.tolist()
y_test = [y[0] for y in y_test]

X_train, y_train, X_test, y_test, embedding_matrix, verification, validation_y = prepare_input(X_train, y_train, X_test,
                                                                                               y_test)

# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
# print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
print('y_test:', y_test[:100])

print(embedding_matrix[:100])

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, emb_dim, weights=[embedding_matrix]))
model.add(SpatialDropout1D(0.2))
model.add(QRNN(emb_dim, window_size=3, dropout=0.4,
               kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
               kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

plot_losses = PlotLosses()
# plot_accuracy = PlotAccuracy()
callbacks_list = [plot_losses]  # plot_accuracy]

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Loading data...')

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_test, y_test), callbacks=callbacks_list)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

model_json = model.to_json()
with open("/tmp/model_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/tmp/model_1.h5")
print("Model is saved to disk")

# json_file = open('/tmp/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("/tmp/model.h5")
# loaded_model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

train_res = model.predict_classes(X_train)
train_res = [i[0] for i in train_res]
calculate_all_metrics(y_train, train_res, 'TRAIN')

test_res = model.predict_classes(X_test)
test_res = [i[0] for i in test_res]
calculate_all_metrics(y_test, test_res, 'TEST')

ver_res = model.predict_classes(verification)
path_to_verification = '/mnt/shdstorage/tmp/verification_2.csv'
data = pd.read_csv(path_to_verification)
label = data['label'].tolist()
ver_res = [i[0] for i in ver_res]
calculate_all_metrics(label, ver_res, 'VERIFICATION')

# strike = 0
# positive_negative = 0
# positive_positive = 0
# negative_positive = 0
# negative_negative = 0
# for i, result in enumerate(val_res):
#     print(i, result[0], label[i])
#     if result[0] == 1 and label[i] == 0:
#         positive_negative += 1
#     if result[0] == 1 and label[i] == 1:
#         positive_positive += 1
#         strike += 1
#     if result[0] == 0 and label[i] == 1:
#         negative_positive += 1
#     if result[0] == 0 and label[i] == 0:
#         negative_negative += 1
#         strike += 1
# print()
# print(positive_negative, positive_positive, negative_positive, negative_negative)
#
# print('Accuracy:', strike/len(label))
# print(data)
