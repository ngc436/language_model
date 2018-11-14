# TODO: perform optimization with hyperopt with text accuracy maximization

# /tmp/model.h5,
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "1"
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session

#
set_session(session)

# config = tf.ConfigProto(
#     device_count={'GPU': 0}
# )
# session = tf.Session(config=config)
# set_session(session)

import time
import numpy as np
import logging

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
from keras import optimizers

import pandas as pd
from data import prepare_input

# data = pd.read_csv('/mnt/shdstorage/tmp/validation.csv')
# print(data)

# 89208 tokens


timing = str(int(time.time()))

batch_size = 256
# TODO: control the dictionary length
max_features = 228654  # 172567 in the 3rd version
max_len = 100  # reduce
emb_dim = 300

x_train_set_name = '/mnt/shdstorage/for_classification/X_train_5.csv'
x_train_name = x_train_set_name.split('/')[-1].split('.')[0]
x_test_set_name = '/mnt/shdstorage/for_classification/X_test_5.csv'
y_train_labels = '/mnt/shdstorage/for_classification/y_train_5.csv'
y_test_labels = '/mnt/shdstorage/for_classification/y_test_5.csv'

# x_train_set_name = '/mnt/shdstorage/tmp/classif_tmp/X_train_3.csv'
# x_train_name = x_train_set_name.split('/')[-1].split('.')[0]
# x_test_set_name = '/mnt/shdstorage/tmp/classif_tmp/X_test_3.csv'
# y_train_labels = '/mnt/shdstorage/tmp/classif_tmp/y_train_3.csv'
# y_test_labels = '/mnt/shdstorage/tmp/classif_tmp/y_test_3.csv'

# set new verification
# verification_name = '/mnt/shdstorage/tmp/classif_tmp/comments_big.csv'

verification_name = '/mnt/shdstorage/tmp/verification_big.csv'
path_to_goal_sample = '/mnt/shdstorage/tmp/classif_tmp/comments_big.csv'

emb_type = 'fasttext_2'

X_train = pd.read_csv(x_train_set_name, header=None).values.tolist()
X_test = pd.read_csv(x_test_set_name, header=None).values.tolist()
y_train = pd.read_csv(y_train_labels, header=None).values.tolist()
y_train = [y[0] for y in y_train]
y_test = pd.read_csv(y_test_labels, header=None).values.tolist()
y_test = [y[0] for y in y_test]

if path_to_goal_sample:
    X_train, y_train, X_test, y_test, embedding_matrix, verification, goal = prepare_input(X_train, y_train, X_test,
                                                                                           y_test,
                                                                                           max_features=max_features,
                                                                                           verification_name=verification_name,
                                                                                           emb_type=emb_type,
                                                                                           max_len=max_len,
                                                                                           x_train_name=x_train_name,
                                                                                           path_to_goal_sample=path_to_goal_sample)
else:
    X_train, y_train, X_test, y_test, embedding_matrix, verification = prepare_input(X_train, y_train, X_test, y_test,
                                                                                     max_features=max_features,
                                                                                     verification_name=verification_name,
                                                                                     emb_type=emb_type,
                                                                                     max_len=max_len,
                                                                                     x_train_name=x_train_name)

# ======= PARAMS =======
spatial_dropout = 0.2
window_size = 3
dropout = 0.2
kernel_regularizer = 1e-6
bias_regularizer = 1e-6
kernel_constraint = maxnorm(6)
bias_constraint = maxnorm(6)
loss = 'binary_crossentropy'
optimizer = 'adam'
model_type = 'Bidirectional'
lr = 0.00002 # changed from 0.00001
clipnorm = None
epochs = 15 # 20
weights = True
trainable = True
previous_weights = None
# ======= =======

print('Build model...')
model = Sequential()

if weights:
    model.add(Embedding(max_features, emb_dim, weights=[embedding_matrix], trainable=trainable))
else:
    model.add(Embedding(max_features, emb_dim))

model.add(SpatialDropout1D(spatial_dropout))
model.add(Bidirectional(QRNN(emb_dim, window_size=window_size, dropout=dropout,
                             kernel_regularizer=l2(kernel_regularizer), bias_regularizer=l2(bias_regularizer),
                             kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

plot_losses = PlotLosses()
callbacks_list = [plot_losses]

if clipnorm:
    optimizer = optimizers.Adam(lr=lr, clipnorm=clipnorm)
else:
    optimizer = optimizers.Adam(lr=lr)
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

# norm = math.sqrt(sum(numpy.sum(K.get_value(w)) for w in model.optimizer.weights))

print('Loading data...')

print('Train...')
# previous_weights = "models_dir/model_1542197516.h5"
# model.load_weights(previous_weights)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
          callbacks=callbacks_list)

# ================================= MODEL SAVE =========================================

path_to_weights = "models_dir/model_%s.h5" % (timing)
model.save_weights(path_to_weights)
print('Model is saved %s' % path_to_weights)

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

# serialize weights to HDF5

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
train_1, train_0 = calculate_all_metrics(y_train, train_res, 'TRAIN')

test_res = model.predict_classes(X_test)
test_res = [i[0] for i in test_res]
test_1, test_0 = calculate_all_metrics(y_test, test_res, 'TEST')

ver_res = model.predict_classes(verification)
path_to_verification = verification_name
data = pd.read_csv(path_to_verification)
label = data['label'].tolist()
ver_res = [i[0] for i in ver_res]
verif_1, verif_0 = calculate_all_metrics(label, ver_res, 'VERIFICATION >= 10')
true_positive = []
true_negative = []
for i in range(len(ver_res)):
    if ver_res[i] == 0 and label[i] == 0:
        true_negative.append(i)
    if ver_res[i] == 1 and label[i] == 1:
        true_positive.append(i)

# text = data['text'].tolist()
# raw_text = data['raw_text'].tolist()
# pd.set_option('display.max_colwidth', -1)

if path_to_goal_sample:
    goal_set_name = path_to_goal_sample.split('/')[-1].split('.')[0]
    goal_res = model.predict_classes(goal)
    data = pd.read_csv(path_to_goal_sample)['text'].sample(frac=1).tolist()
    ver_res = [i[0] for i in goal_res]
    positive = []
    negative = []
    pos = open('results/positive_%s_%s.txt' % (goal_set_name, timing), 'w')
    neg = open('results/negative_%s_%s.txt' % (goal_set_name, timing), 'w')
    for i in range(len(ver_res)):
        if goal_res[i] == 0:
            neg.write('\n==============================================================\n')
            neg.write('\n')
            neg.write(data[i])
            neg.write('\n')
        else:
            pos.write('\n==============================================================\n')
            pos.write('\n')
            pos.write(data[i])
            pos.write('\n')
    pos.close()
    neg.close()

# ver_res = model.predict_classes(verification)
# path_to_verification = verification_name
# data = pd.read_csv(path_to_verification)
# label = data['label'].tolist()
# ver_res = [i[0] for i in ver_res]

# print('=========================  TRUE POSITIVE  ============================ ')
# print()
# for i in true_positive:
#     print(text[i])
#     print(raw_text[i])
#     print()
#
# print()
# print()
# print('=========================  TRUE NEGATIVE  ============================ ')
# print()
# for i in true_negative:
#     print(text[i])
#     print(raw_text[i])
#     print()

# ======================= LOGS =======================

logs_name = 'logs/qrnn/%s.txt' % timing

with open(logs_name, 'w') as f:
    f.write('======= DATASETS =======\n')
    f.write('Train set data: %s\n' % x_train_set_name)
    f.write('Test set data: %s\n' % x_test_set_name)
    f.write('Train labels: %s\n' % y_train_labels)
    f.write('Test labels: %s\n' % y_test_labels)
    f.write('Verification data: %s\n' % verification_name)
    f.write('======= MODEL PARAMS =======\n')
    f.write('model type %s, previous weights: %s \n' % (model_type, previous_weights))
    if weights:
        f.write('emb_type: %s, emb dim: %s, trainable: %s\n' % (emb_type, emb_dim, trainable))
    f.write('batch size: %s, max features: %s, max len: %s\n' % (
        batch_size, max_features, max_len))
    f.write('spatial dropout: %s, window size: %s, dropout: %s\n' % (spatial_dropout, window_size, dropout))
    f.write('kernel regularizer: %s, bias regularizer: %s, kernel constraint: %s, bias constraint: %s\n' % (
        kernel_regularizer, bias_regularizer, 'maxnorm(10)', 'maxnorm(10)'))
    f.write('loss: %s, optimizer: %s, learning rate: %s, clipnorm: %s, epochs: %s\n' % (
        loss, optimizer, lr, clipnorm, epochs))
    f.write('======= RESULTS =======\n')
    f.write(train_1 + '\n')
    f.write(train_0 + '\n')
    f.write(test_1 + '\n')
    f.write(test_0 + '\n')
    f.write(verif_1 + '\n')
    f.write(verif_0 + '\n')
    f.write('model weights: %s' % path_to_weights + '\n')

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

# 56320/147454 [==========>...................] - ETA: 1:07 - loss: 0.5321 - acc: 0.7312
