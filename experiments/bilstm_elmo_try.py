# TODO: add verification results printing

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
from collections import OrderedDict

set_session(session)

import time
import numpy as np
import logging

from language_model import TimestepDropout

from vis_tools import *

np.random.seed(1337)

from language_model import QRNN
from metrics import calculate_all_metrics

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, SpatialDropout1D, Dropout, Bidirectional, LSTM, GlobalMaxPool1D, \
    BatchNormalization, Input
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers

import pandas as pd
from data import Processor
import pickle

timing = str(int(time.time()))

batch_size = 128

y_train = pd.read_csv('/mnt/shdstorage/for_classification/train_v7.csv')['label'].tolist()
y_test = pd.read_csv('/mnt/shdstorage/for_classification/test_v7.csv')['label'].tolist()

X_train = pickle.load(open('/mnt/shdstorage/for_classification/elmo/train_v7.pkl', 'rb'))
X_test = pickle.load(open('/mnt/shdstorage/for_classification/elmo/test_v7.pkl', 'rb'))
X_ver = pickle.load(open('/mnt/shdstorage/for_classification/elmo/ver_v7.pkl', 'rb'))

ff = np.zeros((len(X_train), 1024))
for i in range(len(X_train)):
    ff[i] = X_train[i][0]
X_train = ff
X_train = X_train[:(len(X_train)//batch_size)*batch_size, :]

ff = np.zeros((len(X_test), 1024))
for i in range(len(X_test)):
    ff[i] = X_test[i][0]
X_test = ff
X_test = X_test[:(len(X_test)//batch_size)*batch_size, :]

ff = np.zeros((len(X_ver), 1024))
for i in range(len(X_ver)):
    ff[i] = X_ver[i][0]
X_ver = ff

X_train = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_sent_train_v7.pkl', 'rb'))
X_train = X_train[:(len(X_train)//batch_size)*batch_size]
X_test = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_sent_test_v7.pkl', 'rb'))
X_test = X_test[:(len(X_test)//batch_size)*batch_size]
X_ver = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_sent_ver.pkl', 'rb'))

if isinstance(y_train[0], list):
    y_train = [y[0] for y in y_train]
y_train = np.asarray(y_train)
y_train =y_train[:(len(y_train)//batch_size)*batch_size]

if isinstance(y_test[0], list):
    y_test = [y[0] for y in y_test]
y_test = np.asarray(y_test)
y_test = y_test[:(len(y_test)//batch_size)*batch_size]

# ============= PARAMS ===============
window_size = 3
dropout = 0.1
recurrent_dropout = 0.1
word_dropout = None
units = 100  #
kernel_regularizer = 1e-6
bias_regularizer = 1e-6
kernel_constraint = 6
bias_constraint = 6
loss = 'binary_crossentropy' #
optimizer = 'adam'  # changed from adam
model_type = 'Bidirectional'
lr = 0.001
clipnorm = None
epochs = 5
weights = True
trainable = True
previous_weights = None
activation = 'sigmoid'
time_distributed = False

# ======================================

# ================================ MODEL ==========================================

print('Build model...')
model = Sequential()

# model.add(SpatialDropout1D(spatial_dropout))

# model.add(
#     Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout), batch_size=batch_size,
#                   input_shape=(1024, 1)))
model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, batch_size=batch_size,
                  input_shape=(1024, 1)))
# model.add(GlobalMaxPool1D())
# model.add(Dense(1, activation=activation))
model.add(Dropout(dropout))
model.add(Dense(1, activation=activation))

plot_losses = PlotLosses()
plot_accuracy = PlotAccuracy()
#
early_stopping = EarlyStopping(monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss')
callbacks_list = [plot_losses, reduce_lr, plot_accuracy]

if clipnorm:
    optimizer = optimizers.Adam(lr=lr, clipnorm=clipnorm)
else:
    optimizer = optimizers.Adam(lr=lr)
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

# ================================== END ============================================

# ================================== RESHAPING ======================================


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_ver = np.reshape(X_ver, (X_ver.shape[0], X_ver.shape[1], 1))

# making timesteps to be one
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# X_ver = np.reshape(X_ver, (X_ver.shape[0], 1, X_ver.shape[1]))


# ===================================================================================

# print('Loading weights...')
# previous_weights = "/mnt/shdstorage/for_classification/models_dir/model_1544540767.h5"
# model.load_weights(previous_weights)

fit = True
if fit:
    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
              callbacks=callbacks_list)

    # ================================= MODEL SAVE =========================================

    # path_to_weights = "models_dir/model_%s.h5" % (timing)
    path_to_weights = '/mnt/shdstorage/for_classification/models_dir/model_%s.h5' % timing
    path_to_architecture = "/mnt/shdstorage/for_classification/models_dir/architecture/model_%s.h5"
    model.save_weights(path_to_weights)
    model.save(path_to_architecture)
    print('Model is saved %s' % path_to_weights)

else:
    timing = previous_weights.split('/')[-1].split('_')[-1].split('.')[0]

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

train_res = model.predict_classes(X_train)
train_res = [i[0] for i in train_res]
train_1, train_0 = calculate_all_metrics(y_train, train_res, 'TRAIN')

test_res = model.predict_classes(X_test)
test_res = [i[0] for i in test_res]
test_1, test_0 = calculate_all_metrics(y_test, test_res, 'TEST')

ver_res = model.predict_classes(X_ver)
data = pd.read_csv('/mnt/shdstorage/for_classification/new_test.csv')
label = data['label'].tolist()
ver_res = [i[0] for i in ver_res]
verif_1, verif_0 = calculate_all_metrics(label, ver_res, 'VERIFICATION')
