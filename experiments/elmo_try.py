# https://tfhub.dev/google/elmo/2
# https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440

# https://github.com/HIT-SCIR/ELMoForManyLangs

# The embeddings are obtained with pretrained model from ELMoForManyLangs

import tensorflow as tf
import tensorflow_hub as hub

# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# embeddings = elmo(
# ["первое предложение", "второе предложение"],
# signature="default",
# as_dict=True)["elmo"]
# print(embeddings)

# elmo try with bilstm

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
from keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Flatten
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers

import pandas as pd
from data import Processor

from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder

fname_x_batches = '/home/gmaster/projects/other/elmo/embeddings/X_train/'
fname_y = '/mnt/shdstorage/for_classification/y_train_6_no_ent.csv'


# TODO: change to simple batch yield function
class DataGenerator:

    def __init__(self, fname_x_batches, fname_y, batch_size, num_of_batches):
        self.fname_x_batches = fname_x_batches
        self.y = self.prepare_y(fname_y)
        self.num_of_batches = num_of_batches
        self.epoch_batch_count = 0
        self.batch_size = batch_size
        self.on_epoch_end()

    def prepare_y(self, fname_y):
        y = pd.read_csv(fname_y, header=None).values.tolist()
        if isinstance(y[0], list):
            y = [y[0] for y in y]
        y = np.asarray(y)
        return y

    def __len__(self):
        return int(np.floor(len(self.y) / self.batch_size))

    def __next__(self):
        x_batch = np.load(self.fname_x_batches + 'X_train_' + str(self.epoch_batch_count) + '.npy')
        y_batch = self.y[(self.batch_size * self.epoch_batch_count):(
                self.batch_size * self.epoch_batch_count + self.batch_size)]
        self.epoch_batch_count += 1
        print(y_batch.shape)
        return x_batch, y_batch

    def on_epoch_end(self):
        print("I'm here!")
        self.epoch_batch_count = 0


# try simple function to generate batches
# 

timing = str(int(time.time()))

batch_size = 32
# cut words with 1, 2 appearances
# 61502 in version with no ent
# 123004 in cut version without entities
max_len = 100
emb_dim = 1024

num_of_batches = 3

# ======= PARAMS =======
spatial_dropout = None
window_size = 3
dropout = 0.5
recurrent_dropout = 0.5
units = 300
kernel_regularizer = 1e-6
bias_regularizer = 1e-6
kernel_constraint = 6
bias_constraint = 6
loss = 'binary_crossentropy'
optimizer = 'adam'
model_type = 'Bidirectional'
lr = 0.001
clipnorm = None
epochs = 50
weights = True
trainable = False
previous_weights = None
activation = 'sigmoid'
time_distributed = False
# ======= =======

print('Build model...')

model = Sequential()
model.add(Bidirectional(
    LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout), input_shape=(max_len, emb_dim)))
model.add(Dropout(dropout))
model.add(Dense(1, activation=activation))
plot_losses = PlotLosses()
reduce_lr = ReduceLROnPlateau(monitor='val_loss')
callbacks_list = [plot_losses, reduce_lr]

model.compile(loss=loss,
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])
print(model.summary())

print('Loading data...')
training_generator = DataGenerator(fname_x_batches, fname_y, batch_size, num_of_batches)
model.fit_generator(training_generator, steps_per_epoch=2, epochs=3, callbacks=callbacks_list)
