# TODO: save matrix of latent representation

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# config = tf.ConfigProto(
#     device_count={'GPU': 0}
# )
# session = tf.Session(config=config)
# set_session(session)
#
from keras.backend.tensorflow_backend import set_session

#
set_session(session)

import numpy as np
from vis_tools import *

from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Activation, Embedding, SpatialDropout1D, Bidirectional, \
    Input, LSTM, Dropout, Lambda, RepeatVector, TimeDistributed, Layer
from keras.constraints import maxnorm
from keras.layers.advanced_activations import ELU
import keras.backend as K
from keras.backend import clear_session
from keras.models import Model
from keras.optimizers import Adam

import pandas as pd
from data import prepare_input

print('Build model...')

# ======= PARAMS =======
batch_size = 32
max_features = 273046  # 172567
latent_dim = 200
max_len = 80  # reduce
emb_dim = 300
int_dim = 96
recurrent_dropout = 0.2
spatial_dropout = 0.1
window_size = 3
dropout_1 = 0.2
dropout_2 = 0.2
kernel_regularizer = 1e-6
bias_regularizer = 1e-6
kernel_constraint = maxnorm(10)
bias_constraint = maxnorm(10)
loss = 'binary_crossentropy'
optimizer = 'adam'
epochs = 10
weights = True
trainable = False

epsilon_std = 1
act = ELU()

# ======= DATASETS =======

x_train_set_name = '/mnt/shdstorage/for_classification/X_train_5.csv'
x_train_name = x_train_set_name.split('/')[-1].split('.')[0]
x_test_set_name = '/mnt/shdstorage/for_classification/X_test_5.csv'
y_train_labels = '/mnt/shdstorage/for_classification/y_train_5.csv'
y_test_labels = '/mnt/shdstorage/for_classification/y_test_5.csv'

emb_type = 'fasttext_2'

verification_name = '/mnt/shdstorage/tmp/verification_big.csv'
path_to_goal_sample = '/mnt/shdstorage/tmp/classif_tmp/test.csv'

X_train = pd.read_csv(x_train_set_name, header=None).values.tolist()
x_train_name = x_train_set_name.split('/')[-1].split('.')[0]
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

tmp = int((len(X_train)) / batch_size) * batch_size
X_train = X_train[:tmp]

tmp = int((len(X_test)) / batch_size) * batch_size
X_test = X_test[:tmp]

x = Input(batch_shape=(None, max_len))

if weights:
    embedding = Embedding(max_features, emb_dim, weights=[embedding_matrix], trainable=trainable)(x)
else:
    embedding = Embedding(max_features, emb_dim)(x)

h = Bidirectional(LSTM(int_dim, return_sequences=False, recurrent_dropout=recurrent_dropout), merge_mode='concat')(
    embedding)
h = Dropout(dropout_1)(h)
h = Dense(int_dim, activation='linear')(h)
h = act(h)
h = Dropout(dropout_2)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0, stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


print(z_mean, z_log_var)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
repeated_context = RepeatVector(max_len)
decoder_h = LSTM(int_dim, return_sequences=True, recurrent_dropout=recurrent_dropout)
decoder_mean = TimeDistributed(Dense(max_features, activation='linear'))
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

    def vae_loss(self, x, x_decoded_mean):
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels,
                                                           weights=self.target_weights,
                                                           average_across_timesteps=False,
                                                           average_across_batch=False), axis=-1)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return K.ones_like(x)


loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
opt = Adam(lr=0.01)
vae.compile(optimizer='adam', loss=[zero_loss])
print(vae.summary())

plot_losses = PlotLosses()
callbacks_list = [plot_losses]
vae.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, X_test),
        callbacks=callbacks_list)

encoder = Model(x, z_mean)

decoder_input = Input(shape=(latent_dim))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample

def print_sentence(sentence_vect):
    # TODO: high need in fasttext model
    sentence = ''


def sentence_variation(sentence_1, sentence_2, batch, dim):
    # TODO: prepare sentences
    # sentence_1 = sentence_1...
    # sentence_2 = sentence_2...
    encode_1 = encoder.predict(sentence_1, batch_size=batch)
    encode_2 = encoder.predict(sentence_2, batch_size=batch)
    test_hom =  shortest_homology(encode_1[0], encode_2[0], 5)

    for point in test_hom:
        p = generator.predict(np.array([point]))[0]
        print_sentence(p)



# get latent representation
