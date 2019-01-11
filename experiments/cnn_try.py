# https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/sentiment_cnn.py

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

set_session(session)

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from data import Processor
from keras.callbacks import ReduceLROnPlateau
from vis_tools import *
from keras.preprocessing.sequence import pad_sequences
import pickle
import time
from metrics import calculate_all_metrics

np.random.seed(0)
timing = str(int(time.time()))

# =========== PARAMS ============
emb_dim = 300
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

trainable = False
emb_type = 'fasttext_2'

batch_size = 64
epochs = 50

max_features = 60002
max_len = 100

# ===============================

# =========== DATA PREP ============

verification_name = '/mnt/shdstorage/for_classification/new_test.csv'
train_fname = '/mnt/shdstorage/for_classification/train_6_edited_text.csv'

p = Processor(max_features=max_features, emb_type=emb_type, max_len=max_len, emb_dim=emb_dim)
y_train = pd.read_csv(train_fname)['label'].tolist()
x_train_name = train_fname.split('/')[-1].split('.')[0]
y_test = pd.read_csv('/mnt/shdstorage/for_classification/test_6_edited_text.csv')['label'].tolist()

if isinstance(y_train[0], list):
    y_train = [y[0] for y in y_train]
y_train = np.asarray(y_train)

if isinstance(y_test[0], list):
    y_test = [y[0] for y in y_test]
y_test = np.asarray(y_test)

X_train = np.load('/mnt/shdstorage/for_classification/trn_ids.npy')
X_test = np.load('/mnt/shdstorage/for_classification/val_ids.npy')

X_train = pad_sequences(X_train, maxlen=max_len)
print('Train params: ', len(X_train), len(y_train))
X_test = pad_sequences(X_test, maxlen=max_len)
print('Test params: ', len(X_test), len(y_test))

vocabulary = pickle.load(open('/mnt/shdstorage/for_classification/itos.pkl', 'rb'))

verification = np.load('/mnt/shdstorage/for_classification/tok_ver_60.npy')
verification = pad_sequences(verification, maxlen=max_len)
p.prepare_custom_embedding(vocabulary, x_train_name=x_train_name)

# ===============================

# =========== MODEL =============

model_input = Input(shape=(max_len,))
# TODO embedding matrix initialization
z = Embedding(max_features, emb_dim, weights=[p.embedding_matrix], trainable=trainable, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)

# convolutional
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding='valid',
                         activation='relu',
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation='relu')(z)
model_output = Dense(1, activation='sigmoid')(z)

model = Model(model_input, model_output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# ===============================

plot_losses = PlotLosses()
plot_accuracy = PlotAccuracy()
#
reduce_lr = ReduceLROnPlateau(monitor='val_loss')
callbacks_list = [plot_losses, reduce_lr, plot_accuracy]

print('Loading weights...')
previous_weights = "/mnt/shdstorage/for_classification/models_dir/model_1543849921.h5"
model.load_weights(previous_weights)

fit = False
if fit:
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
              callbacks=callbacks_list)

    path_to_weights = '/mnt/shdstorage/for_classification/models_dir/model_%s.h5' % (timing)
    path_to_architecture = "/mnt/shdstorage/for_classification/models_dir/architecture/model_%s.h5"
    model.save_weights(path_to_weights)
    model.save(path_to_architecture)
    print('Model is saved %s' % path_to_weights)

# score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
#
# print('Test score:', score)
# print('Test accuracy:', acc)

# ================= PRINT METRICS ======================

# train_res = model.predict(X_train)
# train_res = [i[0] for i in train_res]
# train_res = [1 if i > 0.5 else 0 for i in train_res]
# train_1, train_0 = calculate_all_metrics(y_train, train_res, 'TRAIN')
#
# test_res = model.predict(X_test)
# test_res = [i[0] for i in test_res]
# test_res = [1 if i > 0.5 else 0 for i in test_res]
# test_1, test_0 = calculate_all_metrics(y_test, test_res, 'TEST')

ver_res = model.predict(verification)
path_to_verification = verification_name
data = pd.read_csv(path_to_verification)
label = data['negative'].tolist()
ver_res = [i[0] for i in ver_res]
ver_res = [1 if i > 0.5 else 0 for i in ver_res]
verif_1, verif_0 = calculate_all_metrics(label, ver_res, 'VERIFICATION')

# ====================== PROCESSING VERIFICATION ============================

if path_to_verification:
    verification_set_name = path_to_verification.split('/')[-1].split('.')[0]
    data = pd.read_csv(path_to_verification)['text'].tolist()
    label = pd.read_csv(path_to_verification)['label'].tolist()
    ver_res = [i[0] for i in ver_res]
    positive_counter = 0
    negative_counter = 0
    pos = open('results/positive_%s_%s.txt' % (verification_set_name, timing), 'w')
    neg = open('results/negative_%s_%s.txt' % (verification_set_name, timing), 'w')
    for i in range(len(ver_res)):
        if ver_res[i] == 0:
            neg.write('\n[%s] ============================================================== [%s]\n' % (i, label[i]))
            neg.write('\n')
            neg.write(data[i])
            neg.write('\n')
        else:
            pos.write('\n[%s] ============================================================== [%s]\n' % (i, label[i]))
            pos.write('\n')
            pos.write(data[i])
            pos.write('\n')
    pos.close()
    neg.close()
    print('results/positive_%s_%s.txt' % (verification_set_name, timing))
    print('results/negative_%s_%s.txt' % (verification_set_name, timing))
