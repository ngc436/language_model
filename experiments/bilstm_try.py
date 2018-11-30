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
from keras.layers import Dense, Embedding, SpatialDropout1D, Dropout, Bidirectional, LSTM, GlobalMaxPool1D
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers

import pandas as pd
from data import Processor

timing = str(int(time.time()))

batch_size = 128
# cut words with 1, 2 appearances
# 61502 in version with no ent
# 123004 in cut version without entities
max_features = 200000  # 291837  # 172567 in the 3rd version # 228654 in the 4th version
max_len = 100
emb_dim = 300

x_train_set_name = '/mnt/shdstorage/for_classification/X_train_6_no_ent.csv'
x_train_name = x_train_set_name.split('/')[-1].split('.')[0]
x_test_set_name = '/mnt/shdstorage/for_classification/X_test_6_no_ent.csv'
y_train_labels = '/mnt/shdstorage/for_classification/y_train_6_no_ent.csv'
y_test_labels = '/mnt/shdstorage/for_classification/y_test_6_no_ent.csv'

verification_name = '/mnt/shdstorage/for_classification/new_test.csv'
#verification_name = '/mnt/shdstorage/tmp/verification_big.csv'
path_to_goal_sample = '/mnt/shdstorage/for_classification/new_test.csv'
# path_to_goal_sample = '/mnt/shdstorage/tmp/classif_tmp/comments_big.csv'
# file:///mnt/shdstorage/tmp/classif_tmp/comments_big_unlem.csv
#

emb_type = 'fasttext_2'

X_train = pd.read_csv(x_train_set_name, header=None).values.tolist()
X_test = pd.read_csv(x_test_set_name, header=None).values.tolist()
y_train = pd.read_csv(y_train_labels, header=None).values.tolist()
y_test = pd.read_csv(y_test_labels, header=None).values.tolist()

p = Processor(max_features=max_features, emb_type=emb_type, max_len=max_len, emb_dim=emb_dim)
p.fit_processor(x_train=X_train, x_test=X_test, x_train_name=x_train_name)
X_train, y_train = p.prepare_input(X_train, y_train)
print('Train params: ', len(X_train), len(y_train))
X_test, y_test = p.prepare_input(X_test, y_test)
print('Test params: ', len(X_test), len(y_test))

path_to_verification = verification_name
data = pd.read_csv(path_to_verification)
texts = data.processed_text.tolist()
verification = p.prepare_input(texts)

goal = pd.read_csv(path_to_goal_sample)
texts = goal.processed_text.tolist()
goal = p.prepare_input(texts)

# ======= PARAMS =======
spatial_dropout = None
window_size = 3
dropout = 0.7
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

if weights:
    model.add(Embedding(max_features, emb_dim, weights=[p.embedding_matrix], trainable=trainable))
else:
    model.add(Embedding(max_features, emb_dim))

# model.add(SpatialDropout1D(spatial_dropout))

model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
# model.add(GlobalMaxPool1D())
# model.add(Dense(1, activation=activation))
# TODO: Add batch normalization
model.add(Dropout(dropout))
model.add(Dense(1, activation=activation))

plot_losses = PlotLosses()
plot_accuracy = PlotAccuracy()
#
# early_stopping = EarlyStopping(monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss')
callbacks_list = [plot_losses, reduce_lr, plot_accuracy]  # early_stopping]

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

previous_weights = "models_dir/model_1543408507.h5"
model.load_weights(previous_weights)

fit = False
if fit:
    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
              callbacks=callbacks_list)

    # ================================= MODEL SAVE =========================================

    path_to_weights = "models_dir/model_%s.h5" % (timing)
    path_to_architecture = "models_dir/architecture/model_%s.h5"
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

# ======================= LOGS =======================

logs_name = 'logs/bilstm/%s.txt' % timing

try:
    stream = open('logs/bilstm/%s.txt' % timing, 'r')
    stream.close()
except:
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
            f.write('emb_type: %s, emb dim: %s, trainable: %s, time distributed: %s\n' % (
                emb_type, emb_dim, trainable, time_distributed))
        f.write('batch size: %s, max features: %s, max len: %s\n' % (
            batch_size, max_features, max_len))
        f.write('window size: %s, dropout: %s, recurrent dropout: %s, units: %s, activation: %s\n' % (
            window_size, dropout, recurrent_dropout, units, activation))
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


# ====================== PROCESSING TEST ============================

if path_to_goal_sample:
    goal_set_name = path_to_goal_sample.split('/')[-1].split('.')[0]
    goal_res = model.predict_classes(goal)
    data = pd.read_csv(path_to_goal_sample)['text']
    data = data.reset_index(drop=True)
    data = data.tolist()
    ver_res = [i[0] for i in goal_res]
    positive_counter = 0
    negative_counter = 0
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

print('results/positive_%s_%s.txt' % (goal_set_name, timing))
print('results/negative_%s_%s.txt' % (goal_set_name, timing))

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


raise ValueError

# ====== compute feature importance with LIME ======
from lime.lime_text import LimeTextExplainer


def predict_wrapper(text):
    '''

    :param text:
    :return answer: array of shape = [n_samples, n_classes]
    '''
    if isinstance(text, str):
        text = p.prepare_sequence(text)
        answer = np.zeros(shape=(1, 2))
        answer[0][0] = model.predict(text)[0][0]
        answer[0][1] = 1 - answer[0][0]
        print('Probability(class 1) = %s, Probability(class 0) = %s\n True class: %s' % (
            answer[0][0], answer[0][1], model.predict_classes(text)[0][0]))
    if isinstance(text, list):
        answer = np.zeros(shape=(len(text), 2))
        for i in range(len(text)):
            tmp = p.prepare_sequence(text[i])
            answer[i][0] = model.predict(tmp)[0][0]  # probability of class 1 (with negative comments)
            answer[i][1] = 1 - answer[i][0]  # probability of class 0 (without negative comments)
    return np.array(answer)


class_names = ['positive', 'negative']  #
explainer = LimeTextExplainer(class_names=class_names)  # class_names)
print()
num_features = 20
num_samples = 2500

data = pd.read_csv(path_to_goal_sample)
# data = data.sample(frac=1).reset_index(drop=True)

idx = [143, 103, 3309, 10625, 67, 42, 37, 23, 237, 267, 2002, 2025, 2099, 2140, 13, 140, 137, 2128, 263, 3481, 11696]

train = data.processed_text_no_org.tolist()
texts = data.text.tolist()

for i in range(len(texts)):
    tmp = train[i]
    print()
    print('====================')
    print()
    print(predict_wrapper(tmp))
    exp = explainer.explain_instance(text_instance=tmp, classifier_fn=predict_wrapper, num_features=num_features,
                                     num_samples=num_samples)
    print('[%s]' % i, exp.as_list())

    # TODO: change current behavior - explanations are rewritten now
    exp.save_to_file('lime_explanations/idx_%s_%s_ver_%s.html' % (i, num_samples, timing))
    weights = OrderedDict(exp.as_list())
    lime_weights = pd.DataFrame({'words': list(weights.keys()), 'weights': list(weights.values())})
    print(list(weights.keys()))
    print('True text: %s' % texts[i])
    print()
    print('lime_explanations/idx_%s_%s_ver_%s.html' % (i, num_samples, timing))

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

# model_1542708525.h5 -> model_1542711790.h5

# models_dir/model_1543411674.h5

# model_1543408507.h5
