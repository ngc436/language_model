# TODO: perform optimization with hyperopt with text accuracy maximization

# /tmp/model.h5,
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
# from matplotlib.backends.backend_pdf import PdfPages

#
set_session(session)

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

# config = tf.ConfigProto(
#     device_count={'GPU': 0}
# )
# session = tf.Session(config=config)
# set_session(session)

import time
import numpy as np

from vis_tools import *

np.random.seed(1337)  # for reproducibility

from language_model import QRNN
from metrics import calculate_all_metrics

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, Bidirectional, LSTM, Dropout
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers

import pandas as pd
from data import Processor

from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

# ======= LIME =======
from lime.lime_text import LimeTextExplainer

# data = pd.read_csv('/mnt/shdstorage/tmp/validation.csv')
# print(data)

# 89208 tokens


timing = str(int(time.time()))

batch_size = 128
# TODO: control the dictionary length
max_features = 224465  # 291837  # 172567 in the 3rd version # 228654 in the 4th version
max_len = 100  # reduce
emb_dim = 300

x_train_set_name = '/mnt/shdstorage/for_classification/X_train_6_no_ent.csv'
x_train_name = x_train_set_name.split('/')[-1].split('.')[0]
x_test_set_name = '/mnt/shdstorage/for_classification/X_test_6_no_ent.csv'
y_train_labels = '/mnt/shdstorage/for_classification/y_train_6_no_ent.csv'
y_test_labels = '/mnt/shdstorage/for_classification/y_test_6_no_ent.csv'
path_to_goal_sample = None

verification_name = '/mnt/shdstorage/for_classification/new_test.csv'

emb_type = 'fasttext_2'

X_train = pd.read_csv(x_train_set_name, header=None).values.tolist()
X_test = pd.read_csv(x_test_set_name, header=None).values.tolist()
y_train = pd.read_csv(y_train_labels, header=None).values.tolist()
y_test = pd.read_csv(y_test_labels, header=None).values.tolist()

path_to_verification = verification_name
data = pd.read_csv(path_to_verification)
# column processed_texts_no_tags is missing
texts = data.processed_text.tolist()
print(texts)

p = Processor(max_features=max_features, emb_type=emb_type, max_len=max_len, emb_dim=emb_dim)
p.fit_processor(x_train=X_train, x_train_name=x_train_name, other=texts)
# X_train, y_train = p.prepare_input(X_train, y_train)
# print('Train params: ', len(X_train), len(y_train))
# X_test, y_test = p.prepare_input(X_test, y_test)
# print('Test params: ', len(X_test), len(y_test))

verification = p.prepare_input(texts)

if path_to_goal_sample:
    goal = pd.read_csv(path_to_goal_sample)
    texts = goal.processed_text.tolist()
    goal = p.prepare_input(texts)


# ======= PARAMS =======
spatial_dropout = 0.1
window_size = 3
dropout = 0.3
kernel_regularizer = 1e-6
bias_regularizer = 1e-6
kernel_constraint = 6
bias_constraint = 6
loss = 'binary_crossentropy'
optimizer = 'adam'
model_type = 'Bidirectional'
lr = 0.00005  # changed from  0.00001
clipnorm = None
epochs = 25  # 20
weights = True
trainable = True
previous_weights = None
activation = 'sigmoid'
# ======= =======

print('Build model...')
model = Sequential()

if weights:
    model.add(Embedding(max_features, emb_dim, weights=[p.embedding_matrix], trainable=trainable))
else:
    model.add(Embedding(max_features, emb_dim))

model.add(SpatialDropout1D(spatial_dropout))
model.add(Bidirectional(QRNN(emb_dim//2, window_size=window_size, dropout=dropout,
                             kernel_regularizer=l2(kernel_regularizer), bias_regularizer=l2(bias_regularizer),
                             kernel_constraint=maxnorm(kernel_constraint), bias_constraint=maxnorm(bias_constraint))))
model.add(Dropout(dropout))
model.add(Dense(1, activation=activation))

plot_losses = PlotLosses()
plot_accuracy = PlotAccuracy()

reduce_rate = ReduceLROnPlateau(monitor='val_loss')
callbacks_list = [plot_losses, reduce_rate, plot_accuracy]

if clipnorm:
    optimizer = optimizers.Adam(lr=lr, clipnorm=clipnorm)
else:
    optimizer = optimizers.Adam(lr=lr)
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
print(model.summary())

# norm = math.sqrt(sum(numpy.sum(K.get_value(w)) for w in model.optimizer.weights))


print('Loading weights...')
previous_weights = "/mnt/shdstorage/for_classification/models_dir/model_1543691537.h5"
model.load_weights(previous_weights)


fit = False
if fit:
    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
              callbacks=callbacks_list)

    # ================================= MODEL SAVE =========================================

    path_to_weights = '/mnt/shdstorage/for_classification/models_dir/model_%s.h5' % (timing)
    path_to_architecture = "/mnt/shdstorage/for_classification/models_dir/architecture/model_%s.h5"
    model.save_weights(path_to_weights)
    model.save(path_to_architecture)
    print('Model is saved %s' % path_to_weights)

else:
    timing = previous_weights.split('/')[-1].split('_')[-1].split('.')[0]

#score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

# print('Test score:', score)
# print('Test accuracy:', acc)
# ============================== PRINT METRICS =====================================

# train_res = model.predict_classes(X_train)
# train_res = [i[0] for i in train_res]
# train_1, train_0 = calculate_all_metrics(y_train, train_res, 'TRAIN')
#
# test_res = model.predict_classes(X_test)
# test_res = [i[0] for i in test_res]
# test_1, test_0 = calculate_all_metrics(y_test, test_res, 'TEST')

ver_res = model.predict_classes(verification)
path_to_verification = verification_name
data = pd.read_csv(path_to_verification)
label = data['negative'].tolist()
ver_res = [i[0] for i in ver_res]
verif_1, verif_0 = calculate_all_metrics(label, ver_res, 'VERIFICATION')

# ======================= LOGS =======================


logs_name = 'logs/qrnn/%s.txt' % timing

try:
    stream = open('logs/qrnn/%s.txt' % timing, 'r')
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
            f.write('emb_type: %s, emb dim: %s, trainable: %s\n' % (emb_type, emb_dim, trainable))
        f.write('batch size: %s, max features: %s, max len: %s\n' % (
            batch_size, max_features, max_len))
        f.write('spatial dropout: %s, window size: %s, dropout: %s, activation: %s\n' % (
            spatial_dropout, window_size, dropout, activation))
        f.write('kernel regularizer: %s, bias regularizer: %s, kernel constraint: %s, bias constraint: %s\n' % (
            kernel_regularizer, bias_regularizer, 'maxnorm(%s)' % kernel_constraint, 'maxnorm(%s)' % bias_constraint))
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

# ====================== PROCESSING VERIFICATION ============================

if path_to_verification:
    verification_set_name = path_to_verification.split('/')[-1].split('.')[0]
    verification_res = model.predict_classes(verification)
    data = pd.read_csv(path_to_verification)['text'].tolist()
    label = pd.read_csv(path_to_verification)['label'].tolist()
    ver_res = [i[0] for i in verification_res]
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

# ====================== PROCESSING TEST ============================

if path_to_goal_sample:
    goal_set_name = path_to_goal_sample.split('/')[-1].split('.')[0]
    goal_res = model.predict_classes(goal)
    data = pd.read_csv(path_to_goal_sample)['text'].tolist()
    label = pd.read_csv(path_to_goal_sample)['label'].tolist()
    ver_res = [i[0] for i in goal_res]
    positive_counter = 0
    negative_counter = 0
    pos = open('results/positive_%s_%s.txt' % (goal_set_name, timing), 'w')
    neg = open('results/negative_%s_%s.txt' % (goal_set_name, timing), 'w')
    for i in range(len(ver_res)):
        if goal_res[i] == 0:
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
    print('results/positive_%s_%s.txt' % (goal_set_name, timing))
    print('results/negative_%s_%s.txt' % (goal_set_name, timing))


# ====== compute feature importance with LIME ======

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

train = data.processed_text.tolist()
texts = data.text.tolist()

for i in idx:
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

# ====== END ======


# text = data['text'].tolist()
# raw_text = data['raw_text'].tolist()
# pd.set_option('display.max_colwidth', -1)


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

# model_1542979324