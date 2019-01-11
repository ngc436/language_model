from language_model import QRNN

import sklearn
import hyperopt
from hyperopt import STATUS_OK, fmin, hp, tpe
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle

from metrics import calculate_all_metrics

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

session = tf.Session(config=config)


def score(params):
    n_estimators = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(train_features, label=y_train)
    dvalid = xgb.DMatrix(test_features, label=y_test)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, n_estimators,
                          evals=watchlist,
                          verbose_eval=True)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    score = roc_auc_score(y_test, predictions)
    print("\tScore {0}\n\n".format(score))
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}


def score_model(params):
    n_estimators = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(train_features, label=y_train)
    dvalid = xgb.DMatrix(test_features, label=y_test)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, n_estimators,
                          evals=watchlist,
                          verbose_eval=True)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    score = roc_auc_score(y_test, predictions)
    print("\tScore {0}\n\n".format(score))
    loss = 1 - score
    return gbm_model


# 0.80767599266
def run_xgboost_experiments(seed=42):
    space = {
        # 'booster': hp.choice()
        'n_estimators': hp.quniform('n_estimators', 100, 500, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),  # step size shrinkage to prevent overfitting [0, 1]
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),  # minimum loss reduction [0, inf]
        'max_depth': hp.choice('max_depth', np.arange(1, 15, dtype=int)),  # max depth of the tree [0,inf]
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        # minimum sim of instance weight needed for a child [0,inf]
        # 'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1) # maximum delta step [0, inf]
        'subsample': hp.quniform('subsample', 0.2, 0.8, 0.1),  # subsample ratio of the training instances (0,1]
        'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 0.8, 0.1),
        # subsample ratio of columns whtn constructing each tree (0,1]
        'colsample_bylevel': hp.quniform('colsample_bylevel', 0.2, 0.8, 0.1),
        # subsample ratio of columns for each shlit in each level
        'lambda': hp.quniform('lambda', 1, 2, 0.2),  # L2 regularization term on weights
        'alpha': hp.quniform('alpha', 0, 1, 0.2),  # L1 regularization term on weights
        # 'tree_method': hp.choice('tree_method', ['auto', 'approx', 'gpu_exact', 'gpu_hist']),
        # the tree construction algorithm
        # 'sketch_eps':hp.quniform('sketch_eps', 0.001, 0.03, 0.001), # for tree_method='approx', number of bins (0,1)
        'scale_pos_weight': hp.quniform('scale_pos_weights', 0.5, 1.5, 0.1),
        # control the balance of positive and negative instances
        # 'updater'
        # 'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),  #
        'seed': 42
    }
    best = fmin(score, space, algo=tpe.suggest, max_evals=50)
    return best


def plot_importance():
    pass


# woring with elmo embeddings
print('Train')
train_features = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_sent_train_v7.pkl', 'rb'))
y_train = pd.read_csv('/mnt/shdstorage/for_classification/train_v7.csv')['label'].values.tolist()
# y_train = [[val] for val in y_train]

print('Test')
test_features = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_sent_test_v7.pkl', 'rb'))
y_test = pd.read_csv('/mnt/shdstorage/for_classification/test_v7.csv')['label'].values.tolist()
# y_test = [[val] for val in y_test]


# train_features = pd.read_csv()
# print('Preparing train...')
# theta = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/theta_50.csv')
# theta_trans = theta.T
# indexing = theta_trans.index.values.tolist()[1:]
# train_features = theta_trans.values[1:]
# indexing = [int(x) for x in indexing]
#
# print('Preparing test...')
# theta_test = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/theta_test_50.csv')
# theta_test_trans = theta_test.T
# indexing_test = theta_test_trans.index.values.tolist()[1:]
# valid_features = theta_test_trans.values[1:]
# indexing_test = [int(x) for x in indexing_test]
#
# print('Preparing y train...')
# y_train = []
# y_train_tmp = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_train.csv', header=None).values.tolist()
# for ix in indexing:
#     y_train += [y_train_tmp[ix][0]]
# print(y_train[:10])
#
# print('Preparing y test...')
# y_valid = []
# y_valid_tmp = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/y_test.csv', header=None).values.tolist()
# for ix in indexing_test:
#     y_valid += [y_valid_tmp[int(ix)][0]]
# print(y_valid[:10])
#
# print('Preparing verification')

# best_hyperparams = run_xgboost_experiments()
# print("The best hyperparameters are: ", "\n")
# print(best_hyperparams)

# {'alpha': 0.8, 'colsample_bylevel': 0.6000000000000001, 'colsample_bytree': 0.7000000000000001, 'eta': 0.07500000000000001, 'gamma': 0.55, 'lambda': 1.6, 'max_depth': 7, 'min_child_weight': 9.0, 'n_estimators': 490.0, 'scale_pos_weights': 1.5, 'subsample': 0.7000000000000001}

# {'alpha': 0.0, 'colsample_bylevel': 0.4, 'colsample_bytree': 0.6000000000000001, 'eta': 0.15000000000000002, 'gamma': 0.7000000000000001, 'lambda': 1.6, 'max_depth': 11, 'min_child_weight': 2.0, 'n_estimators': 434.0, 'scale_pos_weights': 1.0, 'subsample': 0.8}
#

params = {
    'alpha': 0.8,
    'colsample_bylevel': 0.4,
    'colsample_bytree': 0.6,
    'eta': 0.075,
    'gamma': 0.55,
    'lambda': 1.6,
    'max_depth': 7,
    'min_child_weight': 9,
    'n_estimators': 490,
    'scale_pos_weights': 1.5,
    'subsample': 0.7
}

# model = score_model(params)
#
# with open('/mnt/shdstorage/for_classification/xgboost_model.pickle', 'wb') as handle:
#     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/mnt/shdstorage/for_classification/xgboost_model.pickle', 'rb') as handle:
    model = pickle.load(handle)


# print('set 3')
# train_features = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_ver_set_3.pkl', 'rb'))
# dtrain = xgb.DMatrix(train_features)
# data = model.predict(dtrain)
# print(data)
# prediction = np.array(data)
# print(prediction)
# prediction = prediction >= 0.5
# prediction = prediction.astype(int)
# set_3 = pd.read_csv('/mnt/shdstorage/for_classification/labeled/stated_workers-users-comments-tokens_gt_25_posts_gt_2_and_lt_100_labeled.csv')
# set_3['elmo_label'] = prediction
# set_3.to_csv('/mnt/shdstorage/for_classification/labeled/stated_workers-users-comments-tokens_gt_25_posts_gt_2_and_lt_100_labeled.csv')


# print('set 1')
# train_features = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_ver_set_1.pkl', 'rb'))
# dtrain = xgb.DMatrix(train_features)
# data = model.predict(dtrain)
# print(data)
# prediction = np.array(data)
# print(prediction)
# prediction = prediction >= 0.5
# prediction = prediction.astype(int)
# set_1 = pd.read_csv('/mnt/shdstorage/for_classification/labeled/stated_workers-users-comments-tokens_gt_10_posts_gt_2_and_lt_100_labeled.csv')
# set_1['elmo_label'] = prediction
# set_1.to_csv('/mnt/shdstorage/for_classification/labeled/stated_workers-users-comments-tokens_gt_10_posts_gt_2_and_lt_100_labeled.csv')

# print('set 2')
# train_features = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_ver_set_2.pkl', 'rb'))
# dtrain = xgb.DMatrix(train_features)
# data = model.predict(dtrain)
# print(data)
# prediction = np.array(data)
# print(prediction)
# prediction = prediction >= 0.5
# prediction = prediction.astype(int)
# set_2 = pd.read_csv('/mnt/shdstorage/for_classification/labeled/stated_workers-users-comments-tokens_gt_5_posts_gt_2_and_lt_100_labeled.csv')
# set_2['elmo_label'] = prediction
# set_2.to_csv('/mnt/shdstorage/for_classification/labeled/stated_workers-users-comments-tokens_gt_5_posts_gt_2_and_lt_100_labeled.csv')

# print('set 1')
# test_features = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_sent_test_v7.pkl', 'rb'))
# label = pd.read_csv('/mnt/shdstorage/for_classification/test_v7.csv')['label'].values.tolist()
# dtest = xgb.DMatrix(test_features)
# data = model.predict(dtest)
# print(data)
# prediction = np.array(data)
# print(prediction)
# prediction = prediction >= 0.5
# prediction = prediction.astype(int)
#
# test_1, test_0 = calculate_all_metrics(label, prediction, 'TEST')
#
# print('set 2')

print('Ver')
ver_features = pickle.load(open('/mnt/shdstorage/for_classification/elmo/elmo_sent_ver.pkl', 'rb'))
label = pd.read_csv('/mnt/shdstorage/for_classification/new_test.csv')['negative'].values.tolist()
dver = xgb.DMatrix(ver_features)
data = model.predict(dver)
print(data)
prediction = np.array(data)
print(prediction)
prediction = prediction >= 0.5
prediction = prediction.astype(int)

verif_1, verif_0 = calculate_all_metrics(label, prediction, 'VERIFICATION')

