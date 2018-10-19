import sklearn
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split


def score(params):
    n_estimators = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(train_features, label=y_train)
    dvalid = xgb.DMatrix(valid_features, label=y_valid)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    score = roc_auc_score(y_valid, predictions)
    print("\tScore {0}\n\n".format(score))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}

#
def run_xgboost_experiments(seed=42):
    space = {
        # 'booster': hp.choice()
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),# step size shrinkage to prevent overfitting [0, 1]
        'gamma': hp.quniform('gamma', 0.5, 2, 0.05), # minimum loss reduction [0, inf]
        'max_depth': hp.choice('max_depth', np.arange(1, 100, dtype=int)),# max depth of the tree [0,inf]
        'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1), # minimum sim of instance weight needed for a child [0,inf]
        #'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1) # maximum delta step [0, inf]
        'subsample': hp.quniform('subsample', 0.2, 0.8, 0.1), # subsample ratio of the training instances (0,1]
        'colsample_bytree':hp.quniform('colsample_bytree', 0.2,0.8,0.1), # subsample ratio of columns whtn constructing each tree (0,1]
        'colsample_bylevel':hp.quniform('colsample_bylevel',0.2, 0.8, 0.1), # subsample ratio of columns for each shlit in each level
        'lambda': hp.quniform('lambda', 1, 2, 0.2), # L2 regularization term on weights
        'alpha': hp.quniform('alpha', 0, 1, 0.2), # L1 regularization term on weights
        'tree_method':hp.choice('tree_method',['auto','approx','gpu_exact','gpu_hist']), # the tree construction algorithm
        #'sketch_eps':hp.quniform('sketch_eps', 0.001, 0.03, 0.001), # for tree_method='approx', number of bins (0,1)
        'scale_pos_weight': hp.quniform('scale_pos_weights', 0.5, 1.5, 0.1), # control the balance of positive and negative instances
        #'updater'
        'grow_policy':hp.choice('grow_policy', ['depthwise','lossguide']), #
        'seed':42
    }
    best = fmin(score, space, algo=tpe.suggest,max_evals=100)
    return best

train_features = pd.read_csv()