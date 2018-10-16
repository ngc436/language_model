import sklearn
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
import numpy as np

from sklearn import svm


def xgboost(params):
    num_estimators = int(params['n_estimators'])

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
        'scale_pos_weight': 


    }
    return best
