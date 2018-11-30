import os.path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np

# metrics calculation
def calculate_all_metrics(y_true, y_pred, set_title):
    # print('Metrics for %s (1)' % set_title)
    result_1 = 'Metrics for %s (1)\nRecall: %s, Precision: %s, F1: %s' % (set_title, calculate_recall(y_true, y_pred),
                                                                          calculate_precision(y_true, y_pred),
                                                                          calculate_f1(y_true, y_pred))
    print(result_1)
    # print('Metrics for %s (0)' % set_title)
    y_true = 1 - (np.array(y_true))
    y_pred = 1 - (np.array(y_pred))
    result_0 = 'Metrics for %s (0)\nRecall: %s, Precision: %s, F1: %s' % (set_title, calculate_recall(y_true, y_pred),
                                                                          calculate_precision(y_true, y_pred),
                                                                          calculate_f1(y_true, y_pred))
    print(result_0)

    print()
    return result_1, result_0


def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


y = [0, 1, 1, 1, 0]
y_true = [1, 1, 1, 1, 0]
# TP = 3, TN = 1, FP = 1, FN =
# recall(1):
print(y)
print(1 - np.array(y))
print(1 - np.array(y_true))
