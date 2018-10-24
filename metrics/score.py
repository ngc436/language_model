from sklearn.metrics import recall_score, precision_score, f1_score


def calculate_all_metrics(y_true, y_pred, set_title):
    print('Metrics for %s' % set_title)
    print('Recall: %s, Precision: %s, F1: %s' % (calculate_recall(y_true, y_pred),
                                                 calculate_precision(y_true, y_pred),
                                                 calculate_f1(y_true, y_pred)))


def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)
