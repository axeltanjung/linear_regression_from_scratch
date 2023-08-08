import numpy as np

def mean_squarred_error(y_act, y_pred):
    return np.mean((y_act-y_pred)**2)