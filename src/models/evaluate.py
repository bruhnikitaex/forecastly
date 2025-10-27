import numpy as np

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom==0] = 1
    return 100.0 * np.mean(2 * np.abs(y_pred - y_true) / denom)
