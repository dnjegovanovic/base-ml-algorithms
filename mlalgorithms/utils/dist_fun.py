import numpy as np


def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc
