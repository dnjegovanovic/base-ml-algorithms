import numpy as np

from .base_regression import BaseRegression


class LinearRegression(BaseRegression):
    def __init__(self, lr: float = 0.001, n_iters: int = 1000) -> None:
        super().__init__(lr, n_iters)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
