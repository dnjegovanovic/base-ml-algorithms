import numpy as np

from .base_regression import BaseRegression


class LogisticRegression(BaseRegression):
    def __init__(self, lr=0.01, n_iters=10000) -> None:
        super().__init__(lr, n_iters, "logistic")

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(y_pred)
        y_cls = [1 if i > 0.5 else 0 for i in y_pred]

        return y_cls
