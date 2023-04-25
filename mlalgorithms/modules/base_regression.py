import numpy as np

from scipy.special import expit


class BaseRegression:
    def __init__(self, lr=0.01, n_iters=10000, regression_type: str = "linear") -> None:
        self.learning_rate = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.regression_type = regression_type

    def fit(self, X, y):
        # initiaize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.random.random(n_features)
        self.bias = 0

        # gradient des
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            if self.regression_type == "logistic":
                y_pred = self._sigmoid(y_pred)

            # calc derivate by formula
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _sigmoid(self, z):
        return 1.0 / (1 + expit(-z))
