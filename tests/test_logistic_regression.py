import numpy as np

from mlalgorithms.modules.logistic_regression import LogisticRegression
from mlalgorithms.utils.dist_fun import mse


from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def test_logistic_regression():
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    log_reg = LogisticRegression(lr=0.0001, n_iters=10000)
    log_reg.fit(X_train, y_train)
    pred = log_reg.predict(X_test)

    def accuracy(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

    acc = accuracy(y_test, pred)

    print(f"Logistic regression acc: {acc}")
