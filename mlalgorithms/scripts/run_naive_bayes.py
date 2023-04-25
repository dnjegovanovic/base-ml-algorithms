import numpy as np

from mlalgorithms.modules.naive_bayes import NaiveBayes

from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mlalgorithms.utils.dist_fun import accuracy


def run_naive_bayes():
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    pred = nb.predict(X_test)

    acc = accuracy(y_test, pred)

    print(f"Naive Bayes classification accuracy: {acc}")
