import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from mlalgorithms.utils.dist_fun import accuracy
from mlalgorithms.modules.decision_tree import DecisionTree


def run_decision_tree():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf_dt = DecisionTree(max_depth=10)
    clf_dt.fit(X_train, y_train)

    y_pred = clf_dt.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"Decision Tree acc: {acc}")
