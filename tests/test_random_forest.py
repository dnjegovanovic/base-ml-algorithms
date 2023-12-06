import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from mlalgorithms.utils.dist_fun import accuracy
from mlalgorithms.modules.random_forest import RandomForest


if __name__ == "__main__":

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = RandomForest(n_trees=3, max_depth=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)