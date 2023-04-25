import numpy as np

from mlalgorithms.modules.knn import KNN

from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def test_knn():
    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    print(f"train data shape: {X_train.shape}")
    print(f"target data shape: {y_train.shape}")

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", s=20)
    plt.show()

    knn = KNN(5)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    acc = np.sum(pred == y_test) / len(y_test)
    print(f"Accuracy: {acc}")
