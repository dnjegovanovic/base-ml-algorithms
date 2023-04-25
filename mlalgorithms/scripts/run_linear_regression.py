import numpy as np

from mlalgorithms.modules.linear_regression import LinearRegression
from mlalgorithms.utils.dist_fun import mse


from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def run_linearregresion():
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    print(f"train data shape: {X_train.shape}")
    print(f"target data shape: {y_train.shape}")

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
    plt.show()

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    pred = regressor.predict(X_test)

    mse_val = mse(y_test, pred)

    print(f"MSE val: {mse_val}")
    pred_line = regressor.predict(X)

    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, pred_line, color="black", linewidth=2, label="prediction")
    plt.show()
