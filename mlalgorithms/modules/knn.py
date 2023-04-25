import numpy as np

from ..utils.dist_fun import euclidean_dist


from collections import Counter


class KNN:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_label = [self._predict(x) for x in X]
        return np.array(predicted_label)

    def _predict(self, x):
        # compute dist
        dist = [euclidean_dist(x, x_train) for x_train in self.X_train]
        # get k  nearest samples, labesl
        k_near_ind = np.argsort(dist)[: self.k]
        k_ner_labels = [self.y_train[i] for i in k_near_ind]
        # vore, most commmon class label

        most_common_label = Counter(k_ner_labels).most_common(1)

        return most_common_label[0][0]
