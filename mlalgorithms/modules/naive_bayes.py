import numpy as np


class NaiveBayes:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # inint mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)

            # frequency of specific class in data
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posterios = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            cls_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + cls_conditional

            posterios.append(posterior)

        return self._classes[np.argmax(posterios)]

    def _pdf(self, cls_idx, x):
        # probability dens func, gaussian
        mean = self._mean[cls_idx]
        var = self._var[cls_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
