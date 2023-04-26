import numpy as np

from mlalgorithms.modules.svm import SVM

from sklearn import datasets

from mlalgorithms.utils.dist_fun import visualize_svm


def run_svm():
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )

    y = np.where(y == 0, -1, 1)

    clf_svn = SVM()
    clf_svn.fit(X, y)

    print(f"SVM w:{clf_svn.w}, b:{clf_svn.b}")
    visualize_svm(X, clf_svn.w, clf_svn.b, y)
