import numpy as np
from collections import Counter
from ..utils.dist_fun import entropy


class Node:
    def __init__(
        self, features=None, treshold=None, left=None, right=None, *, value=None
    ) -> None:
        self.feature = features
        self.treshold = treshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # grow tree
        self.n_feats = (
            X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        )  # num of features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stoping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_val = self._most_common_label(y)
            return Node(value=leaf_val)

        feat_idx = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_feat, best_tresh = self._best_criteria(X, y, feat_idx)
        left_idx, right_idx = self._split(X[:, best_feat], best_tresh)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feat, best_tresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_tresh = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            tresh = np.unique(X_col)
            for tr in tresh:
                gain = self._information_gain(y, X_col, tr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_tresh = tr

        return split_idx, split_tresh

    def _information_gain(self, y, X_col, split_tresh):
        # parent E
        paretn_entropy = entropy(y)
        # generate split
        left_idx, right_idx = self._split(X_col, split_tresh)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        # weighted avg childe E
        n_sampl = len(y)
        n_left_samp, n_right_sampl = len(left_idx), len(right_idx)
        e_l, e_r = entropy(y[left_idx]), entropy(y[right_idx])
        child_entropy = (n_left_samp / n_sampl) * e_l + (n_right_sampl / n_sampl) * e_r
        # retunr info
        inf_gain = paretn_entropy - child_entropy

        return inf_gain

    def _split(self, X_col, split_treshold):
        l_idx = np.argwhere(X_col <= split_treshold).flatten()
        r_idx = np.argwhere(X_col > split_treshold).flatten()

        return l_idx, r_idx

    def _most_common_label(self, y):
        counter = Counter(y)
        mc = counter.most_common(1)[0][0]
        return mc

    def predict(self, X):
        # traverese tree
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node: Node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.treshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
