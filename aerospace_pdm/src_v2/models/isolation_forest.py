# isolation_forest.py
# From-scratch implementation of Isolation Forest (Liu, Ting & Zhou, 2008) - the
# same unsupervised anomaly detection algorithm scikit-learn's IsolationForest
# provides. It is reimplemented here in plain numpy because this sandbox has no
# internet access to install scikit-learn; the algorithm and its guarantees are
# identical to the production library version.
#
# Core idea: anomalies are "few and different", so they are easier to ISOLATE with
# random splits than normal points are. Build many random binary trees, each
# splitting on a random feature at a random threshold; a point that reaches a leaf
# in very few splits (a short "path length") is likely an anomaly.

import numpy as np                          # all the math lives here

EULER_MASCHERONI = 0.5772156649


def _average_path_length(n):
    # c(n): the expected path length of an UNSUCCESSFUL search in a Binary Search
    # Tree of n points - used to normalize raw path lengths into a 0..1 anomaly score.
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + EULER_MASCHERONI) - (2.0 * (n - 1) / n)


class _IsolationTree:
    # One random binary tree. Each internal node picks ONE random feature and ONE
    # random split point between that feature's min and max value in the current subset.
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None            # which feature index this node splits on (None = leaf)
        self.split_value = None              # the threshold used to split
        self.left = None                     # left child (< split_value)
        self.right = None                    # right child (>= split_value)
        self.size = 0                        # how many training points landed in this node (leaves only need this)

    def fit(self, X, depth, rng):
        self.size = X.shape[0]
        n_features = X.shape[1]
        # stop growing (become a leaf) once we run out of depth budget, points, or
        # every feature is constant in this subset (nothing left to split on)
        if depth >= self.max_depth or X.shape[0] <= 1:
            return
        feature = rng.integers(0, n_features)
        col = X[:, feature]
        min_val, max_val = col.min(), col.max()
        if min_val == max_val:                 # this feature can't separate anything further here
            return
        split_value = rng.uniform(min_val, max_val)
        left_mask = col < split_value
        if left_mask.all() or (~left_mask).all():   # split didn't actually separate any points
            return

        self.split_feature = feature
        self.split_value = split_value
        self.left = _IsolationTree(self.max_depth)
        self.left.fit(X[left_mask], depth + 1, rng)
        self.right = _IsolationTree(self.max_depth)
        self.right.fit(X[~left_mask], depth + 1, rng)

    def path_length(self, x, depth=0):
        # walk one data point x down the tree, returning how many splits it took
        # (plus a correction term if we bottom out at a non-trivial-sized leaf)
        if self.split_feature is None:               # reached a leaf
            return depth + _average_path_length(self.size)
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


class IsolationForest:
    def __init__(self, n_trees=100, subsample_size=256, max_depth=None, random_state=0):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.subsample_size_used_ = None

    def fit(self, X):
        # X: 2D numpy array, shape (n_samples, n_features)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        psi = min(self.subsample_size, n)                 # can't subsample more rows than we have
        self.subsample_size_used_ = psi
        max_depth = self.max_depth or int(np.ceil(np.log2(max(psi, 2))))

        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=psi, replace=False)     # subsample WITHOUT replacement, as per the paper
            tree = _IsolationTree(max_depth)
            tree.fit(X[idx], depth=0, rng=rng)
            self.trees_.append(tree)
        return self

    def score(self, X):
        # returns an anomaly score per row: close to 1.0 = anomalous, close to 0.5 or
        # below = normal. This mirrors scikit-learn's IsolationForest.score_samples
        # convention (just without the sign flip it applies internally).
        c_psi = _average_path_length(self.subsample_size_used_)
        if c_psi == 0:
            c_psi = 1e-9
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            avg_path = np.mean([tree.path_length(X[i]) for tree in self.trees_])
            scores[i] = 2.0 ** (-avg_path / c_psi)
        return scores
