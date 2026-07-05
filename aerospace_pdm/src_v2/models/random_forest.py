# random_forest.py
# From-scratch implementation of a Random Forest Regressor - the same bagged
# decision-tree-ensemble algorithm scikit-learn's RandomForestRegressor provides.
# Reimplemented here in plain numpy because this sandbox has no internet access to
# install scikit-learn; the algorithm (bootstrap sampling + feature bagging +
# variance-reduction splits + averaging) is the same one used in production
# RUL/PdM systems built on scikit-learn.
#
# For speed on larger datasets, split-finding uses HISTOGRAM BINNING (a fixed
# number of candidate thresholds per feature, taken from that feature's quantiles
# in the current node) instead of testing every possible threshold - the same
# trick production gradient-boosting libraries (e.g. LightGBM, XGBoost) use to
# keep training fast on large data.

import numpy as np


class _DecisionTreeRegressor:
    def __init__(self, max_depth, min_samples_split, min_samples_leaf,
                 max_features_fraction, n_bins):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_fraction = max_features_fraction
        self.n_bins = n_bins
        # tree structure, filled in during fit()
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None            # leaf prediction (mean of y in this node)

    def fit(self, X, y, depth, rng):
        self.value = float(y.mean())                      # every node stores a fallback leaf value
        n_samples, n_features = X.shape

        if (depth >= self.max_depth or n_samples < self.min_samples_split or
                n_samples < 2 * self.min_samples_leaf or np.all(y == y[0])):
            return                                          # stop here: this is a leaf

        best_gain, best_feature, best_threshold = 0.0, None, None
        n_candidate_features = max(1, int(np.ceil(n_features * self.max_features_fraction)))
        candidate_features = rng.choice(n_features, size=n_candidate_features, replace=False)

        parent_sse = np.sum((y - y.mean()) ** 2)             # sum of squared error before splitting

        for feature in candidate_features:
            col = X[:, feature]
            # histogram binning: try candidate thresholds at evenly spaced QUANTILES
            # of this feature's values in the current node, instead of every unique value
            quantiles = np.linspace(0, 1, self.n_bins + 2)[1:-1]
            thresholds = np.unique(np.quantile(col, quantiles))
            for threshold in thresholds:
                left_mask = col < threshold
                n_left, n_right = left_mask.sum(), (~left_mask).sum()
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                y_left, y_right = y[left_mask], y[~left_mask]
                sse = np.sum((y_left - y_left.mean()) ** 2) + np.sum((y_right - y_right.mean()) ** 2)
                gain = parent_sse - sse                       # bigger drop in squared error = better split
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        if best_feature is None:                              # no split improved on the leaf -> stay a leaf
            return

        self.feature = best_feature
        self.threshold = best_threshold
        left_mask = X[:, best_feature] < best_threshold

        self.left = _DecisionTreeRegressor(self.max_depth, self.min_samples_split,
                                            self.min_samples_leaf, self.max_features_fraction, self.n_bins)
        self.left.fit(X[left_mask], y[left_mask], depth + 1, rng)
        self.right = _DecisionTreeRegressor(self.max_depth, self.min_samples_split,
                                             self.min_samples_leaf, self.max_features_fraction, self.n_bins)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1, rng)

    def predict_row(self, x):
        if self.feature is None:                              # leaf
            return self.value
        if x[self.feature] < self.threshold:
            return self.left.predict_row(x)
        return self.right.predict_row(x)

    def predict(self, X):
        return np.array([self.predict_row(x) for x in X])


class RandomForestRegressor:
    def __init__(self, n_trees=40, max_depth=8, min_samples_split=10, min_samples_leaf=5,
                 max_features_fraction=0.7, n_bins=16, random_state=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features_fraction = max_features_fraction
        self.n_bins = n_bins
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, size=n)                    # bootstrap sample WITH replacement (bagging)
            tree = _DecisionTreeRegressor(self.max_depth, self.min_samples_split,
                                           self.min_samples_leaf, self.max_features_fraction, self.n_bins)
            tree.fit(X[idx], y[idx], depth=0, rng=rng)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        # average every tree's prediction -> this averaging is what makes a RANDOM
        # FOREST more stable than any single decision tree (reduces overfitting/variance)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])
        return all_preds.mean(axis=0)

    def feature_importances(self, n_features):
        # how many times each feature was used as a split point, across every tree -
        # a simple but standard proxy for feature importance (more usage = more useful).
        importances = np.zeros(n_features)

        def walk(node):
            if node is None or node.feature is None:            # leaf node -> nothing to count
                return
            importances[node.feature] += 1
            walk(node.left)
            walk(node.right)

        for tree in self.trees_:
            walk(tree)
        total = importances.sum()
        return importances / total if total > 0 else importances
