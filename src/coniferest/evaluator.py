import math

import numpy as np
from scipy.sparse import csr_array

from ._core import Tree, calc_apply, calc_feature_delta_sum, calc_paths_sum  # noqa
from .utils import average_path_length

__all__ = ["ForestEvaluator"]


class ForestEvaluator:
    def __init__(self, samples, trees, *, num_threads, sampletrees_per_batch):
        """
        Base class for the forest evaluators. Does the trivial job:
        * runs calc_paths_sum written in Rust,
        * combines per-tree leaf values into a global array.

        Parameters
        ----------
        samples
            Number of samples in every tree.

        trees
            List of `coniferest._core.Tree` objects.

        num_threads : int or None
            Number of threads to use for calculations. If None or negative,
            all available CPUs are used.

        sampletrees_per_batch : int
            Target number of sample-tree pairs per parallel batch.
        """
        self.samples = samples
        self.trees = list(trees)

        if num_threads is None or num_threads < 0:
            # Ask Rust's rayon to use all available threads
            self.num_threads = 0
        else:
            self.num_threads = num_threads

        self.sampletrees_per_batch = sampletrees_per_batch

    @property
    def batch_size(self):
        return math.ceil(self.sampletrees_per_batch / self.n_trees)

    @staticmethod
    def combine_leaf_values(trees):
        """Concatenate per-tree leaf values into one global-leaf-indexed array."""
        if len(trees) == 0:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate([tree.leaf_values() for tree in trees])

    @property
    def n_trees(self):
        return len(self.trees)

    @property
    def n_leaves(self):
        return int(sum(tree.n_leaves for tree in self.trees))

    def score_samples(self, x):
        """
        Perform the computations.

        Parameters
        ----------
        x
            Features to calculate scores of. Should be C-contiguous for performance.

        Returns
        -------
        Array of scores.
        """
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        return -(
            2
            ** (
                -calc_paths_sum(
                    self.trees,
                    x,
                    num_threads=self.num_threads,
                    batch_size=self.batch_size,
                )
                / (self.average_path_length(self.samples) * self.n_trees)
            )
        )

    def _feature_delta_sum(self, x):
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        return calc_feature_delta_sum(
            self.trees,
            x,
            num_threads=self.num_threads,
            batch_size=self.batch_size,
        )

    def feature_signature(self, x):
        delta_sum, hit_count = self._feature_delta_sum(x)

        return delta_sum / hit_count / self.average_path_length(self.samples)

    def feature_importance(self, x):
        delta_sum, hit_count = self._feature_delta_sum(x)

        return np.sum(delta_sum, axis=0) / np.sum(hit_count, axis=0) / self.average_path_length(self.samples)

    def _dense_apply(self, x):
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        return calc_apply(
            self.trees,
            x,
            num_threads=self.num_threads,
            batch_size=self.batch_size,
        )

    def _sparse_apply(self, x):
        dense = self._dense_apply(x)

        n_samples, n_estimators = dense.shape
        n_leaves = self.n_leaves

        data = np.ones(n_samples * n_estimators, dtype=np.float32)
        indices = dense.ravel()
        indptr = np.arange(0, n_samples * n_estimators + 1, n_estimators)

        return csr_array((data, indices, indptr), shape=(n_samples, n_leaves))

    def apply(self, x, output=None):
        """
        Apply the forest to X, return leaf indices.

        Parameters
        ----------
        x : ndarray shape (n_samples, n_features)
            2-d array with features.
        output : {"dense", "sparse"}, default="dense"
            If "dense", returns a dense array of leaf indices per tree.
            If "sparse", returns a sparse CSR matrix of shape (n_samples, n_leaves)
            where each row has non-zero entries for leaves reached by the sample.

        Returns
        -------
        x_leafs : ndarray of shape (n_samples, n_estimators) or csr_matrix of shape (n_samples, n_leaves)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in (dense format).
            If output="sparse", returns a sparse matrix with 1.0 in entries where
            sample reaches the leaf.
        """

        if output is None:
            output = "dense"

        if output not in ["dense", "sparse"]:
            raise ValueError("output is neither dense nor sparse")

        if output == "dense":
            return self._dense_apply(x)
        elif output == "sparse":
            return self._sparse_apply(x)

    def _common_leaf_ratio_distance(self, x, y):
        n_trees = self.n_trees

        x = np.atleast_2d(x)
        x_leaves = self._sparse_apply(x)

        if y is not None:
            y = np.atleast_2d(y)
            y_leaves = self._sparse_apply(y)
        else:
            y_leaves = x_leaves

        return 1 - (x_leaves @ y_leaves.T).toarray() / n_trees

    def distance(self, x, y, *, method=None):
        """
        Compute distance matrix between samples based on leaf co-occurrence.

        The distance is defined as 1 minus the fraction of trees where two samples
        land in the same leaf. This gives a measure of dissimilarity between
        samples based on their paths through the forest.

        Parameters
        ----------
        x : ndarray shape (n_samples_x, n_features) or (n_features,)
            Input samples. If 1-D, treated as a single sample.
        y : ndarray shape (n_samples_y, n_features) or (n_features,), optional
            Second set of samples for pairwise distance. If None (default),
            computes distances between all pairs in x.
        method : {"common_leaf_ratio"}, default="common_leaf_ratio"
            Distance computation method. Currently only "common_leaf_ratio"
            is supported.

        Returns
        -------
        distances : ndarray shape (n_samples_x, n_samples_y)
            Distance matrix where distances[i, j] is the distance between
            the i-th sample in x and j-th sample in y.
            If y is None, returns a square symmetric matrix of shape
            (n_samples_x, n_samples_x).

        Raises
        ------
        ValueError
            If method is not one of the known methods.
        """
        KNOWN_METHODS = ["common_leaf_ratio"]

        if method is None:
            method = "common_leaf_ratio"

        if method not in KNOWN_METHODS:
            raise ValueError(f"method is not one of {', '.join(KNOWN_METHODS)}.")

        return self._common_leaf_ratio_distance(x, y)

    @classmethod
    def average_path_length(cls, n_nodes):
        """
        Average path length is abstracted because in different cases we may want to
        use a bit different formulas to make the exact match with other software.

        By default we use our own implementation.
        """
        return average_path_length(n_nodes)
