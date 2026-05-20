import math

import numpy as np
from scipy.sparse import csr_array

from .calc_trees import calc_apply, calc_feature_delta_sum, calc_paths_sum, selector_dtype  # noqa
from .utils import average_path_length

__all__ = ["ForestEvaluator"]


class ForestEvaluator:
    selector_dtype = selector_dtype

    def __init__(self, samples, selectors, node_offsets, leaf_offsets, *, num_threads, sampletrees_per_batch):
        """
        Base class for the forest evaluators. Does the trivial job:
        * runs calc_paths_sum written in Rust,
        * helps to combine several trees' representations into two arrays.

        Parameters
        ----------
        samples
            Number of samples in every tree.

        selectors
            Array with all the nodes of all the trees.

        node_offsets
            Offsets for tree indices for node-sized arrays. For example
            for two trees of node-length `len1` and `len2` it would be:
            [0, len1, len1 + len2]

        leaf_offsets
            Offsets for tree indices for leaf-sized arrays.

        num_threads : int or None
            Number of threads to use for calculations. If None then
        """
        self.samples = samples

        self.selectors = selectors
        self.node_offsets = node_offsets
        self.leaf_offsets = leaf_offsets

        if num_threads is None or num_threads < 0:
            # Ask Rust's rayon to use all available threads
            self.num_threads = 0
        else:
            self.num_threads = num_threads

        self.sampletrees_per_batch = sampletrees_per_batch

    @property
    def batch_size(self):
        return math.ceil(self.sampletrees_per_batch / self.n_trees)

    @classmethod
    def combine_selectors(cls, selectors_list):
        """
        Combine several node arrays into one array of nodes and one array of
        start node_offsets.

        Parameters
        ----------
        selectors_list
            List of node arrays to combine.

        Returns
        -------
        np.ndarray of selectors
            Node array with all the nodes from all the trees.
        np.ndarray of int
            Array of tree offsets for node-arrays.
        np.ndarray of int
            Array of tree offsets for leaf-arrays.
        """
        lens = [len(sels) for sels in selectors_list]
        full_len = sum(lens)

        selectors = np.empty((full_len,), dtype=cls.selector_dtype)

        node_offsets = np.zeros((len(selectors_list) + 1,), dtype=np.uintp)
        node_offsets[1:] = np.add.accumulate(lens)

        for i in range(len(selectors_list)):
            selectors[node_offsets[i] : node_offsets[i + 1]] = selectors_list[i]

        # Assign a unique sequential index to every leaf
        # The index is used for weighted scores
        leaf_mask = selectors["feature"] < 0

        # Each offset tells how many leafs are in all previous trees
        leaf_offsets = np.zeros_like(node_offsets)
        leaf_offsets[1:] = np.cumsum(leaf_mask)[node_offsets[1:] - 1]
        leaf_count = leaf_offsets[-1]

        selectors["left"][leaf_mask] = np.arange(0, leaf_count)

        return selectors, node_offsets, leaf_offsets

    @property
    def n_trees(self):
        return self.node_offsets.shape[0] - 1

    @property
    def n_leaves(self):
        return self.leaf_offsets[-1]

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
                    self.selectors,
                    self.node_offsets,
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
            self.selectors,
            self.node_offsets,
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
            self.selectors,
            self.node_offsets,
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

    @classmethod
    def average_path_length(cls, n_nodes):
        """
        Average path length is abstracted because in different cases we may want to
        use a bit different formulas to make the exact match with other software.

        By default we use our own implementation.
        """
        return average_path_length(n_nodes)
