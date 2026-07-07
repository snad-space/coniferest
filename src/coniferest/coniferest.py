from abc import ABC, abstractmethod
from warnings import warn

import numpy as np

from ._core import Tree, build_trees  # noqa
from .evaluator import ForestEvaluator

__all__ = ["Coniferest", "ConiferestEvaluator", "Tree"]


class Coniferest(ABC):
    """
    Base class for the forests in the package. It settles the basic
    low-level machinery with the Rust tree builder, used here.

    Parameters
    ----------
    trees : list or None, optional
        List with the trees in the forest. If None, then empty list is used.

    n_subsamples : int, optional
        Subsamples to use for the training.

    max_depth : int or None, optional
        Maximum depth of the trees in use. If None, then `log2(n_subsamples)` is used.

    n_jobs : int, default=-1
        Number of threads to use for building and scoring. If -1, use all
        available CPUs.

    random_seed : int or None, optional
        Seed for the reproducibility. If None, then random seed is used.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Available only after
        the forest has been built (i.e. after :meth:`build_trees`,
        :meth:`fit`, or :meth:`fit_known` has been called).
    """

    def __init__(
        self, trees=None, n_subsamples=256, max_depth=None, n_jobs=-1, random_seed=None, sampletrees_per_batch=1 << 20
    ):
        self.trees = trees or []
        self.n_subsamples = n_subsamples
        self.max_depth = max_depth or int(np.log2(n_subsamples))

        self.n_jobs = n_jobs
        self.sampletrees_per_batch = sampletrees_per_batch

        self.rng = np.random.default_rng(random_seed)

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        if len(self.trees) == 0:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute n_features_in_")
        return self.trees[0].n_features

    @property
    def num_threads(self):
        """`n_jobs` converted to the Rust extension conventions: 0 means all CPUs."""
        if self.n_jobs is None or self.n_jobs < 0:
            return 0
        return self.n_jobs

    @staticmethod
    def _prepare_data(data):
        """Convert data to a C-contiguous float32/float64 array."""
        data = np.asarray(data)
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)
        return np.ascontiguousarray(data)

    def build_trees(self, data, n_trees):
        """
        Just build `n_trees` trees from supplied `data`.

        Trees are built in parallel, each tree from its own random subsample
        of `data` rows. Random seeds for the trees are sampled in advance,
        so the result is reproducible and does not depend on the number of
        threads.

        Parameters
        ----------
        data
            Features.

        n_trees
            Number of trees to build

        Returns
        -------
        List of trees.
        """
        data = self._prepare_data(data)
        n_population, _n_features = data.shape

        n_samples = self.n_subsamples
        if n_samples > n_population:
            msg1 = "population should be greater or equal than subsamples number"
            msg2 = f"got n_population < n_subsamples ({n_population} < {n_samples})"
            msg3 = f"assuming n_subsamples = {n_population}"
            warn(msg1 + ", " + msg2 + ", " + msg3)
            n_samples = n_population

        seed = int(self.rng.integers(0, 1 << 64, dtype=np.uint64))

        return build_trees(data, seed, n_trees, n_samples, int(self.max_depth), num_threads=self.num_threads)

    def build_one_tree(self, data):
        """
        Build just one tree from the whole `data`, without subsampling.

        Parameters
        ----------
        data
            Features to build that one tree of.

        Returns
        -------
        A tree.
        """
        data = self._prepare_data(data)

        seed = int(self.rng.integers(0, 1 << 64, dtype=np.uint64))
        (tree,) = build_trees(data, seed, 1, data.shape[0], int(self.max_depth), num_threads=1)

        return tree

    @staticmethod
    def _validate_known_data(known_data=None, known_labels=None):
        known_data = np.asarray(known_data) if known_data is not None else None
        known_labels = np.asarray(known_labels) if known_labels is not None else None

        if (known_data is None) != (known_labels is None):
            raise ValueError("known_data and known_labels must be provided together or both be None")

        if (known_data is not None) and len(known_data) != len(known_labels):
            raise ValueError(
                f"known_data and known_labels must have the same length: {len(known_data)} != {len(known_labels)}"
            )

        return known_data, known_labels

    @abstractmethod
    def fit(self, data, labels=None):
        """
        Fit to the applied data.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_known(self, data, known_data=None, known_labels=None):
        """
        Fit to the applied data with priors.
        """
        raise NotImplementedError()

    @abstractmethod
    def score_samples(self, samples):
        """
        Evaluate scores for samples.
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_signature(self, x):
        raise NotImplementedError()

    @abstractmethod
    def feature_importance(self, x):
        raise NotImplementedError()


class ConiferestEvaluator(ForestEvaluator):
    """
    Fast evaluator of scores for Coniferests.

    Parameters
    ----------
    coniferest : Coniferest
        The forest for building the evaluator from.
    """

    def __init__(self, coniferest):
        super().__init__(
            samples=coniferest.n_subsamples,
            trees=coniferest.trees,
            num_threads=coniferest.n_jobs,
            sampletrees_per_batch=coniferest.sampletrees_per_batch,
        )
