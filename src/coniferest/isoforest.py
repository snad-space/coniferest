from .coniferest import Coniferest, ConiferestEvaluator

__all__ = ["IsolationForest"]


class IsolationForest(Coniferest):
    """
    Isolation forest.

    This is a reimplementation of sklearn.ensemble.IsolationForest,
    which trains and evaluates much faster. It also supports multi-threading
    for evaluation (sample scoring).

    Parameters
    ----------
    n_trees : int, optional
        Number of trees in forest to build.

    n_subsamples : int, optional
        Number of subsamples to use for building the trees.

    max_depth : int or None, optional
        Maximal tree depth. If None, `log2(n_subsamples)` is used.

    n_jobs : int or None, optional
        Number of threads to use for evaluation. If None, use all available CPUs.

    chunksize : int, optional
        Size of the chunk to use for multithreading calculations. If 0, then automatic numer is used.

    random_seed : int or None, optional
        Seed for reproducibility. If None, random seed is used.
    """

    def __init__(
        self,
        n_trees=100,
        n_subsamples=256,
        max_depth=None,
        n_jobs=None,
        chunksize=None,
        random_seed=None,
    ):
        super().__init__(
            trees=[],
            n_subsamples=n_subsamples,
            max_depth=max_depth,
            n_jobs=n_jobs,
            chunksize=chunksize,
            random_seed=random_seed,
        )
        self.n_trees = n_trees
        self.evaluator = None

    def fit(self, data, labels=None):
        """
        Build the trees based on data.

        Parameters
        ----------
        data
            2-d array with features.

        labels
            Unused. Defaults to None.

        Returns
        -------
        self
        """
        self.trees = self.build_trees(data, self.n_trees)
        self.evaluator = ConiferestEvaluator(self)
        return self

    def score_samples(self, samples):
        """
        Compute scores for given samples.

        Parameters
        ----------
        samples
            2-d array with features.

        Returns
        -------
        1-d array with scores.
        """
        return self.evaluator.score_samples(samples)

    def fit_known(self, data, known_data=None, known_labels=None):
        return self.fit(data)

    def feature_signature(self, x):
        return self.evaluator.feature_signature(x)

    def feature_importance(self, x):
        return self.evaluator.feature_importance(x)

    def apply(self, x):
        """
        Apply the forest to X, return leaf indices.

        Parameters
        ----------
        x : ndarray shape (n_samples, n_features)
            2-d array with features.

        Returns
        -------
        x_leafs : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        return self.evaluator.apply(x)
