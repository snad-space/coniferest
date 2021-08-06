from .coniferest import Coniferest, ConiferestEvaluator
from .experiment import AnomalyDetector


class IsolationForest(Coniferest):
    def __init__(self, n_trees=100, n_subsamples=256, max_depth=None, random_seed=None):
        """
        Isolation forest. Just isolation forest.

        Parameters
        ----------
        n_trees
            Number of trees in forest to build.

        n_subsamples
            Number of subsamples to use for building the trees.

        max_depth
            Maximal tree depth.

        random_seed
            Seed for reproducibility.
        """
        super().__init__(trees=[],
                         n_subsamples=n_subsamples,
                         max_depth=max_depth,
                         random_seed=random_seed)
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


class IsolationForestAnomalyDetector(AnomalyDetector):
    def __init__(self, isoforest, title='Isolation Forest'):
        """
        Anomaly detection with isolation forest.
        Anomaly detectors are the wrappers around forests for
        interaction with the AnomalyDetectionExperiment.

        Parameters
        ----------
        isoforest
            Regressor to detect anomalies with.

        title
            Title for plots.
        """
        super().__init__(title)
        self.isoforest = isoforest

    def train(self, data):
        """
        Training is just building the trees out of the supplied data.

        Parameters
        ----------
        data
            2-d array with features to build trees of.

        Returns
        -------
        Trained forest.
        """
        return self.isoforest.fit(data)

    def score(self, data):
        """
        Compute the scores for the data.

        Parameters
        ----------
        data
            2-d array with features to compute scores of.

        Returns
        -------
        Scores.
        """
        return self.isoforest.score_samples(data)

    def observe(self, point, label):
        """
        Learn about new data. Observe it!

        Parameters
        ----------
        point
            Features of the data point.

        label
            True label of the data point.

        Returns
        -------
        Whether the regressor changed itself.
        """
        super().observe(point, label)
        # done nothing, it's classic, you know
        return False
