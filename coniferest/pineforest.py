import numpy as np

from .coniferest import Coniferest, ConiferestEvaluator
from .experiment import AnomalyDetector
from .utils import average_path_length
from .datasets import Label

from sklearn.tree._tree import DTYPE as TreeDTYPE  # noqa


class PineForest(Coniferest):
    def __init__(self,
                 n_trees=100,
                 n_subsamples=256,
                 max_depth=None,
                 n_spare_trees=400,
                 regenerate_trees=False,
                 weight_ratio=1,
                 random_seed=None):
        """
        Pine Forests are filtering isolation forests. That's a simple concept
        of incorporating prior knowledge about what is anomalous and what is not.

        Standard fit procedure with two parameters works exactly the same as the
        isolation forests' one. It differs when we supply additional parameter
        `labels`, than the behaviour changes. At that case fit generates additional
        not only `n_trees` but with additional `n_spare_trees` and then filters out
        `n_spare_trees`, leaving only those `n_trees` that deliver better scores
        for the data known to be anomalous.

        Parameters
        ----------
        n_trees
            Number of trees to keep for estimating anomaly scores.

        n_subsamples
            How many subsamples should be used to build every tree.

        max_depth
            Maximum depth of every tree.

        n_spare_trees
            Number of trees to generate additionally for further filtering.

        regenerate_trees
            Should we through out all the trees during retraining or should we
            mix old trees with the fresh ones. False by default, so we mix.

        weight_ratio
            What is the relative weight of false positives relative to true
            positives (i.e. we are not interested in negatives in anomaly
            detection, right?). The weight is used during the filtering
            process.

        random_seed
            Random seed. For reproducibility.
        """
        super().__init__(trees=[],
                         n_subsamples=n_subsamples,
                         max_depth=max_depth,
                         random_seed=random_seed)
        self.n_trees = n_trees
        self.n_spare_trees = n_spare_trees
        self.weight_ratio = weight_ratio
        self.regenerate_trees = regenerate_trees

        self.evaluator = None

    def _expand_trees(self, data, n_trees):
        """
        Expand the forest. Grow new trees.

        Parameters
        ----------
        data
            Data to build the trees from.

        n_trees
            What tree population are we aiming at?

        Returns
        -------
        None
        """
        n = n_trees - len(self.trees)
        if n > 0:
            self.trees.extend(self.build_trees(data, n))

    def _contract_trees(self, known_data, known_labels, n_trees):
        """
        Contract the forest with a chainsaw. According to the known data.

        Parameters
        ----------
        known_data
            Array with features of objects we know labels about.

        known_labels
            Array with labels. Of Label type.

        n_trees
            Aiming tree population after contraction.

        Returns
        -------
        None
        """
        n_filter = len(self.trees) - n_trees
        if n_filter > 0:
            self.trees = self.filter_trees(trees=self.trees,
                                           data=known_data,
                                           labels=known_labels,
                                           n_filter=n_filter,
                                           weight_ratio=self.weight_ratio)

    def fit(self, data, labels=None):
        """
        Build the trees with the data `data`.

        Parameters
        ----------
        data
            Array with feature values of objects.

        labels
            Optional. Labels of objects. May be regular, anomalous or unknown.
            See `Label` data for details.

        Returns
        -------
        self
        """
        # If no labels were supplied, train with them.
        if labels is None:
            self.fit_known(data)
            return

        # Otherwise select known data, and train on it.
        labels = np.asarray(labels)
        index = labels != Label.UNKNOWN
        return self.fit_known(data, data[index, :], labels[index])

    def fit_known(self, data, known_data=None, known_labels=None):
        """
        The same `fit` but with a bit of different API. Known data and labels
        are separated from training data for time and space optimality. High
        chances are that `known_data` is much smaller that `data`. At that case
        it is not reasonable to hold the labels for whole `data`.

        Parameters
        ----------
        data
            Training data (array with feature values) to build trees with.

        known_data
            Feature values of known data.

        known_labels
            Labels of known data.

        Returns
        -------
        self
        """
        if self.regenerate_trees:
            self.trees = []

        if known_data is None or len(known_data) == 0 or \
                known_labels is None or len(known_labels) == 0 or \
                np.all(known_labels == Label.UNKNOWN):
            self._expand_trees(data, self.n_trees)
        else:
            self._expand_trees(data, self.n_trees + self.n_spare_trees)
            self._contract_trees(known_data, known_labels, self.n_trees)

        self.evaluator = ConiferestEvaluator(self)
        return self

    @staticmethod
    def filter_trees(trees, data, labels, n_filter, weight_ratio=1):
        """
        Filter the trees out.

        Parameters
        ----------
        trees
            Trees to filter.

        n_filter
            Number of trees to filter out.

        data
            The labeled objects themselves.

        labels
            The labels of the objects. -1 is anomaly, 1 is not anomaly, 0 is uninformative.

        weight_ratio
            Weight of the false positive experience relative to false negative. Defaults to 1.
        """
        data = np.asarray(data, dtype=TreeDTYPE)

        n_samples, _ = data.shape
        n_trees = len(trees)

        heights = np.empty(shape=(n_samples, n_trees))
        for tree_index in range(n_trees):
            tree = trees[tree_index]
            leaves_index = tree.apply(data)
            n_samples_leaf = tree.n_node_samples[leaves_index]

            heights[:, tree_index] = \
                np.ravel(tree.decision_path(data).sum(axis=1)) + \
                average_path_length(n_samples_leaf) - 1

        weights = labels.copy()
        weights[labels == Label.REGULAR] = weight_ratio
        weighted_paths = (heights * np.reshape(weights, (-1, 1))).sum(axis=0)
        indices = weighted_paths.argsort()[n_filter:]

        return [trees[i] for i in indices]

    def score_samples(self, samples):
        """
        Computer scores for the supplied data.

        Parameters
        ----------
        samples
            Feature values to compute scores on.

        Returns
        -------
        Array with computed scores.
        """
        return self.evaluator.score_samples(samples)


class PineForestAnomalyDetector(AnomalyDetector):
    def __init__(self,
                 pine_forest,
                 lazy_training=True,
                 title='Pine Forest (filtered Isolation Forest)'):
        """
        Detector of anomalies with Pine Forest.

        Parameters
        ----------
        pine_forest
            Instance of PineForest to detect anomalies with.

        lazy_training
            Should we be lazy and don't retrain the forest after true positive
            results? True by default. So retrain only after receiving falses.

        title
            What title to use on plots.
        """
        super().__init__(title)
        self.pine_forest = pine_forest
        self.lazy_training = lazy_training
        self.train_data = None

    def train(self, data):
        """
        Build the forest.

        Parameters
        ----------
        data
            Features to build with.

        Returns
        -------
        None
        """
        self.train_data = data
        self.retrain()

    def retrain(self):
        """
        Retrain the forest according to available information about known data.

        Returns
        -------
        None
        """
        if self.train_data is None:
            raise ValueError('retrain called while no train data set')

        if self.known_data is None:
            self.pine_forest.fit(self.train_data)
        else:
            self.pine_forest.fit_known(self.train_data, self.known_data, self.known_labels)

    def score(self, data):
        """
        Calculate scores for given features.

        Parameters
        ----------
        data
            Given features.

        Returns
        -------
        Scores of the data.
        """
        return self.pine_forest.score_samples(data)

    def observe(self, point, label):
        """
        Learn about the next outlier.

        Parameters
        ----------
        point
            Features of the object.

        label
            True Label of the object.

        Returns
        -------
        bool, whether the regressor was changed.
        """
        super().observe(point, label)

        # Do retraining either on false positive result or if we are not lazy.
        do_retrain = label == Label.REGULAR or not self.lazy_training
        if do_retrain:
            self.retrain()

        return do_retrain