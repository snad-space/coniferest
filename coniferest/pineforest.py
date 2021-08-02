import numpy as np

from .coniferest import Coniferest, ConiferestEvaluator
from .experiment import AnomalyDetector
from .utils import average_path_length

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
        n = n_trees - len(self.trees)
        if n > 0:
            self.trees.extend(self.build_trees(data, n))

    def _contract_trees(self, known_data, known_labels, n_trees):
        n_filter = len(self.trees) - n_trees
        if n_filter > 0:
            self.trees = self.filter_trees(trees=self.trees,
                                           data=known_data,
                                           labels=known_labels,
                                           n_filter=n_filter,
                                           weight_ratio=self.weight_ratio)

    def fit(self, data, labels=None):
        if labels is None:
            self.fit_known(data)
            return

        labels = np.asarray(labels)
        index = labels != 0
        self.fit_known(data, data[index, :], labels[index])

    def fit_known(self, data, known_data=None, known_labels=None):
        if self.regenerate_trees:
            self.trees = []

        if known_data is None or len(known_data) == 0 or \
                known_labels is None or len(known_labels) == 0 or \
                np.all(known_labels) == 0:
            self._expand_trees(data, self.n_trees)
        else:
            self._expand_trees(data, self.n_trees + self.n_spare_trees)
            self._contract_trees(known_data, known_labels, self.n_trees)

        self.evaluator = ConiferestEvaluator(self)

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
        weights[labels == 1] = weight_ratio
        weighted_paths = (heights * np.reshape(weights, (-1, 1))).sum(axis=0)
        indices = weighted_paths.argsort()[n_filter:]

        return [trees[i] for i in indices]

    def score_samples(self, samples):
        return self.evaluator.score_samples(samples)


class PineForestAnomalyDetector(AnomalyDetector):
    def __init__(self,
                 pine_forest,
                 lazy_training=True,
                 title='Pine Forest (filtered Isolation Forest)'):
        super().__init__(title)
        self.pine_forest = pine_forest
        self.lazy_training = lazy_training
        self.train_data = None

    def train(self, data):
        self.train_data = data
        self.retrain()

    def retrain(self):
        if self.train_data is None:
            raise ValueError('retrain called while no train data set')

        if self.known_data is None:
            self.pine_forest.fit(self.train_data)
        else:
            self.pine_forest.fit_known(self.train_data, self.known_data, self.known_labels)

    def score(self, data):
        return self.pine_forest.score_samples(data)

    def observe(self, point, label):
        super().observe(point, label)
        self.retrain()
        return label == 1 and self.lazy_training