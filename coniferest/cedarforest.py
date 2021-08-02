from .coniferest import Coniferest, ConiferestEvaluator
from .experiment import AnomalyDetector


class PineForest(Coniferest):
    def __init__(self,
                 n_trees=100,
                 n_subsamples=256,
                 max_depth=None,
                 n_spare_trees=400,
                 weight_ratio=1,
                 random_seed=None):
        super().__init__(trees=[],
                         n_subsamples=n_subsamples,
                         max_depth=max_depth,
                         random_seed=random_seed)
        self.n_trees = n_trees
        self.n_spare_trees = n_spare_trees
        self.weight_ratio = weight_ratio

        self.evaluator = None

    def fit(self, data, labels=None):
        n_spare_trees = 0 if labels is None else self.n_spare_trees
        n = self.n_trees + n_spare_trees - len(self.trees)

        if n > 0:
            self.trees.extend(self.build_trees(data, n))

        self.evaluator = ConiferestEvaluator(self)

        n_filter = len(self.trees) - self.n_trees
        if n_filter > 0:
            self.trees = self.filter_trees(trees=self.trees,
                                           data=data,
                                           labels=labels,
                                           n_filter=n_filter,
                                           weight=self.weight_ratio)
            self.evaluator = ConiferestEvaluator(self)

        return self

    @staticmethod
    def filter_trees(trees, data, labels, n_filter, weight=1):
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

        weight
            Weight of the false positive experience relative to false negative. Defaults to 1.
        """
        n_samples = data.shape[0]
        n_trees = len(trees)

        heights = np.empty(shape=(n_samples, n_trees))
        for tree_index in range(len(self.estimators_)):
            estimator = self.estimators_[tree_index]
            leaves_index = estimator.apply(X)
            n_samples_leaf = estimator.tree_.n_node_samples[leaves_index]

            heights[:, tree_index] = \
                np.ravel(estimator.decision_path(X).sum(axis=1)) + \
                _average_path_length(n_samples_leaf) - 1

        weights = X_labels.copy()
        weights[X_labels == 1] = weight
        scores = (heights * np.reshape(weights, (-1, 1))).sum(axis=0)
        indices = scores.argsort()[n_filter:]

        self.estimators_ = [self.estimators_[i] for i in indices]
        self.estimators_features_ = [self.estimators_features_[i] for i in indices]

        return 1

    def score_samples(self, samples):
        return self.evaluator.score_samples(samples)



class FilteredIsoforestLazyAnomalyDetector(AnomalyDetector):
    def __init__(self, n_estimators=100, n_spares=400, weight=1, title='Filtered Isolation Forest', **kwargs):
        super().__init__(title)
        self.n_estimators = n_estimators
        self.n_spares = n_spares
        self.args = kwargs
        self.isoforest = None
        self.train_data = None
        self.weight = weight

    def train(self, data):
        self.train_data = data
        self.retrain()

    def retrain(self):
        if self.train_data is None:
            raise ValueError('retrain called while no train data set')

        if self.known_data is None:
            self.args['n_estimators'] = self.n_estimators
            self.isoforest = FilteredIsolationForest(**self.args)
            self.isoforest.fit(self.train_data)
        else:
            self.args['n_estimators'] = self.n_estimators + self.n_spares
            self.isoforest = FilteredIsolationForest(**self.args)
            self.isoforest.fit(self.train_data)
            self.isoforest.filter_trees(self.n_spares, self.known_data, self.known_labels, weight=self.weight)

    def score(self, data):
        return self.isoforest.score_samples(data)

    def observe(self, point, label):
        super().observe(point, label)
        self.retrain()
        return label == 1