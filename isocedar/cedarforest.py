from .experiment import AnomalyDetector

class FilteredIsolationForestMixin:
    def filter_trees(self, n_filter, X, X_labels, weight=1):
        """
        Filter the trees out.

        Parameters
        ----------
        n_filter
            Number of trees to filter out.

        X
            The labeled objects themselves.

        X_labels
            The labels of the objects. -1 is anomaly, 1 is not anomaly, 0 is uninformative.

        weight
            Weight of the false positive experience. Defaults to 1.
        """
        n_samples = X.shape[0]
        n_trees = len(self.estimators_)

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
        # Can't set attribute:
        # self.estimators_samples_ = [self.estimators_samples_[i] for i in indices]


class FilteredIsolationForest(IsolationForest, FilteredIsolationForestMixin):
    pass


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