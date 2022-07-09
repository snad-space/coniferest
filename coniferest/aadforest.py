import numpy as np
from scipy.optimize import minimize

from .coniferest import Coniferest, ConiferestEvaluator
from .experiment import AnomalyDetector
from .datasets import Label
from .calc_paths_sum import calc_paths_sum, calc_paths_sum_transpose  # noqa


class AADEvaluator(ConiferestEvaluator):
    def __init__(self, aad):
        super(AADEvaluator, self).__init__(aad, map_value=np.reciprocal)
        self.weights = np.full(shape=(self.leaf_count,), fill_value=np.reciprocal(np.sqrt(self.leaf_count)))

    def calc_mean_values(self, x, weights=None):
        """
        Perform the computations.

        Parameters
        ----------
        x
            Features to calculate scores of. Should be C-contiguous for performance.
        weights
            Specific leaf weights to use instead of self.weights

        Returns
        -------
        Array of scores.
        """
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)

        if weights is None:
            weights = self.weights

        return calc_paths_sum(self.selectors, self.indices, x, weights)

    def loss(self, weights, known_data, known_labels, q_tau, C_a = 1.0, prior_influence = 1.0, prior_weights = None):
        scores = q_tau - self.score_samples(known_data, weights)

        if prior_weights is None:
            prior_weights = self.weights

        anomaly_count = np.count_nonzero(known_labels == Label.ANOMALY)
        nominal_count = np.count_nonzero(known_labels == Label.REGULAR)

        l = 0.0
        if anomaly_count:
            l += C_a * np.sum(scores[(known_labels == Label.ANOMALY) & (scores >= 0)]) / anomaly_count
        if nominal_count:
            l -= np.sum(scores[(known_labels == Label.REGULAR) & (scores <= 0)]) / nominal_count
        delta_weights = weights - prior_weights
        l += 0.5 * prior_influence * np.inner(delta_weights, delta_weights)

        return l

    def loss_gradient(self, weights, known_data, known_labels, q_tau, C_a = 1.0, prior_influence = 1.0, prior_weights = None):
        scores = q_tau - self.score_samples(known_data, weights)

        if prior_weights is None:
            prior_weights = self.weights

        anomaly_count = np.count_nonzero(known_labels == Label.ANOMALY)
        nominal_count = np.count_nonzero(known_labels == Label.REGULAR)

        sample_weights = np.zeros(known_data.shape[0])
        if anomaly_count:
            sample_weights[(known_labels == Label.ANOMALY) & (scores >= 0)] = -C_a / anomaly_count
        if nominal_count:
            sample_weights[(known_labels == Label.REGULAR) & (scores <= 0)] = 1.0 / nominal_count

        grad = calc_paths_sum_transpose(self.selectors, self.indices, known_data, self.leaf_count, sample_weights)
        delta_weights = weights - prior_weights
        grad += prior_influence * delta_weights

        return grad


class AADForest(Coniferest):
    def __init__(self,
                 n_trees=100,
                 n_subsamples=256,
                 max_depth=None,
                 tau=0.97,
                 C_a=1.0,
                 prior_influence=1.0,
                 random_seed=None):
        """

        Parameters
        ----------
        n_trees
            Number of trees to keep for estimating anomaly scores.

        n_subsamples
            How many subsamples should be used to build every tree.

        max_depth
            Maximum depth of every tree.

        random_seed
            Random seed. For reproducibility.
        """
        super().__init__(trees=[],
                         n_subsamples=n_subsamples,
                         max_depth=max_depth,
                         pdf=False,
                         random_seed=random_seed)
        self.n_trees = n_trees
        self.tau = tau
        self.C_a = C_a
        self.prior_influence = prior_influence

        self.evaluator = None

    def _build_trees(self, data):
        if len(self.trees) == 0:
            self.trees = self.build_trees(data, self.n_trees)
            self.evaluator = AADEvaluator(self)

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

        self._build_trees(data)

        if known_data is None or len(known_data) == 0 or \
                known_labels is None or len(known_labels) == 0 or \
                np.all(known_labels == Label.UNKNOWN):
            return self

        scores = self.score_samples(data)
        q_tau = np.quantile(scores, self.tau)

        def fun(weights):
            return self.evaluator.loss(weights, known_data, known_labels, q_tau, self.C_a, self.prior_influence)

        def jac(weights):
            return self.evaluator.loss_gradient(weights, known_data, known_labels, q_tau, self.C_a, self.prior_influence)

        def hessp(_weights, vector):
            return vector * self.prior_influence

        res = minimize(fun, self.evaluator.weights, method="trust-krylov", jac=jac, hessp=hessp, tol=1e-4)
        weights_norm = np.sqrt(np.inner(res.x, res.x))
        self.evaluator.weights = res.x / weights_norm

        return self

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
        return self.evaluator.calc_mean_values(samples)


class AADForestAnomalyDetector(AnomalyDetector):
    def __init__(self,
                 aad_forest,
                 title='AAD Forest'):
        """
        Detector of anomalies with AAD Forest.

        Parameters
        ----------
        aad_forest
            Instance of PineForest to detect anomalies with.

        title
            What title to use on plots.
        """
        super().__init__(title)
        self.aad_forest = aad_forest
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
            self.aad_forest.fit(self.train_data)
        else:
            self.aad_forest.fit_known(self.train_data, self.known_data, self.known_labels)

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
        return -self.aad_forest.score_samples(data)

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

        self.retrain()

        return True
