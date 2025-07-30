from numbers import Real
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .calc_trees import calc_paths_sum, calc_paths_sum_transpose  # noqa
from .coniferest import Coniferest, ConiferestEvaluator
from .label import Label

__all__ = ["AADForest"]


class AADEvaluator(ConiferestEvaluator):
    def __init__(self, aad):
        super(AADEvaluator, self).__init__(aad, map_value=aad.map_value)
        self.C_a = aad.C_a
        self.budget = aad.budget
        self.prior_influence = aad.prior_influence
        self.weights = np.full(shape=(self.n_leaves,), fill_value=np.reciprocal(np.sqrt(self.n_leaves)))

    def _q_tau(self, scores):
        if isinstance(self.budget, int):
            if self.budget >= len(scores):
                return np.max(scores)

            return np.partition(scores, self.budget)[self.budget]
        elif isinstance(self.budget, float):
            return np.quantile(scores, self.budget)

        raise ValueError("self.budget must be an int or float")

    def score_samples(self, x, weights=None):
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
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        if weights is None:
            weights = self.weights

        return calc_paths_sum(
            self.selectors,
            self.node_offsets,
            x,
            weights,
            num_threads=self.num_threads,
            batch_size=self.get_batch_size(self.n_trees),
        )

    def loss(
        self,
        weights,
        known_data,
        known_labels,
        anomaly_count,
        nominal_count,
        q_tau,
        C_a=1.0,
        prior_influence=1.0,
        prior_weights=None,
    ):
        """Loss for the known data.

        Adopted from Eq3 of Das et al. 2019 https://arxiv.org/abs/1901.08930
        with the respect to
        1) different anomaly labels - they use +1 for anomalies, -1 for
              nomalies, we use `Label` enum, which is opposite and includes
              `UNKNOWN` label;
        2) different direction of the score axis - they use higher scores
              for anomalies, we use lower scores for anomalies.
        """

        # This new score is negative for the "anomalous" subsample and
        # positive for the "nominal" subsample.
        scores = self.score_samples(known_data, weights) - q_tau

        if prior_weights is None:
            prior_weights = self.weights

        loss = 0.0
        if anomaly_count:
            # For anomalies in "nominal" subsample we add their positive scores.
            loss += C_a * np.sum(scores[(known_labels == Label.ANOMALY) & (scores >= 0)]) / anomaly_count
        if nominal_count:
            # Add for nominals in "anomalous" subsample we add their inverse scores (positive number).
            loss -= np.sum(scores[(known_labels == Label.REGULAR) & (scores <= 0)]) / nominal_count
        delta_weights = weights - prior_weights
        loss += 0.5 * prior_influence * np.inner(delta_weights, delta_weights)

        return loss

    def loss_gradient(
        self,
        weights,
        known_data,
        known_labels,
        anomaly_count,
        nominal_count,
        q_tau,
        C_a=1.0,
        prior_influence=1.0,
        prior_weights=None,
    ):
        scores = self.score_samples(known_data, weights) - q_tau

        if prior_weights is None:
            prior_weights = self.weights

        sample_weights = np.zeros(known_data.shape[0])
        if anomaly_count:
            sample_weights[(known_labels == Label.ANOMALY) & (scores >= 0)] = C_a / anomaly_count
        if nominal_count:
            sample_weights[(known_labels == Label.REGULAR) & (scores <= 0)] = -1.0 / nominal_count

        grad = calc_paths_sum_transpose(
            self.selectors,
            self.node_offsets,
            self.leaf_offsets,
            known_data,
            sample_weights,
            num_threads=self.num_threads,
            batch_size=self.get_batch_size(len(known_data)),
        )
        delta_weights = weights - prior_weights
        grad += prior_influence * delta_weights

        return grad

    def loss_hessian(
        self,
        weights,
        vector,
        known_data,
        known_labels,
        q_tau,
        anomaly_count,
        nominal_count,
        C_a=1.0,
        prior_influence=1.0,
        prior_weights=None,
    ):
        return vector * prior_influence

    def fit_known(self, data, known_data, known_labels):
        scores = self.score_samples(data)
        q_tau = self._q_tau(scores)

        anomaly_count = np.count_nonzero(known_labels == Label.ANOMALY)
        nominal_count = np.count_nonzero(known_labels == Label.REGULAR)
        prior_influence = self.prior_influence(anomaly_count, nominal_count)

        res = minimize(
            self.loss,
            self.weights,
            args=(known_data, known_labels, anomaly_count, nominal_count, q_tau, self.C_a, prior_influence),
            method="trust-krylov",
            jac=self.loss_gradient,
            hessp=self.loss_hessian,
            tol=1e-4,
        )
        weights_norm = np.sqrt(np.inner(res.x, res.x))
        self.weights = res.x / weights_norm


class AADForest(Coniferest):
    """
    Active Anomaly Detection with Isolation Forest.

    See Das et al., 2017 https://arxiv.org/abs/1708.09441

    Parameters
    ----------
    n_trees : int, optional
        Number of trees in the isolation forest.

    n_subsamples : int, optional
        How many subsamples should be used to build every tree.

    max_depth : int or None, optional
        Maximum depth of every tree. If None, `log2(n_subsamples)` is used.

    budget : int or float, optional
        Budget of anomalies. If the type is floating point it is considered as
        fraction of full data. If the type is integer it is considered as the
        number of items. Default is 0.03.

    n_jobs : int or None, optional
        Number of threads to use for scoring. If None - all available CPUs are used.

    random_seed : int or None, optional
        Random seed to use for reproducibility. If None - random seed is used.

    prior_influence : float or callable, optional
        An regularization coefficient value in the loss functioin. Default is 1.0.
        Signature: '(anomaly_count, nominal_count) -> float'

    map_value : ["const", "exponential", "linear", "reciprocal"] or callable, optional
        An function applied to the leaf depth before weighting. Possible
        meaning variants are: 1, 1-exp(-x), x, -1/x.
    """

    def __init__(
        self,
        n_trees=100,
        n_subsamples=256,
        max_depth=None,
        budget=0.03,
        C_a=1.0,
        prior_influence=1.0,
        n_jobs=None,
        random_seed=None,
        sampletrees_per_batch=1 << 20,
        map_value=None,
    ):
        super().__init__(
            trees=[],
            n_subsamples=n_subsamples,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_seed=random_seed,
            sampletrees_per_batch=sampletrees_per_batch,
        )
        self.n_trees = n_trees

        if not isinstance(budget, (int, float)):
            raise ValueError("budget must be an int or float")

        self.budget = budget
        self.C_a = C_a

        if isinstance(prior_influence, Callable):
            self.prior_influence = prior_influence
        elif isinstance(prior_influence, Real):
            self.prior_influence = lambda anomaly_count, nominal_count: prior_influence
        else:
            raise ValueError("prior_influence is neither a callable nor a constant")

        MAP_VALUES = {
            "const": np.ones_like,
            "exponential": lambda x: -np.expm1(-x),
            "linear": lambda x: x,
            "reciprocal": lambda x: -np.reciprocal(x),
        }

        if map_value is None:
            map_value = "reciprocal"

        if isinstance(map_value, Callable):
            self.map_value = map_value
        elif map_value in MAP_VALUES:
            self.map_value = MAP_VALUES[map_value]
        else:
            raise ValueError(f"map_value is neither a callable nor one of {', '.join(MAP_VALUES.keys())}.")

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
            return self.fit_known(data)

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

        known_data, known_labels = self._validate_known_data(known_data, known_labels)

        self._build_trees(data)

        if (
            known_data is None
            or len(known_data) == 0
            or known_labels is None
            or len(known_labels) == 0
            or np.all(known_labels == Label.UNKNOWN)
        ):
            return self

        self.evaluator.fit_known(data, known_data, known_labels)

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
        return self.evaluator.score_samples(samples)

    def feature_signature(self, x):
        raise NotImplementedError()

    def feature_importance(self, x):
        raise NotImplementedError()

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
