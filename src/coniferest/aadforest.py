from numbers import Real
from typing import Callable

import clarabel
import numpy as np
from scipy import sparse

from .calc_trees import calc_paths_sum  # noqa
from .coniferest import Coniferest, ConiferestEvaluator
from .label import Label

__all__ = ["AADForest"]


class AADEvaluator(ConiferestEvaluator):
    def __init__(self, aad):
        super(AADEvaluator, self).__init__(aad, map_value=aad.map_value)
        self.C_a = aad.C_a
        self.budget = aad.budget
        self.n_jobs = aad.n_jobs
        self.prior_influence = aad.prior_influence
        self.weights = np.full(shape=(self.n_leaves,), fill_value=np.reciprocal(np.sqrt(self.n_leaves)))

        leaf_mask = self.selectors["feature"] < 0
        self.leaf_values = self.selectors["value"][leaf_mask]

    def _q_tau(self, scores):
        if self.budget == "auto":
            # When the regularization is disabled then the problem is degenerate
            if self.prior_influence == 0:
                return 1.0

            return None
        elif isinstance(self.budget, int):
            if self.budget >= len(scores):
                return np.max(scores)

            return np.partition(scores, self.budget)[self.budget]
        elif isinstance(self.budget, float):
            return np.quantile(scores, self.budget)

        raise ValueError('self.budget must be an int or float or "auto"')

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
            batch_size=self.batch_size,
        )

    def fit_known(self, data, known_data, known_labels, prior_weights=None):
        scores = self.score_samples(data)
        q_tau = self._q_tau(scores)
        known_leafs = self.apply(known_data)

        n_anomalies = np.count_nonzero(known_labels == Label.ANOMALY)
        n_nominals = np.count_nonzero(known_labels == Label.REGULAR)
        prior_influence = self.prior_influence(n_anomalies, n_nominals)

        if prior_weights is None:
            prior_weights = self.weights

        n_weights = self.weights.shape[0]
        n_knowns = n_anomalies + n_nominals
        offset_knowns = n_weights
        n_q_tau = int(q_tau is None)
        offset_q_tau = n_weights + n_knowns
        n_variables = n_weights + n_knowns + n_q_tau
        n_constraints = 2 * n_knowns

        # Problem matrix P
        data = np.full_like(self.weights, prior_influence)
        ind = np.arange(n_weights)
        P = sparse.csc_matrix((data, (ind, ind)), shape=(n_variables, n_variables))

        # Problem vector q
        q_known = np.zeros_like(known_labels, dtype=self.weights.dtype)
        if n_anomalies > 0:
            q_known[known_labels == Label.ANOMALY] = self.C_a * np.reciprocal(float(n_anomalies))
        if n_nominals > 0:
            q_known[known_labels == Label.REGULAR] = np.reciprocal(float(n_nominals))

        q = np.concatenate(
            [
                -prior_influence * prior_weights,
                q_known,
                np.zeros(shape=(n_q_tau,), dtype=self.weights.dtype),
            ]
        )

        # Constraints matrix A:
        ## - Weight variables
        ## - Cost variables
        ## - q_tau variable
        col_ind = np.concatenate(
            [
                known_leafs.flatten(),
                np.tile(np.arange(offset_knowns, offset_q_tau), 2),
                np.full((n_knowns,), offset_q_tau) if n_q_tau else [],
            ]
        )
        row_ind = np.concatenate(
            [
                np.repeat(np.arange(n_knowns), self.n_trees),
                np.arange(n_constraints),
                np.arange(n_knowns) if n_q_tau else [],
            ]
        )
        data = np.concatenate(
            [
                (-self.leaf_values[known_leafs] * known_labels.reshape(-1, 1)).flatten(),
                np.full((n_constraints,), -1),
                known_labels if n_q_tau else [],
            ]
        )

        A = sparse.csc_matrix((data, (row_ind, col_ind)), shape=(n_constraints, n_variables))

        # Constraints vector b
        b = np.concatenate(
            [
                np.zeros((n_knowns,)) if n_q_tau else -q_tau * known_labels,
                np.zeros((n_knowns,)),
            ]
        )

        # Constraints type: all \ge 0
        cones = [
            clarabel.NonnegativeConeT(n_constraints),
        ]

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        # max_threads = 0 means automatic
        settings.max_threads = self.n_jobs if self.n_jobs > 0 else 0

        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)

        res = solver.solve()

        self.weights = np.asarray(res.x[:n_weights])
        weights_norm = np.linalg.norm(self.weights)
        self.weights /= weights_norm


class AADForest(Coniferest):
    r"""
    Active Anomaly Detection with Isolation Forest.

    See Das et al., 2017 https://arxiv.org/abs/1708.09441

    The method solves the optimization problem:

    .. math::

       \mathbf{w} = \arg\min_{\mathbf{w}} \left(
        \frac{C_a}{\left|\mathcal{A}\right|}
            \sum_{i \in \mathcal{A}} \mathrm{ReLU}\left(s(\mathbf{x_i} | \mathbf{w}) - q_{\tau}\right) +
        \frac{1}{\left|\mathcal{N}\right|}
            \sum_{i \in \mathcal{N}} \mathrm{ReLU}\left(q_{\tau} - s(\mathbf{x_i} | \mathbf{w})\right) +
        \frac{\alpha}{2} \lVert \mathbf{w} - \mathbf{w_0}\rVert^2\right),

    where :math:`C_a` is `C_a`, regularization parameter :math:`\alpha` is
    `prior_influence`, :math:`\mathcal{A}` is a set of known anomalies,
    :math:`\mathcal{N}` is a set of known nominals, :math:`s(\mathbf{x_i} |
    \mathbf{w})` is the anomaly score of instance with features
    :math:`\mathbf{x_i}` given weights :math:`\mathbf{w}`.

    This problem is reformulated as an equivalent quadratic programming problem:

    .. math::

       \begin{bmatrix}
       \mathbf{w}\\
       \mathbf{u}
       \end{bmatrix} = \arg\min_{\mathbf{w}, \mathbf{u}} \left(
        \frac{C_a}{\left|\mathcal{A}\right|} \sum_{i \in \mathcal{A}} u_i +
        \frac{1}{\left|\mathcal{N}\right|} \sum_{i \in \mathcal{N}} u_i +
        \frac{\alpha}{2} \lVert \mathbf{w} - \mathbf{w_0} \rVert^2\right),

    with the following convex constraints:

    .. math::

       u_i &\ge 0 \quad & i \in \mathcal{A} \cup \mathcal{N},\\
       u_i - s(\mathbf{x_i} | \mathbf{w}) &\ge - q_{\tau}\quad & i \in \mathcal{A},\\
       u_i + s(\mathbf{x_i} | \mathbf{w}) &\ge q_{\tau}\quad & i \in \mathcal{N}.\\

    Parameters
    ----------
    n_trees : int, optional
        Number of trees in the isolation forest.

    n_subsamples : int, optional
        How many subsamples should be used to build every tree.

    max_depth : int or None, optional
        Maximum depth of every tree. If None, `log2(n_subsamples)` is used.

    budget : int or float or "auto", optional
        Budget of anomalies. If the type is floating point it is considered as
        fraction of full data. If the type is integer it is considered as the
        number of items. If string "auto" is set then the exact parameter is
        found during the train. Default is "auto".

    n_jobs : int, default=-1
        Number of threads to use for scoring. If -1, use all available CPUs.

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
        budget="auto",
        C_a=1.0,
        prior_influence=1.0,
        n_jobs=-1,
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

        if not (isinstance(budget, (int, float)) or budget == "auto"):
            raise ValueError('budget must be an int or float or "auto"')

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

        index = known_labels != Label.UNKNOWN
        known_data, known_labels = known_data[index], known_labels[index]

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
