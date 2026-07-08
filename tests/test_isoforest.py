import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from sklearn.ensemble import IsolationForest as SkIsolationForest

from coniferest.datasets import MalanchevDataset
from coniferest.evaluator import ForestEvaluator
from coniferest.isoforest import IsolationForest


@pytest.fixture()
def isoforest_results():
    return IsoforestResults()


class IsoforestResults:
    def __init__(self):
        seed = 622341
        self.dataset = MalanchevDataset(inliers=1000, outliers=50, regions=[1, 1, -1], rng=seed)

        data = self.dataset.data
        trees = 1000

        forest = IsolationForest(n_trees=trees, random_seed=seed + 1)
        self.scores = self.calc_forest_scores(forest, data)
        self.forest = forest

        forest = SkIsolationForest(n_estimators=trees, random_state=seed + 2)
        self.skores0 = self.calc_forest_scores(forest, data)

        forest = SkIsolationForest(n_estimators=trees, random_state=seed + 3)
        self.skores1 = self.calc_forest_scores(forest, data)

    @staticmethod
    def calc_forest_scores(forest, data):
        forest.fit(data)
        return forest.score_samples(data)


def test_isolation_forest(isoforest_results):
    """
    Does assigned scores by our isoforest are somewhere near assigned by sklearn's isoforest?
    """
    r = isoforest_results
    diff_sk_to_sk = np.max(np.abs(r.skores0 - r.skores1))
    diff_coni_to_sk = np.max(np.abs(r.skores0 - r.scores))
    assert diff_coni_to_sk < 1.5 * diff_sk_to_sk


def test_rank_correlation_with_sklearn(isoforest_results):
    """
    Do our scores rank samples like sklearn's ones? Our decorrelation from
    sklearn must be comparable to sklearn's seed-to-seed decorrelation.
    """
    from scipy.stats import spearmanr

    r = isoforest_results
    rho_sk_to_sk = spearmanr(r.skores0, r.skores1).statistic
    rho_coni_to_sk = spearmanr(r.skores0, r.scores).statistic
    assert rho_coni_to_sk >= 1.0 - 1.5 * (1.0 - rho_sk_to_sk)


def test_top_anomalies_match_sklearn(isoforest_results):
    """
    Do we find the same top anomalies as sklearn? The dataset has 50 outliers,
    so we compare the top-50 sets, with sklearn's seed-to-seed agreement as
    the yardstick.
    """
    r = isoforest_results
    k = 50

    def top(scores):
        return set(np.argsort(scores)[:k])

    overlap_sk_to_sk = len(top(r.skores0) & top(r.skores1))
    overlap_coni_to_sk = len(top(r.skores0) & top(r.scores))
    assert overlap_coni_to_sk >= overlap_sk_to_sk - int(0.5 * (k - overlap_sk_to_sk)) - 1


def test_serialization(isoforest_results):
    """
    Does (de)serialization work correctly?
    """
    import pickle

    r = isoforest_results
    s = pickle.dumps(r.forest)
    reforest = pickle.loads(s)
    assert_allclose(reforest.score_samples(r.dataset.data), r.scores, atol=1e-12)


def assert_forest_scores(forest1: IsolationForest, forest2: IsolationForest, data=None, n_features=None):
    if data is None:
        if n_features is None:
            raise ValueError("Either data or n_features")
        data = np.random.standard_normal((1024, n_features))
    assert_equal(forest1.score_samples(data), forest2.score_samples(data))


def build_forest(n_features: int, random_seed: int) -> IsolationForest:
    n_trees = 100
    n_subsamples = 256

    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_trees * n_subsamples, n_features))

    forest = IsolationForest(
        n_trees=n_trees,
        n_subsamples=n_subsamples,
        max_depth=None,
        random_seed=random_seed,
    )
    forest.fit(data)
    return forest


def test_reproducibility():
    n_features = 16
    random_seed = np.random.randint(1 << 16)
    forest1 = build_forest(n_features, random_seed)
    forest2 = build_forest(n_features, random_seed)
    assert_forest_scores(forest1, forest2, n_features=n_features)


def test_apply_dense():
    n_features = 16
    n_trees = 100
    n_subsamples = 256

    random_seed = np.random.randint(1 << 16)
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_trees * n_subsamples, n_features))

    forest = IsolationForest(
        n_trees=n_trees,
        n_subsamples=n_subsamples,
        max_depth=None,
        random_seed=random_seed,
    )

    forest.fit(data)

    leafs = forest.apply(data)
    scores = np.sum(ForestEvaluator.combine_leaf_values(forest.trees)[leafs], axis=1)
    scores = -(2 ** (-scores / (forest.evaluator.average_path_length(n_subsamples) * n_trees)))
    assert_allclose(forest.score_samples(data), scores)


def test_apply_sparse():
    n_features = 16
    n_trees = 100
    n_subsamples = 256

    random_seed = np.random.randint(1 << 16)
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_trees * n_subsamples, n_features))

    forest = IsolationForest(
        n_trees=n_trees,
        n_subsamples=n_subsamples,
        max_depth=None,
        random_seed=random_seed,
    )

    forest.fit(data)

    leafs = forest.apply(data, "sparse")
    scores = leafs @ ForestEvaluator.combine_leaf_values(forest.trees)
    scores = -(2 ** (-scores / (forest.evaluator.average_path_length(n_subsamples) * n_trees)))
    assert_allclose(forest.score_samples(data), scores)


@pytest.mark.regression
def test_regression(regression_data):
    random_seed = 0
    n_features = 16
    n_samples = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = build_forest(n_features=n_features, random_seed=random_seed)
    scores = forest.score_samples(data)
    regression_data.assert_allclose(scores)


@pytest.mark.regression
def test_regression_signatures(regression_data):
    random_seed = 0
    n_features = 16
    n_samples = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = build_forest(n_features=n_features, random_seed=random_seed)
    signatures = forest.feature_signature(data)
    regression_data.assert_allclose(signatures)


@pytest.mark.regression
def test_regression_importance(regression_data):
    random_seed = 0
    n_features = 16
    n_samples = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = build_forest(n_features=n_features, random_seed=random_seed)
    importance = forest.feature_importance(data)
    regression_data.assert_allclose(importance)


def test_n_jobs():
    random_seed = 0
    n_features = 16
    n_samples = 1024
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))

    reference_forest = IsolationForest(n_trees=5, random_seed=random_seed)
    reference_forest.fit(data)

    for n_jobs in [1, 2, -1]:
        forest = IsolationForest(n_trees=5, n_jobs=n_jobs, random_seed=random_seed)
        forest.fit(data)
        assert_forest_scores(reference_forest, forest, data=data)


@pytest.mark.benchmark
@pytest.mark.long
def test_benchmark_score_float32(n_jobs, benchmark):
    benchmark.group = f"IsolationForest.score_samples float32 {n_jobs = :2d}"
    benchmark.name = "coniferest.isoforest.IsolationForest"

    random_seed = 0
    n_samples = 1 << 20
    n_features = 16
    n_trees = 100
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    forest = IsolationForest(n_trees=n_trees, n_jobs=n_jobs, random_seed=random_seed)
    forest.fit(data)

    benchmark(forest.score_samples, data)


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_trees", [128, 1024])
def test_benchmark_fit(n_trees, n_jobs, benchmark):
    benchmark.group = f"IsolationForest.fit {n_trees = :4d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.isoforest.IsolationForest"

    random_seed = 0
    n_samples = 16_384
    n_features = 16
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = IsolationForest(n_trees=n_trees, n_jobs=n_jobs, random_seed=random_seed)

    benchmark(forest.fit, data)


# We need to merge it with previous one when we make interface consistent with sklearn's
# https://github.com/snad-space/coniferest/issues/113
@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_trees", [128, 1024])
def test_benchmark_fit_sklearn(n_trees, n_jobs, benchmark):
    benchmark.group = f"IsolationForest.fit {n_trees = :4d}, {n_jobs = :2d}"
    benchmark.name = "sklearn.ensemble.IsolationForest"

    random_seed = 0
    n_samples = 16_384
    n_features = 16
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = SkIsolationForest(n_estimators=n_trees, n_jobs=n_jobs, random_state=random_seed)

    benchmark(forest.fit, data)


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_samples", [1 << 10, 1 << 20])
def test_benchmark_score(n_samples, n_jobs, benchmark):
    benchmark.group = f"IsolationForest.score_samples {n_samples = :7d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.isoforest.IsolationForest"

    random_seed = 0
    n_features = 16
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = IsolationForest(n_trees=128, n_jobs=n_jobs, random_seed=random_seed)
    forest.fit(data)

    benchmark(forest.score_samples, data)


# We need to merge it with previous one when we make interface consistent with sklearn's
# https://github.com/snad-space/coniferest/issues/113
@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_samples", [1 << 10, 1 << 20])
def test_benchmark_score_sklearn(n_samples, n_jobs, benchmark):
    benchmark.group = f"IsolationForest.score_samples {n_samples = :7d}, {n_jobs = :2d}"
    benchmark.name = "sklearn.ensemble.IsolationForest"

    random_seed = 0
    n_features = 16
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = SkIsolationForest(n_estimators=128, n_jobs=n_jobs, random_state=random_seed)
    forest.fit(data)

    benchmark(forest.score_samples, data)


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_features", [2, 128])
def test_benchmark_feature_signature(n_features, n_jobs, benchmark):
    benchmark.group = f"IsolationForest.feature_signature {n_features = :3d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.isoforest.IsolationForest"

    random_seed = 0
    n_samples = 1 << 14
    n_trees = 1024
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    forest = IsolationForest(n_trees=n_trees, n_jobs=n_jobs, random_seed=random_seed)
    forest.fit(data)

    benchmark(forest.feature_signature, data[:1])


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_samples", [1, 1 << 5, 1 << 10, 1 << 20])
@pytest.mark.parametrize("n_trees", [1 << 6, 1 << 7, 1 << 8])
@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_benchmark_score_samples(n_samples, n_trees, n_jobs, benchmark):
    benchmark.group = f"IsolationForest.score_samples {n_samples = :7d}, {n_trees = :4d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.isoforest.IsolationForest"

    random_seed = 0
    n_features = 374
    n_samples_build = 1 << 20
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples_build, n_features))
    forest = IsolationForest(n_trees=n_trees, n_jobs=n_jobs, random_seed=random_seed)
    forest.fit(data)

    test_data = data[:n_samples]
    benchmark(forest.score_samples, test_data)


def test_float32_data():
    """
    Trees are built on the data dtype; scoring data of a different dtype
    is cast (copied) to it.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2048, 4), dtype=np.float32)

    forest = IsolationForest(n_trees=32, random_seed=0)
    forest.fit(data)
    assert all(tree.dtype == "float32" for tree in forest.trees)
    assert forest.evaluator.dtype == np.float32

    scores = forest.score_samples(data)
    assert_equal(forest.score_samples(data.astype(np.float64)), scores)
