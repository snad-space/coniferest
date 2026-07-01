import numpy as np
import pytest

from coniferest.aadforest import AADForest
from coniferest.datasets import single_outlier
from coniferest.label import Label


def infinite_coefficient_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((64, 4))
    known_data = data[[0, 1, 2]]
    known_labels = np.array([Label.ANOMALY, Label.REGULAR, Label.REGULAR])

    return data, known_data, known_labels


def test_scores_negative():
    data = np.arange(1024.0).reshape(256, 4)
    forest = AADForest(n_trees=10, random_seed=0).fit(data)
    scores = forest.score_samples(data)
    assert np.all(scores < 0)


def test_single_outlier():
    data, _metadata = single_outlier()
    forest = AADForest(n_trees=10, random_seed=0).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


def test_prior_influence_value():
    data, _metadata = single_outlier()
    forest = AADForest(n_trees=10, random_seed=0, prior_influence=2.0).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


def test_prior_influence_callable():
    data, _metadata = single_outlier()
    forest = AADForest(
        n_trees=10,
        random_seed=0,
        prior_influence=lambda ac, nc: np.max(1.0, 0.5 / (ac + nc)),
    ).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


def test_infinite_c_a_produces_finite_scores():
    data, known_data, known_labels = infinite_coefficient_data()

    forest = AADForest(n_trees=10, n_subsamples=32, random_seed=0, C_a=np.inf, budget=0.1, n_jobs=1)
    forest.fit_known(data, known_data=known_data, known_labels=known_labels)

    assert np.all(np.isfinite(forest.evaluator.weights))
    assert np.all(np.isfinite(forest.score_samples(data)))


@pytest.mark.parametrize(
    ("infinite_params", "normalized_params"),
    [
        ({"C_a": np.inf}, {"C_a": 1.0, "C_n": 0.0, "prior_influence": 0.0}),
        ({"C_n": np.inf}, {"C_a": 0.0, "C_n": 1.0, "prior_influence": 0.0}),
        ({"prior_influence": np.inf}, {"C_a": 0.0, "C_n": 0.0, "prior_influence": 1.0}),
        (
            {"prior_influence": lambda anomaly_count, nominal_count: np.inf},
            {"C_a": 0.0, "C_n": 0.0, "prior_influence": 1.0},
        ),
    ],
)
def test_infinite_coefficient_matches_normalized_loss(infinite_params, normalized_params):
    data, known_data, known_labels = infinite_coefficient_data()
    forest_params = dict(
        n_trees=10,
        n_subsamples=32,
        random_seed=0,
        budget=0.1,
        n_jobs=1,
    )

    forest = AADForest(**(forest_params | infinite_params)).fit_known(
        data,
        known_data=known_data,
        known_labels=known_labels,
    )
    normalized_forest = AADForest(**(forest_params | normalized_params)).fit_known(
        data,
        known_data=known_data,
        known_labels=known_labels,
    )

    np.testing.assert_allclose(
        forest.score_samples(data),
        normalized_forest.score_samples(data),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "forest_params",
    [
        {"C_a": np.inf, "C_n": np.inf},
        {"C_a": np.inf, "prior_influence": np.inf},
        {"C_n": np.inf, "prior_influence": np.inf},
    ],
)
def test_multiple_infinite_coefficients_raise(forest_params):
    with pytest.raises(ValueError, match="At most one"):
        AADForest(**forest_params)


def test_invalid_callable_prior_influence_raises():
    data, known_data, known_labels = infinite_coefficient_data()
    forest = AADForest(
        n_trees=10,
        n_subsamples=32,
        random_seed=0,
        prior_influence=lambda anomaly_count, nominal_count: np.nan,
    )

    with pytest.raises(ValueError, match="prior_influence"):
        forest.fit_known(data, known_data=known_data, known_labels=known_labels)

@pytest.mark.regression
def test_budget_auto(regression_data):
    data, _metadata = single_outlier()
    forest = AADForest(budget="auto", random_seed=0).fit(data)
    scores = forest.score_samples(data)
    regression_data.assert_allclose(scores)


# Single-thread and parallel implementations are a bit different, so here we check both.
# We use n_thread parameter instead of n_jobs, which is a fixture in conftest.py
@pytest.mark.parametrize("n_thread", [1, 2])
@pytest.mark.regression
def test_regression_fit_known(n_thread, regression_data):
    random_seed = 0
    n_samples = 1024
    n_features = 16
    n_known = 16
    n_trees = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    known_data = data[rng.choice(n_samples, n_known, replace=False)]
    known_labels = rng.choice([-1, 1], n_known, replace=True)

    # This small sampletrees_per_batch is inefficient, but it's good for testing to guarantee parallel execution.
    forest = AADForest(
        n_trees=n_trees, random_seed=random_seed, n_jobs=n_thread, sampletrees_per_batch=2048, budget=0.03
    )
    forest.fit(data)
    pre_fit_known_scores = forest.score_samples(data)

    forest.fit_known(data, known_data=known_data, known_labels=known_labels)
    scores = forest.score_samples(data)

    assert not np.allclose(pre_fit_known_scores, scores, rtol=1e-3), "Scores must change after fit_known"

    regression_data.assert_allclose(scores)


@pytest.mark.benchmark
@pytest.mark.long
def test_benchmark_fit_known(n_jobs, benchmark):
    benchmark.group = f"AADForest.fit_known {n_jobs = :2d}"
    benchmark.name = "coniferest.aadforest.AADForest"

    random_seed = 0
    n_samples = 1024
    n_features = 16
    n_known = 16
    n_trees = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    known_data = data[rng.choice(n_samples, n_known, replace=False)]
    known_labels = rng.choice([-1, 1], n_known, replace=True)

    forest = AADForest(n_trees=n_trees, n_jobs=n_jobs, random_seed=random_seed)
    forest.fit(data)

    benchmark(forest.fit_known, data, known_data=known_data, known_labels=known_labels)
