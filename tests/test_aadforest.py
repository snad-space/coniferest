import numpy as np
import pytest

from coniferest.aadforest import AADForest
from coniferest.datasets import single_outlier
from coniferest.label import Label


def test_scores_negative():
    data = np.arange(1024.0).reshape(256, 4)
    forest = AADForest(n_trees=10, random_seed=0).fit(data)
    scores = forest.score_samples(data)
    assert np.all(scores < 0)


def test_single_outlier():
    data, _metadata = single_outlier()
    forest = AADForest(n_trees=10, random_seed=42).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


def test_prior_influence_value():
    data, _metadata = single_outlier()
    forest = AADForest(n_trees=10, random_seed=42, prior_influence=2.0).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


def test_prior_influence_callable():
    data, _metadata = single_outlier()
    forest = AADForest(
        n_trees=10,
        random_seed=42,
        prior_influence=lambda ac, nc: np.max(1.0, 0.5 / (ac + nc)),
    ).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


@pytest.mark.regression
def test_budget_auto(regression_data):
    data, _metadata = single_outlier()
    known_data = data[[0, -1]]
    known_labels = np.asarray([Label.REGULAR, Label.ANOMALY])
    forest = AADForest(budget="auto", random_seed=0).fit_known(data, known_data=known_data, known_labels=known_labels)
    scores = forest.score_samples(data)
    regression_data.assert_allclose(scores)


@pytest.mark.regression
def test_budget_auto_prior_influence_one(regression_data):
    data, _metadata = single_outlier()
    known_data = data[[0, -1]]
    known_labels = np.asarray([Label.REGULAR, Label.ANOMALY])
    forest = AADForest(budget="auto", prior_influence=1.0, random_seed=0).fit_known(
        data, known_data=known_data, known_labels=known_labels
    )
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


@pytest.mark.parametrize("initial_weights", ["uniform", "zero", "random"])
def test_initial_weights(initial_weights):
    data = np.arange(128.0).reshape(32, 4)
    params = dict(n_trees=5, n_subsamples=16, random_seed=42, n_jobs=1)
    default = AADForest(**params).fit(data)
    forest = AADForest(**params, initial_weights=initial_weights).fit(data)
    np.testing.assert_array_equal(forest.apply(data), default.apply(data))
    if initial_weights == "uniform":
        np.testing.assert_array_equal(forest.evaluator.weights, default.evaluator.weights)
        np.testing.assert_array_equal(forest.score_samples(data), default.score_samples(data))
        assert forest.rng.bit_generator.state == default.rng.bit_generator.state
    elif initial_weights == "zero":
        np.testing.assert_array_equal(forest.evaluator.weights, 0)
        np.testing.assert_array_equal(forest.score_samples(data), 0)
        assert forest.rng.bit_generator.state == default.rng.bit_generator.state
    else:
        expected = default.rng.uniform(-1.0, 1.0, size=default.evaluator.n_leaves)
        expected /= np.linalg.norm(expected)
        np.testing.assert_array_equal(forest.evaluator.weights, expected)
        repeated = AADForest(**params, initial_weights="random").fit(data)
        np.testing.assert_array_equal(forest.score_samples(data), repeated.score_samples(data))
        assert np.isclose(np.linalg.norm(forest.evaluator.weights), 1)
    weights = forest.evaluator.weights.copy()
    forest.fit(data)
    np.testing.assert_array_equal(forest.evaluator.weights, weights)


@pytest.mark.parametrize("initial_weights", ["zero", "random"])
@pytest.mark.parametrize("budget", ["auto", 0.2])
@pytest.mark.parametrize("prior_influence", [0.0, 1.0])
def test_initial_weights_feedback(initial_weights, budget, prior_influence):
    data, _metadata = single_outlier()
    forest = AADForest(
        n_trees=5,
        n_subsamples=32,
        n_jobs=1,
        random_seed=42,
        initial_weights=initial_weights,
        budget=budget,
        prior_influence=prior_influence,
    ).fit(data)
    forest.fit_known(data, data[[0, -1]], np.asarray([Label.REGULAR, Label.ANOMALY]))
    assert np.all(np.isfinite(forest.evaluator.weights))
    assert np.all(np.isfinite(forest.score_samples(data)))
    assert np.isclose(np.linalg.norm(forest.evaluator.weights), 1)


@pytest.mark.parametrize("initial_weights", [None, "invalid", 1])
def test_invalid_initial_weights(initial_weights):
    with pytest.raises(ValueError, match="initial_weights"):
        AADForest(initial_weights=initial_weights)
