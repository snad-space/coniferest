import numpy as np
import pytest

from coniferest.aadforest import AADForest
from coniferest.datasets import single_outlier


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
    forest = AADForest(n_trees=n_trees, random_seed=random_seed, n_jobs=n_thread, sampletrees_per_batch=2048, budget=0.03)
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
