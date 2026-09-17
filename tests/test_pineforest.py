import numpy as np
import pytest

from coniferest.label import Label
from coniferest.pineforest import PineForest


# Single-thread and parallel implementations are a bit different, so here we check both.
# We use n_thread parameter instead of n_jobs, which is a fixture in conftest.py
@pytest.mark.parametrize("n_thread", [1, 2])
@pytest.mark.regression
def test_regression_fit_known(n_thread, regression_data):
    """Scores after fitting on labeled data, which grows the spare trees and filters them out again."""
    random_seed = 0
    n_samples = 1024
    n_features = 16
    n_known = 16
    n_trees = 32
    n_spare_trees = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    known_data = data[rng.choice(n_samples, n_known, replace=False)]
    known_labels = rng.choice([Label.ANOMALY, Label.REGULAR], n_known)

    # This small sampletrees_per_batch is inefficient, but it's good for testing to guarantee parallel execution.
    forest = PineForest(
        n_trees=n_trees,
        n_spare_trees=n_spare_trees,
        random_seed=random_seed,
        n_jobs=n_thread,
        sampletrees_per_batch=2048,
    )
    forest.fit_known(data, known_data=known_data, known_labels=known_labels)

    # The spare trees must have been filtered out, i.e. filter_trees ran
    assert len(forest.core_forest) == n_trees

    scores = forest.score_samples(data)
    regression_data.assert_allclose(scores)


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_known", [1 << 0, 1 << 6])
def test_benchmark_filter_trees(n_known, n_jobs, benchmark):
    benchmark.group = f"PineForest.filter_trees {n_known = :3d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.pineforest.PineForest"

    random_seed = 0
    n_samples = 1 << 12
    n_features = 16
    n_trees = 128
    n_spare_trees = 512
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    known_data = data[rng.choice(n_samples, n_known, replace=False)]
    known_labels = rng.choice([Label.ANOMALY, Label.REGULAR], n_known)

    forest = PineForest(
        n_trees=n_trees,
        n_spare_trees=n_spare_trees,
        n_jobs=n_jobs,
        random_seed=random_seed,
    )
    # The pre-filter superset the session builds before contracting back to n_trees
    core_forest = forest.build_forest(data, n_trees + n_spare_trees)

    benchmark(forest.filter_trees, core_forest, known_data, known_labels, n_filter=n_spare_trees)
