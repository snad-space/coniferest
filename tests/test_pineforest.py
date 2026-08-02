import numpy as np
import pytest

from coniferest.label import Label
from coniferest.pineforest import PineForest


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
