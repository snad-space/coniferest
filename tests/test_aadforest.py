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
    forest = AADForest(n_trees=10, random_seed=0, prior_influence=lambda ac, nc: np.max(1.0, 0.5 / (ac + nc))).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1
