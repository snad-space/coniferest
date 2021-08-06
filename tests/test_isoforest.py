import pytest

import numpy as np

from coniferest.sklearn.isoforest import IsolationForestEvaluator
from coniferest.isoforest import IsolationForest
from sklearn.ensemble import IsolationForest as SkIsolationForest

from coniferest.datasets import MalanchevDataset


@pytest.fixture()
def isoforest_results():
    return IsoforestResults()


class IsoforestResults:
    def __init__(self):
        seed = 622341
        self.dataset = MalanchevDataset(inliers=1000,
                                        outliers=50,
                                        regions=[1, 1, -1],
                                        seed=seed)

        data = self.dataset.data
        trees = 1000

        forest = IsolationForest(n_trees=trees, random_seed=seed + 1)
        self.scores = self.calc_forest_scores(forest, data)
        self.forest = forest

        forest = SkIsolationForest(n_estimators=trees, random_state=seed + 2)
        self.skores0 = self.calc_forest_scores(forest, data)

        forest = SkIsolationForest(n_estimators=trees, random_state=seed + 3)
        self.skores1 = self.calc_forest_scores(forest, data)
        self.skores1_by_evaluator = IsolationForestEvaluator(forest).score_samples(data)

    @staticmethod
    def calc_forest_scores(forest, data):
        forest.fit(data)
        return forest.score_samples(data)


def test_sklearn_isolation_forest_evaluator(isoforest_results):
    """
    Does evaluator scores coinside with the ones computed by sklearn?
    """
    r = isoforest_results
    assert np.max(np.abs(r.skores1_by_evaluator - r.skores1)) <= 1e-10


def test_isolation_forest(isoforest_results):
    """
    Does assigned scores by our isoforest are somewhere near assigned by sklearn's isoforest?
    """
    r = isoforest_results
    diff_sk_to_sk = np.max(np.abs(r.skores0 - r.skores1))
    diff_coni_to_sk = np.max(np.abs(r.skores0 - r.scores))
    assert diff_coni_to_sk < 1.5 * diff_sk_to_sk


def test_serialization(isoforest_results):
    """
    Does (de)serialization work correctly?
    """
    import pickle

    r = isoforest_results
    s = pickle.dumps(r.forest)
    reforest = pickle.loads(s)
    assert np.allclose(reforest.score_samples(r.dataset.data), r.scores, atol=1e-12)
