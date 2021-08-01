import numpy as np
from .coniferest import Coniferest, ConiferestEvaluator


class Isoforest(Coniferest):
    def __init__(self, n_trees=100, n_subsamples=256, max_depth=None, random_state=None):
        super(Isoforest, self).__init__(trees=[],
                                        n_subsamples=n_subsamples,
                                        max_depth=max_depth,
                                        random_state=random_state)
        self.n_trees = n_trees
        self.evaluator = None

    def fit(self, data):
        self.trees = self.build_trees(data, self.n_trees)
        self.evaluator = ConiferestEvaluator(self)
        return self

    def score_samples(self, samples):
        return self.evaluator.score_samples(samples)
