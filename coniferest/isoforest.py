from .coniferest import Coniferest, ConiferestEvaluator
from .experiment import AnomalyDetector


class IsolationForest(Coniferest):
    def __init__(self, n_trees=100, n_subsamples=256, max_depth=None, random_seed=None):
        super().__init__(trees=[],
                         n_subsamples=n_subsamples,
                         max_depth=max_depth,
                         random_seed=random_seed)
        self.n_trees = n_trees
        self.evaluator = None

    def fit(self, data, labels=None):
        self.trees = self.build_trees(data, self.n_trees)
        self.evaluator = ConiferestEvaluator(self)
        return self

    def score_samples(self, samples):
        return self.evaluator.score_samples(samples)


class IsolationForestAnomalyDetector(AnomalyDetector):
    def __init__(self, isoforest, title='Isolation Forest'):
        super().__init__(title)
        self.isoforest = isoforest

    def train(self, data):
        return self.isoforest.fit(data)

    def score(self, data):
        return self.isoforest.score_samples(data)

    def observe(self, point, label):
        super().observe(point, label)
        # do nothing, it's classic, you know
        return False
