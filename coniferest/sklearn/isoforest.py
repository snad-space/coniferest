import numpy as np
from sklearn.ensemble import IsolationForest
from ..evaluator import ForestEvaluator
from ..experiment import AnomalyDetector
from ..utils import average_path_length


class IsoforestEvaluator(ForestEvaluator):

    def __init__(self, isoforest):
        selectors_list = [self.extract_selectors(e) for e in isoforest.estimators_]
        selectors, indices = self.combine_selectors(selectors_list)

        super(IsoforestEvaluator, self).__init__(
            samples=isoforest.max_samples_,
            selectors=selectors,
            indices=indices)

    @classmethod
    def extract_selectors(cls, estimator):
        nodes = estimator.tree_.__getstate__()['nodes']
        selectors = np.zeros_like(nodes, dtype=cls.selector_dtype)

        selectors['feature'] = nodes['feature']
        selectors['feature'][selectors['feature'] < 0] = -1

        selectors['left'] = nodes['left_child']
        selectors['right'] = nodes['right_child']
        selectors['value'] = nodes['threshold']

        n_node_samples = nodes['n_node_samples']

        def correct_values(i, depth):
            if selectors[i]['feature'] < 0:
                selectors[i]['value'] = depth + average_path_length(n_node_samples[i])
            else:
                correct_values(selectors[i]['left'], depth + 1)
                correct_values(selectors[i]['right'], depth + 1)

        correct_values(0, 0)

        return selectors


class ClassicIsoforestAnomalyDetector(AnomalyDetector):
    def __init__(self, title='Classic Isolation Forest', **kwargs):
        super().__init__(title)
        self.isoforest = IsolationForest(**kwargs)

    def train(self, data):
        return self.isoforest.fit(data)

    def score(self, data):
        return self.isoforest.score_samples(data)

    def observe(self, point, label):
        super().observe(point, label)
        # do nothing, it's classic, you know
        return False
