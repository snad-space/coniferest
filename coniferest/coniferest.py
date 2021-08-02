import numpy as np
from pkg_resources import parse_version

import sklearn
from sklearn.tree._criterion import MSE  # noqa
from sklearn.tree._splitter import RandomSplitter  # noqa
from sklearn.tree._tree import Tree, DepthFirstTreeBuilder  # noqa
from sklearn.ensemble._bagging import _generate_indices  # noqa
from sklearn.utils.validation import check_random_state

from .evaluator import ForestEvaluator
from .utils import average_path_length

# Instead of doing:
# from sklearn.utils._random import RAND_R_MAX
# we have:
RAND_R_MAX = 0x7FFFFFFF
# Cause RAND_R_MAX is restricted to C-code.


class Coniferest:
    def __init__(self, trees=None, n_subsamples=256, max_depth=None, random_seed=None):
        self.trees = trees or []
        self.n_subsamples = n_subsamples
        self.max_depth = max_depth or int(np.log2(n_subsamples))

        # self.seedseq = np.random.SeedSequence(random_state)
        # seed, = self.seedseq.spawn(1)
        # self.rng = np.random.default_rng(seed)

        self.rng = np.random.default_rng(random_seed)

        self.bootstrap_samples = False
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_leaf = 0
        self.min_impurity_decrease = 0
        self.n_outputs = 1

    def build_trees(self, data, n_trees):
        n_population, n_features = data.shape

        trees = []
        for tree_index in range(n_trees):
            random_state = check_random_state(self.rng.integers(RAND_R_MAX))
            indices = _generate_indices(random_state=random_state,
                                        bootstrap=self.bootstrap_samples,
                                        n_population=n_population,
                                        n_samples=self.n_subsamples)

            subsamples = data[indices, :]
            tree = self.build_one_tree(subsamples)
            trees.append(tree)

        return trees

    def build_one_tree(self, data):
        criterion = MSE(self.n_outputs, self.n_subsamples)

        splitter_state = check_random_state(self.rng.integers(RAND_R_MAX))
        splitter = RandomSplitter(criterion=criterion,
                                  max_features=1,
                                  min_samples_leaf=self.min_samples_leaf,
                                  min_weight_leaf=self.min_weight_leaf,
                                  random_state=splitter_state)

        builder_args = {
            'splitter': splitter,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_leaf': self.min_weight_leaf,
            'max_depth': self.max_depth,
            'min_impurity_decrease': self.min_impurity_decrease
        }

        if parse_version(sklearn.__version__) < parse_version('0.25.0'):
            builder_args['min_impurity_split'] = 0

        builder = DepthFirstTreeBuilder(**builder_args)

        n_samples, n_features = data.shape
        tree = Tree(n_features, np.array([1] * self.n_outputs, dtype=np.int64), self.n_outputs)

        # Cause of sklearn bugs we cannot do this:
        # y = np.zeros((n_samples, self.n_outputs))
        # Instead we do:
        y = np.empty((n_samples, self.n_outputs))
        y_column = np.arange(n_samples)
        for oi in range(self.n_outputs):
            y[:, oi] = y_column
        # The counterpart is rnd.uniform from sklearn.ensemble.IsolationForest.fit.

        builder.build(tree, data, y)

        return tree

    def fit(self, data, labels=None):
        raise NotImplementedError()

    def score_samples(self, samples):
        raise NotImplementedError()


class ConiferestEvaluator(ForestEvaluator):

    def __init__(self, coniferest):
        selectors_list = [self.extract_selectors(t) for t in coniferest.trees]
        selectors, indices = self.combine_selectors(selectors_list)

        super().__init__(
            samples=coniferest.n_subsamples,
            selectors=selectors,
            indices=indices)

    @classmethod
    def extract_selectors(cls, tree):
        nodes = tree.__getstate__()['nodes']
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