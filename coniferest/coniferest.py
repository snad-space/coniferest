import numpy as np

from sklearn.tree._criterion import MSE  # noqa
from sklearn.tree._splitter import RandomSplitter  # noqa
from sklearn.tree._tree import Tree, DepthFirstTreeBuilder  # noqa
from sklearn.ensemble._bagging import _generate_indices

from .evaluator import ForestEvaluator
from .utils import average_path_length

# Instead of doing:
# from sklearn.utils._random import RAND_R_MAX
# we have:
RAND_R_MAX = 0x7FFFFFFF
# Cause RAND_R_MAX in restricted to C-code.


class Coniferest:
    def __init__(self, trees, n_subsamples=256, max_depth=None, random_state=None):
        self.trees = trees
        self.n_subsamples = n_subsamples
        self.max_depth = max_depth or int(np.log2(n_subsamples))

        # self.seedseq = np.random.SeedSequence(random_state)
        # seed, = self.seedseq.spawn(1)
        # self.rng = np.random.default_rng(seed)

        self.rng = np.random.default_rng(random_state)

        self.bootstrap_samples = False
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_leaf = 0
        self.min_impurity_decrease = 0
        self.n_outputs = 1

        self.criterion = MSE(self.n_outputs, self.n_subsamples)

        splitter_seed =  self.rng.integers(RAND_R_MAX)
        self.splitter = RandomSplitter(criterion=self.criterion,
                                       max_features=1,
                                       min_samples_leaf=self.min_samples_leaf,
                                       min_weight_leaf=self.min_weight_leaf,
                                       random_state=splitter_seed)

        self.builder = DepthFirstTreeBuilder(splitter=self.splitter,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_weight_leaf,
                                             min_weight_leaf=self.min_weight_leaf,
                                             max_depth=self.max_depth,
                                             min_impurity_decrease=self.min_impurity_decrease)

    def build_trees(self, data, n_trees):
        n_population, n_features = data.shape

        trees = []
        for tree_index in range(n_trees):
            random_state, =  self.rng.integers(RAND_R_MAX)
            indices = _generate_indices(random_state=random_state,
                                        bootstrap=self.bootstrap_samples,
                                        n_population=n_population,
                                        n_samples=self.n_subsamples)

            tree = self.build_one_tree(data[indices, :])
            trees.append(tree)

        return trees

    def build_one_tree(self, data):
        n_samples, n_features = data.shape
        tree = Tree(n_features, np.array([1] * self.n_outputs, dtype=np.intp), self.n_outputs)
        y = np.empty((n_samples, self.n_outputs))
        self.builder.build(tree, data, y)
        return tree

    def fit(self, data):
        raise NotImplementedError()

    def score_samples(self, samples):
        raise NotImplementedError()


class ConiferestEvaluator(ForestEvaluator):

    def __init__(self, coniferest):
        selectors_list = [self.extract_selectors(e) for e in coniferest.trees]
        selectors, indices = self.combine_selectors(selectors_list)

        super(ConiferestEvaluator, self).__init__(
            samples=coniferest.subsamples,
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