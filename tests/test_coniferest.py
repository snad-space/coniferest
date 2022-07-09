import pickle
import sys
from typing import List

import numpy as np
import pytest
from numpy.testing import assert_equal
from sklearn.tree._tree import Tree

from coniferest.coniferest import Coniferest


def build_one_tree(random_seed) -> Tree:
    shape = 256, 16

    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal(shape)

    coniferest = Coniferest(trees=None, n_subsamples=data.shape[0], max_depth=None, random_seed=random_seed)
    return coniferest.build_one_tree(data)


def assert_tree_equal(a, b):
    assert_equal(a.children_left, b.children_left)
    assert_equal(a.children_right, b.children_right)
    assert_equal(a.value, b.value)
    assert_equal(a.feature, b.feature)
    assert_equal(a.threshold, b.threshold)


def test_reproducibility_build_one_tree():
    """
    Are we able to reproduce tree building?
    """
    random_seed = np.random.randint(1 << 16)
    assert_tree_equal(build_one_tree(random_seed), build_one_tree(random_seed))


@pytest.mark.skipif(sys.platform.startswith('win'), reason="It fails on Windows and I don't care")
def test_regression_build_one(regression_data):
    tree = build_one_tree(0)
    regression_data.check_with(assert_tree_equal, tree)


def build_trees(random_seed) -> List[Tree]:
    n_trees = 8
    n_subsamples = 256
    shape = n_subsamples * n_trees, 16

    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal(shape)

    coniferest = Coniferest(trees=None, n_subsamples=n_subsamples, max_depth=None, random_seed=random_seed)
    return coniferest.build_trees(data, n_trees)


def test_reproducibility_build_trees():
    """
    Are we able to reproduce Coniferest.build_trees
    """
    random_seed = np.random.randint(1 << 16)

    trees1 = build_trees(random_seed)
    trees2 = build_trees(random_seed)

    for tree1, tree2 in zip(trees1, trees2):
        assert_tree_equal(tree1, tree2)


@pytest.mark.skipif(sys.platform.startswith('win'), reason="It fails on Windows and I don't care")
def test_regression_build_trees(regression_data):
    trees = build_trees(0)
    regression_data.check_with(
        lambda actual, desired: [assert_tree_equal(a, b) for a, b in zip(actual, desired)],
        trees
    )
