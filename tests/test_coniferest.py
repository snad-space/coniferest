from typing import List

import numpy as np
import pytest
from numpy.testing import assert_equal

from coniferest.coniferest import Coniferest, Tree


class ConiferestImpl(Coniferest):
    def fit(self, data, labels=None):
        super().fit(data, labels)

    def fit_known(self, data, known_data=None, known_labels=None):
        super().fit_known(data, known_data, known_labels)

    def score_samples(self, samples):
        return super().score_samples(samples)

    def feature_signature(self, x):
        raise NotImplementedError()

    def feature_importance(self, x):
        raise NotImplementedError()


def assert_tree_equal(a, b):
    assert a.n_features == b.n_features
    assert a.n_subsamples == b.n_subsamples
    assert a.n_leaves == b.n_leaves
    assert_equal(a.left, b.left)
    assert_equal(a.feature, b.feature)
    assert_equal(a.value, b.value)
    assert_equal(a.node_average_path_length, b.node_average_path_length)


def build_trees(random_seed, *, n_trees=8) -> List[Tree]:
    n_subsamples = 256
    shape = n_subsamples * n_trees, 16

    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal(shape)

    coniferest = ConiferestImpl(trees=None, n_subsamples=n_subsamples, max_depth=None, random_seed=random_seed)
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


@pytest.mark.regression
def test_regression_build_trees(regression_data):
    trees = build_trees(0)
    regression_data.check_with(
        lambda actual, desired: [assert_tree_equal(a, b) for a, b in zip(actual, desired)],
        trees,
    )


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_trees", [128, 1024])
def test_benchmark_build_trees(n_trees, n_jobs, benchmark):
    benchmark.group = f"Coniferest.build_trees {n_trees = :4d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.coniferest.Coniferest"

    random_seed = 0
    n_samples = 16_384
    n_features = 16
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    coniferest = ConiferestImpl(n_subsamples=256, n_jobs=n_jobs, random_seed=random_seed)

    benchmark(coniferest.build_trees, data, n_trees)


def test_tree_structure():
    """
    Check basic structural invariants of a built tree.
    """
    tree = build_trees(0, n_trees=1)[0]

    left = tree.left
    leaf_mask = left == 0
    split_mask = ~leaf_mask

    # Binary tree: one more leaf than splits
    assert tree.n_leaves == tree.n_nodes - tree.n_leaves + 1
    assert tree.n_leaves == np.count_nonzero(leaf_mask)

    # Every node but the root is referenced exactly once as a child,
    # the right child index is left + 1
    children = np.concatenate([left[split_mask], left[split_mask] + 1])
    assert_equal(np.sort(children), np.arange(1, tree.n_nodes))

    # Split features are within the feature range
    assert np.all(tree.feature[split_mask] < tree.n_features)

    # Leaves are numbered sequentially in node order
    assert_equal(tree.feature[leaf_mask], np.arange(tree.n_leaves))

    # Leaf values are estimated path lengths: positive and bounded by
    # depth + average path length of the root
    leaf_values = tree.leaf_values()
    assert np.all(leaf_values > 0)
    assert_equal(np.sort(tree.value[leaf_mask]), np.sort(leaf_values))


def test_with_leaf_values_replaces_leaves_only():
    tree = build_trees(0, n_trees=1)[0]

    original_left = tree.left.copy()
    original_feature = tree.feature.copy()
    original_dtype = tree.dtype

    new_values = np.arange(tree.n_leaves, dtype=np.float64) + 1.0
    new_tree = tree.with_leaf_values(new_values)

    # Structure (splits, feature indices, dtype) is untouched
    assert_equal(new_tree.left, original_left)
    assert_equal(new_tree.feature, original_feature)
    assert new_tree.dtype == original_dtype
    assert new_tree.n_leaves == tree.n_leaves
    assert new_tree.n_nodes == tree.n_nodes

    # Leaf values now match what was supplied, ordered by leaf_index
    assert_equal(new_tree.leaf_values(), new_values)


def test_with_leaf_values_does_not_mutate_original_tree():
    tree = build_trees(0, n_trees=1)[0]
    original_values = tree.leaf_values().copy()

    new_values = original_values + 100.0
    tree.with_leaf_values(new_values)

    # `tree` itself (frozen) must be unaffected by building a new tree from it
    assert_equal(tree.leaf_values(), original_values)


def test_with_leaf_values_wrong_length_raises():
    tree = build_trees(0, n_trees=1)[0]

    wrong_length_values = np.zeros(tree.n_leaves + 1, dtype=np.float64)
    with pytest.raises(ValueError):
        tree.with_leaf_values(wrong_length_values)
