import numpy as np
import pytest
from coniferest._core import CoreForest, Tree
from numpy.testing import assert_equal

from coniferest.coniferest import Coniferest


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
    assert a.n_subsamples == b.n_subsamples
    assert a.n_leaves == b.n_leaves
    assert_equal(a.left, b.left)
    assert_equal(a.feature, b.feature)
    assert_equal(a.value, b.value)
    assert_equal(a.node_average_path_length, b.node_average_path_length)


def build_forest(random_seed, *, n_trees=8, n_jobs=-1) -> CoreForest:
    n_subsamples = 256
    shape = n_subsamples * n_trees, 16

    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal(shape)

    coniferest = ConiferestImpl(
        core_forest=None, n_subsamples=n_subsamples, max_depth=None, n_jobs=n_jobs, random_seed=random_seed
    )
    return coniferest.build_forest(data, n_trees)


def test_reproducibility_build_trees():
    """
    Are we able to reproduce Coniferest.build_forest
    """
    random_seed = np.random.randint(1 << 16)

    forest1 = build_forest(random_seed)
    forest2 = build_forest(random_seed)

    for tree1, tree2 in zip(forest1, forest2):
        assert_tree_equal(tree1, tree2)


@pytest.mark.regression
def test_regression_build_trees(regression_data):
    trees = build_forest(0)
    regression_data.check_with(
        lambda actual, desired: [assert_tree_equal(a, b) for a, b in zip(actual, desired)],
        trees,
    )


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_trees", [128, 1024])
def test_benchmark_build_trees(n_trees, n_jobs, benchmark):
    benchmark.group = f"Coniferest.build_forest {n_trees = :4d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.coniferest.Coniferest"

    random_seed = 0
    n_samples = 16_384
    n_features = 16
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    coniferest = ConiferestImpl(n_subsamples=256, n_jobs=n_jobs, random_seed=random_seed)

    benchmark(coniferest.build_forest, data, n_trees)


def test_tree_structure():
    """
    Check basic structural invariants of a built tree.
    """
    forest = build_forest(0, n_trees=1)
    tree = forest[0]

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
    assert np.all(tree.feature[split_mask] < forest.n_features)

    # Leaves are numbered sequentially in node order
    assert_equal(tree.feature[leaf_mask], np.arange(tree.n_leaves))

    # Leaf values are estimated path lengths: positive and bounded by
    # depth + average path length of the root
    leaf_values = tree.leaf_values()
    assert np.all(leaf_values > 0)
    assert_equal(np.sort(tree.value[leaf_mask]), np.sort(leaf_values))


def _small_core_forest(dtype, *, n_features=4, seed=0, n_jobs=-1) -> CoreForest:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((256, n_features)).astype(dtype)
    coniferest = ConiferestImpl(core_forest=None, n_subsamples=64, max_depth=8, n_jobs=n_jobs, random_seed=seed)
    return coniferest.build_forest(data, n_trees=3)


def test_core_forest_reconstruct_from_trees(n_jobs):
    """A forest can be rebuilt from its own list of trees."""
    forest = build_forest(0, n_jobs=n_jobs)
    rebuilt = CoreForest(list(forest), n_features=forest.n_features, num_threads=forest.num_threads)

    assert len(rebuilt) == len(forest)
    assert rebuilt.n_features == forest.n_features
    for tree1, tree2 in zip(forest, rebuilt):
        assert_tree_equal(tree1, tree2)


def test_core_forest_pickle_roundtrip(n_jobs):
    """Pickling a forest preserves its trees and attributes."""
    import pickle

    forest = build_forest(0, n_jobs=n_jobs)
    reforest = pickle.loads(pickle.dumps(forest))

    assert len(reforest) == len(forest)
    assert reforest.n_features == forest.n_features
    assert reforest.num_threads == forest.num_threads
    for tree1, tree2 in zip(forest, reforest):
        assert_tree_equal(tree1, tree2)


def test_core_forest_rejects_empty(n_jobs):
    num_threads = ConiferestImpl(n_jobs=n_jobs).num_threads
    with pytest.raises(ValueError):
        CoreForest([], n_features=4, num_threads=num_threads)


def test_core_forest_rejects_dtype_mismatch(n_jobs):
    forest32 = _small_core_forest(np.float32, n_jobs=n_jobs)
    forest64 = _small_core_forest(np.float64, n_jobs=n_jobs)
    with pytest.raises(TypeError):
        CoreForest(list(forest32) + list(forest64), n_features=4, num_threads=forest32.num_threads)


def test_core_forest_rejects_too_few_features(n_jobs):
    forest = _small_core_forest(np.float64, n_features=4, n_jobs=n_jobs)
    with pytest.raises(ValueError):
        CoreForest(list(forest), n_features=1, num_threads=forest.num_threads)


#
# Sequence protocol
#


def test_core_forest_len_and_iteration(n_jobs):
    """len(), iteration, and indexing agree and yield Tree objects."""
    forest = build_forest(0, n_jobs=n_jobs)
    trees = list(forest)

    assert len(forest) == len(trees)
    assert len(trees) > 0
    assert all(isinstance(tree, Tree) for tree in trees)
    for i, tree in enumerate(forest):
        assert_tree_equal(tree, forest[i])


def test_core_forest_getitem_out_of_range(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    with pytest.raises(IndexError):
        forest[len(forest)]


def assert_forest_holds(forest, trees):
    """The forest holds exactly `trees`, in that order."""
    assert isinstance(forest, CoreForest)
    assert len(forest) == len(trees)
    for actual, expected in zip(forest, trees):
        assert_tree_equal(actual, expected)


def test_core_forest_getitem_negative(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    trees = list(forest)

    assert isinstance(forest[-1], Tree)
    assert_tree_equal(forest[-1], trees[-1])
    assert_tree_equal(forest[-len(forest)], trees[0])
    with pytest.raises(IndexError):
        forest[-len(forest) - 1]


@pytest.mark.parametrize(
    "index", [slice(2, 5), slice(None, None, 2), slice(None, None, -1), slice(-3, None), slice(100, 200)]
)
def test_core_forest_getitem_slice(index, n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    trees = list(forest)

    assert_forest_holds(forest[index], trees[index])


def test_core_forest_getitem_int_array(n_jobs):
    """Fancy indexing takes arrays or lists, and may repeat or reorder trees."""
    forest = build_forest(0, n_jobs=n_jobs)
    trees = list(forest)

    assert_forest_holds(forest[np.array([5, 0, 3])], [trees[i] for i in (5, 0, 3)])
    assert_forest_holds(forest[[5, 0, 3]], [trees[i] for i in (5, 0, 3)])
    assert_forest_holds(forest[np.array([2, 2, 2])], [trees[2]] * 3)
    assert_forest_holds(forest[np.array([-1, -len(forest)])], [trees[-1], trees[0]])
    assert_forest_holds(forest[np.array([], dtype=int)], [])


def test_core_forest_getitem_bool_mask(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    trees = list(forest)
    mask = np.zeros(len(forest), dtype=bool)
    mask[[1, 4, 7]] = True

    assert_forest_holds(forest[mask], [trees[i] for i in (1, 4, 7)])
    assert_forest_holds(forest[np.zeros(len(forest), dtype=bool)], [])


def test_core_forest_getitem_fancy_keeps_attributes(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    selected = forest[np.array([1, 0])]

    assert selected.n_features == forest.n_features
    assert selected.num_threads == forest.num_threads


@pytest.mark.parametrize(
    "index",
    [
        np.array([0, 8]),
        np.array([-9, 0]),
        np.zeros(7, dtype=bool),
        np.zeros(9, dtype=bool),
    ],
)
def test_core_forest_getitem_fancy_out_of_range(index, n_jobs):
    forest = build_forest(0, n_trees=8, n_jobs=n_jobs)
    with pytest.raises(IndexError):
        forest[index]


def test_core_forest_setitem(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    replacement = forest[1]
    forest[0] = replacement
    assert_tree_equal(forest[0], replacement)


def test_core_forest_setitem_dtype_mismatch(n_jobs):
    forest64 = _small_core_forest(np.float64, n_jobs=n_jobs)
    forest32 = _small_core_forest(np.float32, n_jobs=n_jobs)
    with pytest.raises(TypeError):
        forest64[0] = forest32[0]


def test_core_forest_delitem(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    n = len(forest)
    kept = [forest[i] for i in range(n) if i != 1]

    del forest[1]

    assert len(forest) == n - 1
    for tree, expected in zip(forest, kept):
        assert_tree_equal(tree, expected)


def test_core_forest_delitem_out_of_range(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    with pytest.raises(IndexError):
        del forest[len(forest)]


def test_core_forest_concat(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    other = build_forest(1, n_jobs=n_jobs)

    combined = forest + other

    assert len(combined) == len(forest) + len(other)
    assert combined.n_features == forest.n_features
    for tree, expected in zip(combined, list(forest) + list(other)):
        assert_tree_equal(tree, expected)


def test_core_forest_concat_feature_mismatch(n_jobs):
    forest4 = _small_core_forest(np.float64, n_features=4, n_jobs=n_jobs)
    forest3 = _small_core_forest(np.float64, n_features=3, n_jobs=n_jobs)
    with pytest.raises(ValueError):
        forest4 + forest3


def test_core_forest_inplace_concat(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    other = build_forest(1, n_jobs=n_jobs)
    expected = list(forest) + list(other)
    original = forest

    forest += other

    assert forest is original
    assert len(forest) == len(expected)
    assert len(other) == len(expected) // 2
    assert forest.n_features == other.n_features
    for tree, tree_expected in zip(forest, expected):
        assert_tree_equal(tree, tree_expected)


def test_core_forest_inplace_concat_self(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    expected = list(forest) * 2
    original = forest

    forest += forest

    assert forest is original
    assert len(forest) == len(expected)
    for tree, tree_expected in zip(forest, expected):
        assert_tree_equal(tree, tree_expected)


def test_core_forest_inplace_concat_feature_mismatch(n_jobs):
    forest4 = _small_core_forest(np.float64, n_features=4, n_jobs=n_jobs)
    forest3 = _small_core_forest(np.float64, n_features=3, n_jobs=n_jobs)
    n = len(forest4)
    with pytest.raises(ValueError):
        forest4 += forest3
    assert len(forest4) == n


def test_core_forest_inplace_concat_dtype_mismatch(n_jobs):
    forest64 = _small_core_forest(np.float64, n_jobs=n_jobs)
    forest32 = _small_core_forest(np.float32, n_jobs=n_jobs)
    n = len(forest64)
    with pytest.raises(TypeError):
        forest64 += forest32
    assert len(forest64) == n


def test_core_forest_repeat(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    n = len(forest)

    repeated = forest * 3

    assert len(repeated) == 3 * n
    assert repeated.n_features == forest.n_features
    for tree, expected in zip(repeated, list(forest) * 3):
        assert_tree_equal(tree, expected)


def test_core_forest_repeat_zero(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    empty = forest * 0
    assert len(empty) == 0
    assert empty.n_features == forest.n_features


def test_core_forest_inplace_repeat(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    expected = list(forest) * 3
    original = forest

    forest *= 3

    assert forest is original
    assert len(forest) == len(expected)
    assert forest.n_features == original.n_features
    for tree, tree_expected in zip(forest, expected):
        assert_tree_equal(tree, tree_expected)


def test_core_forest_inplace_repeat_one(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    expected = list(forest)
    original = forest

    forest *= 1

    assert forest is original
    for tree, tree_expected in zip(forest, expected):
        assert_tree_equal(tree, tree_expected)


def test_core_forest_inplace_repeat_zero(n_jobs):
    forest = build_forest(0, n_jobs=n_jobs)
    n_features = forest.n_features
    original = forest

    forest *= 0

    assert forest is original
    assert len(forest) == 0
    assert forest.n_features == n_features
