import numpy as np
import pytest
from coniferest.calc_trees import argpartial_sort
from numpy.testing import assert_equal


def test_stable_sort():
    x = np.asarray([9, 2, 8, 4, 7, 4, 3], dtype=np.float64)
    expected = np.asarray([1, 6, 3, 5])
    assert_equal(expected, argpartial_sort(x, 4))


def test_empty_array():
    x = np.asarray([], dtype=np.float64)
    expected = np.asarray([], dtype=np.intp)
    assert_equal(expected, argpartial_sort(x, 0))
    assert_equal(expected, argpartial_sort(x, 5))


def test_nonempty_k0():
    x = np.asarray([9, 2, 8, 4], dtype=np.float64)
    expected = np.asarray([], dtype=np.intp)
    assert_equal(expected, argpartial_sort(x, 0))


def test_k_larger_than_array():
    x = np.asarray([3, 1, 2], dtype=np.float64)
    expected = np.asarray([1, 2, 0])
    assert_equal(expected, argpartial_sort(x, 5))


@pytest.mark.benchmark(min_rounds=10, disable_gc=True, warmup=False)
@pytest.mark.long
@pytest.mark.parametrize("size", [128, 1024, 4096])
@pytest.mark.parametrize("k", [1, 10, 100, 10000])
def test_benchmark_argpartial_sort(size, k, benchmark):
    benchmark.group = f"argpartial_sort {size = :4d}, {k = :2d}"
    benchmark.name = "coniferest.calc_trees.argpartial_sort"

    rng = np.random.default_rng(seed=0)

    arr = np.linspace(-1, 1, size, dtype=np.float64)
    x = arr.copy()

    rng.shuffle(x)

    @benchmark
    def construct():
        argpartial_sort(x, k)


@pytest.mark.benchmark(min_rounds=10, disable_gc=True, warmup=False)
@pytest.mark.long
@pytest.mark.parametrize("size", [128, 1024, 4096])
@pytest.mark.parametrize("k", [1, 10, 100, 10000])
def test_benchmark_argpartition(size, k, benchmark):
    benchmark.group = f"argpartial_sort {size = :4d}, {k = :2d}"
    benchmark.name = "numpy.argpartition"

    rng = np.random.default_rng(seed=0)

    arr = np.linspace(-1, 1, size, dtype=np.float64)
    x = arr.copy()

    rng.shuffle(x)

    def argtopk_scores(x, k: int) -> np.ndarray:
        if k >= x.shape[0]:
            return np.argsort(x)
        argtopk_unsort = np.argpartition(x, k)[:k]
        argtopk = argtopk_unsort[np.argsort(x[argtopk_unsort])]
        return argtopk

    @benchmark
    def construct():
        argtopk_scores(x, k)
