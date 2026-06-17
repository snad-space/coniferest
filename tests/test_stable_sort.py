import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from coniferest.calc_trees import argpartial_sort

def test_stable_sort():
    x = np.asarray([9, 2, 8, 4, 7, 4, 3], dtype=np.float64)
    expected = np.asarray([1, 6, 3, 5])
    assert_equal(expected, argpartial_sort(x, 4))
