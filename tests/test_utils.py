import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from coniferest.utils import average_path_length


def _reference_average_path_length(n):
    """Reference implementation of the Liu et al. 2008 formula in float64."""
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # n <= 1 entries produce spurious warnings here but are masked out below.
        result = 2.0 * (np.log(n) + np.euler_gamma + 1.0 / (2.0 * n) - 1.0 / (12.0 * n * n) - 1.0)
    # The formula is only defined for n > 1; otherwise the path length is zero.
    return np.where(n > 1.0, result, 0.0)


# --- scalar input -----------------------------------------------------------


def test_scalar_returns_zero_dim_array():
    # Scalar input is turned into a 0-d array, so the result is a 0-d ndarray.
    result = average_path_length(10)
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == np.float64


@pytest.mark.parametrize("n", [2, 3, 5, 10, 100, 256, 1000])
def test_scalar_matches_reference(n):
    assert_allclose(average_path_length(n), _reference_average_path_length(n))


@pytest.mark.parametrize("n", [0, 1])
def test_scalar_zero_and_one_are_zero(n):
    # n <= 1 has no unsuccessful-search depth, so the path length is exactly 0.
    assert average_path_length(n) == 0.0


@pytest.mark.parametrize("n", [-1, -100])
def test_scalar_negative_is_zero(n):
    assert average_path_length(n) == 0.0


def test_scalar_just_above_one():
    # Float scalars flow through the same n <= 1 branch as integers.
    assert average_path_length(1.0) == 0.0
    assert average_path_length(0.5) == 0.0
    assert average_path_length(2.0) > 0.0


def test_scalar_monotonic_increasing():
    values = [average_path_length(n) for n in range(2, 50)]
    assert np.all(np.diff(values) > 0.0)


# --- array input ------------------------------------------------------------


def test_array_returns_ndarray():
    result = average_path_length(np.asarray([2, 3, 4]))
    assert isinstance(result, np.ndarray)


def test_array_matches_reference():
    n = np.asarray([0, 1, 2, 5, 10, 100, 1000], dtype=np.int64)
    assert_allclose(average_path_length(n), _reference_average_path_length(n))


def test_array_edge_values():
    n = np.asarray([-5, -1, 0, 1], dtype=np.int64)
    assert_array_equal(average_path_length(n), np.zeros(4))


def test_empty_array():
    result = average_path_length(np.asarray([], dtype=np.float64))
    assert result.shape == (0,)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 2)])
def test_shape_is_preserved(shape):
    n = np.full(shape, 10, dtype=np.float64)
    result = average_path_length(n)
    assert result.shape == shape


# --- dtypes -----------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_integer_dtypes_return_float64(dtype):
    n = np.asarray([2, 10, 100], dtype=dtype)
    result = average_path_length(n)
    assert result.dtype == np.float64
    assert_allclose(result, _reference_average_path_length(n))


def test_float32_keeps_float32():
    n = np.asarray([2, 10, 100], dtype=np.float32)
    result = average_path_length(n)
    assert result.dtype == np.float32
    # Computed in single precision, so compare against the reference loosely.
    assert_allclose(result, _reference_average_path_length(n), rtol=1e-6)


def test_float64_keeps_float64():
    n = np.asarray([2, 10, 100], dtype=np.float64)
    result = average_path_length(n)
    assert result.dtype == np.float64
    assert_allclose(result, _reference_average_path_length(n))


def test_python_default_int_dtype():
    # np.asarray([...]) of Python ints picks the platform default integer dtype.
    n = np.asarray([2, 10, 100])
    result = average_path_length(n)
    assert result.dtype == np.float64
    assert_allclose(result, _reference_average_path_length(n))


@pytest.mark.parametrize("dtype", [np.float16, np.complex64, np.complex128, np.bool_])
def test_unsupported_dtypes_raise_type_error(dtype):
    n = np.asarray([1, 2, 3], dtype=dtype)
    with pytest.raises(TypeError):
        average_path_length(n)
