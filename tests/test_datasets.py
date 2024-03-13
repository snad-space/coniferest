import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from coniferest.datasets import ztf_m31, dev_net_dataset


def test_ztf_m31():
    data, metadata = ztf_m31()

    assert data.shape == (57546, 42)
    assert data.dtype == np.float32
    assert_allclose(data.sum(axis=1)[13], 3.4734121e+09, atol=1e-6)

    assert metadata.shape == (data.shape[0],)
    assert metadata.dtype == np.uint64
    assert_allclose(data.mean(), 100942920)


def test_dev_net():
    data, metadata = dev_net_dataset("thyroid")

    assert data.shape == (7200, 21)

    assert metadata.shape == (data.shape[0],)
    assert_array_equal(np.unique(metadata), [-1, 1])
