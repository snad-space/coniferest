import numpy as np
from numpy.testing import assert_allclose

from coniferest.datasets import ztf_m31


def test_ztf_m31():
    data, metadata = ztf_m31()

    assert data.shape == (57546, 42)
    assert data.dtype == np.float32
    assert_allclose(data.sum(axis=1)[13], 3.4734121e+09, atol=1e-6)

    assert metadata.shape == (data.shape[0],)
    assert metadata.dtype == np.uint64
    assert_allclose(data.mean(), 100942920)
