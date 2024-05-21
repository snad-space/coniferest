import pickle
from functools import partial

import pytest
from numpy.testing import assert_allclose

PICKLE_PROTOCOL = 4  # maximum supported by python 3.7


class RegressionData:
    def __init__(self, path):
        self.path = path
        try:
            with open(self.path, 'rb') as fh:
                self.obj = pickle.load(fh)
        except FileNotFoundError:
            self.obj = None

    def check_with(self, assert_func, actual):
        if self.obj is None:
            with open(self.path, 'wb') as fh:
                pickle.dump(actual, fh, protocol=PICKLE_PROTOCOL)
            assert False, f'File {self.path} does not exist, given object saved to this location'
        assert_func(actual, self.obj)

    def allclose(self, a, **kwargs):
        self.check_with(partial(assert_allclose, **kwargs), a)


@pytest.fixture
def regression_data(request):
    path = request.path.parent.joinpath(request.path.stem, f'{request.function.__name__}.pickle')
    return RegressionData(path)


def pytest_addoption(parser):
    parser.addoption("--n_jobs", default=1, type=int, help="n_jobs parameter for benchmarks")


@pytest.fixture
def n_jobs(request):
    return request.config.getoption("--n_jobs")
