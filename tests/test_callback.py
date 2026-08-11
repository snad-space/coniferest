import os

import onnx
import pytest

from coniferest.datasets import single_outlier
from coniferest.isoforest import IsolationForest
from coniferest.session.callback import SaveToOnnx


class _FakeSession:
    def __init__(self, model):
        self.model = model


def _get_fitted_model():
    data, _metadata = single_outlier()
    model = IsolationForest(n_trees=10, random_seed=0).fit(data)
    return model


def test_save_to_onnx_overwrite(tmp_path):
    model = _get_fitted_model()
    fake_session = _FakeSession(model)

    callback = SaveToOnnx(
        directory=str(tmp_path),
        filename="model.onnx",
        every_n_decisions=2,
        overwrite=True,
    )

    expected_path = os.path.join(str(tmp_path), "model.onnx")

    # First call: counter=1, should NOT save yet
    callback(None, None, fake_session)
    assert not os.path.exists(expected_path)

    # Second call: counter=2, should save now
    callback(None, None, fake_session)
    assert os.path.exists(expected_path)

    onnx_model = onnx.load(expected_path)
    onnx.checker.check_model(onnx_model)

    # Third and fourth calls: still overwriting same file
    callback(None, None, fake_session)
    callback(None, None, fake_session)
    assert os.path.exists(expected_path)


def test_save_to_onnx_numbered_files(tmp_path):
    model = _get_fitted_model()
    fake_session = _FakeSession(model)

    callback = SaveToOnnx(
        directory=str(tmp_path),
        filename="model.onnx",
        every_n_decisions=1,
        overwrite=False,
    )

    callback(None, None, fake_session)
    callback(None, None, fake_session)

    path_1 = os.path.join(str(tmp_path), "model_1.onnx")
    path_2 = os.path.join(str(tmp_path), "model_2.onnx")

    assert os.path.exists(path_1)
    assert os.path.exists(path_2)


def test_save_to_onnx_raises_on_unfitted_model(tmp_path):
    unfitted_model = IsolationForest(n_trees=10, random_seed=0)
    fake_session = _FakeSession(unfitted_model)

    callback = SaveToOnnx(directory=str(tmp_path), every_n_decisions=1)

    with pytest.raises(RuntimeError):
        callback(None, None, fake_session)
