import numpy as np
import onnx
import onnxruntime as rt

from coniferest.aadforest import AADForest
from coniferest.datasets import single_outlier
from coniferest.isoforest import IsolationForest
from coniferest.onnx import to_onnx
from coniferest.pineforest import PineForest


def _evaluate_onnx(o, data):
    sess = rt.InferenceSession(o.SerializeToString())
    input_name = sess.get_inputs()[0].name
    label_name = "score"
    return sess.run([label_name], {input_name: data})[0].reshape(-1)


def test_onnx_aadforest():
    data, _metadata = single_outlier()
    model = AADForest(n_trees=10, random_seed=0).fit(data)
    scores_model = model.score_samples(data)

    o = to_onnx(model)
    onnx.checker.check_model(o)
    scores_onnx = _evaluate_onnx(o, data.astype(np.float32))

    np.testing.assert_allclose(scores_onnx, scores_model, 1e-5)


def test_onnx_isoforest():
    data, _metadata = single_outlier()
    data = data.astype(np.float32)
    model = IsolationForest(n_trees=10, random_seed=0).fit(data)
    scores_model = model.score_samples(data)

    o = to_onnx(model)
    onnx.checker.check_model(o)
    scores_onnx = _evaluate_onnx(o, data)

    np.testing.assert_allclose(scores_onnx, scores_model, 1e-5)


def test_onnx_pineforest():
    data, _metadata = single_outlier()
    data = data.astype(np.float32)
    model = PineForest(n_trees=10, random_seed=0).fit(data)
    scores_model = model.score_samples(data)

    o = to_onnx(model)
    onnx.checker.check_model(o)
    scores_onnx = _evaluate_onnx(o, data)

    np.testing.assert_allclose(scores_onnx, scores_model, 1e-5)
