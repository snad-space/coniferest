from onnx import defs
from onnxconverter_common import onnx_ex


def get_maximum_opset_supported():
    from . import __max_supported_opset__

    return min(__max_supported_opset__, onnx_ex.get_maximum_opset_supported())
