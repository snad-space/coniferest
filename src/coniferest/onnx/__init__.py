from . import (
    operator_converters,  # noqa: F401
    shape_calculators,  # noqa: F401
)
from .convert import convert, to_onnx  # noqa: F401

__max_supported_opset__ = 15
