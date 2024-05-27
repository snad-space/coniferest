from onnxconverter_common.registration import register_shape_calculator
from onnxconverter_common.data_types import FloatTensorType
from onnxconverter_common.utils import (
    check_input_and_output_types,
    check_input_and_output_numbers,
)


def calculate_aadforest_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])
    N = operator.inputs[0].type.shape[0]

    operator.outputs[0].type = FloatTensorType(shape=[N])


register_shape_calculator("AADForest", calculate_aadforest_output_shapes)
