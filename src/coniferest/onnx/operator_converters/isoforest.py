import numpy as np
from onnx import TensorProto
from onnx.helper import make_tensor
from onnxconverter_common.data_types import FloatTensorType
from onnxconverter_common.registration import register_converter

from coniferest.utils import average_path_length

from .coniferest import add_leaf, add_node, get_default_attribute_pairs


def add_tree_to_attribute_pairs(attr_pairs, tree, evaluator):
    left = tree.left
    feature = tree.feature
    value = tree.value

    leaf_weight = -np.log(2.0) / average_path_length(evaluator.samples)

    node_index_map = {}

    def process_node(node_id):
        """Recursively add a node/leaf and return its index in attr_pairs."""
        if left[node_id] <= 0:
            # It's a leaf
            weight = leaf_weight * value[node_id]
            add_leaf(attr_pairs, weight)
            return len(attr_pairs["leaf_weights"]) - 1, True

        # It's a branch
        feat_id = int(feature[node_id])
        threshold = value[node_id]
        left_child_id = int(left[node_id])
        right_child_id = left_child_id + 1

        this_index = len(attr_pairs["nodes_featureids"])
        # Reserve a slot first (placeholder) to know our own index
        add_node(attr_pairs, feat_id, threshold, 0, False, 0, False)

        left_idx, left_is_leaf = process_node(left_child_id)
        right_idx, right_is_leaf = process_node(right_child_id)

        # Now fill in the correct child indices
        attr_pairs["nodes_truenodeids"][this_index] = left_idx
        attr_pairs["nodes_trueleafs"][this_index] = int(left_is_leaf)
        attr_pairs["nodes_falsenodeids"][this_index] = right_idx
        attr_pairs["nodes_falseleafs"][this_index] = int(right_is_leaf)

        return this_index, False

    attr_pairs["tree_roots"].append(len(attr_pairs["nodes_featureids"]))
    process_node(0)


def convert_isoforest(scope, operator, container):
    model = operator.raw_operator
    evaluator = model.evaluator

    attr_pairs = get_default_attribute_pairs()

    for tree in evaluator.trees:
        add_tree_to_attribute_pairs(attr_pairs, tree, evaluator)

    # Convert list-based attributes to ONNX tensors as required by TreeEnsemble
    attr_pairs["leaf_weights"] = make_tensor(
        "leaf_weights",
        TensorProto.FLOAT,
        (len(attr_pairs["leaf_weights"]),),
        attr_pairs["leaf_weights"],
    )
    attr_pairs["nodes_splits"] = make_tensor(
        "nodes_splits",
        TensorProto.FLOAT,
        (len(attr_pairs["nodes_splits"]),),
        attr_pairs["nodes_splits"],
    )
    attr_pairs["nodes_modes"] = make_tensor(
        "nodes_modes",
        TensorProto.UINT8,
        (len(attr_pairs["nodes_modes"]),),
        attr_pairs["nodes_modes"],
    )

    leaf_avg_var = scope.declare_local_variable("leaf_avg", FloatTensorType())

    container.add_node(
        "TreeEnsemble",
        operator.input_full_names,
        [leaf_avg_var.full_name],
        op_domain="ai.onnx.ml",
        op_version=5,
        name=scope.get_unique_operator_name("TreeEnsemble"),
        **attr_pairs,
    )

    exp_var = scope.declare_local_variable("exp", FloatTensorType())
    container.add_node(
        "Exp",
        [leaf_avg_var.full_name],
        [exp_var.full_name],
        op_domain="",
        name=scope.get_unique_operator_name("Exp"),
    )

    container.add_node(
        "Neg",
        [exp_var.full_name],
        operator.output_full_names,
        op_domain="",
        name=scope.get_unique_operator_name("Neg"),
    )


register_converter("IsolationForest", convert_isoforest)
register_converter("PineForest", convert_isoforest)
