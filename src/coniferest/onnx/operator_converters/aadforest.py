from onnx import TensorProto
from onnx.helper import make_tensor
from onnxconverter_common.registration import register_converter

from .coniferest import add_leaf, add_node, get_default_attribute_pairs


def add_tree_to_attribute_pairs(attr_pairs, tree, leaf_offset, evaluator):
    left = tree.left
    feature = tree.feature
    value = tree.value

    def process_node(node_id):
        if left[node_id] <= 0:
            leaf_id = leaf_offset + int(feature[node_id])
            weight = evaluator.weights[leaf_id] * evaluator.leaf_values[leaf_id]
            add_leaf(attr_pairs, weight)
            return len(attr_pairs["leaf_weights"]) - 1, True

        feat_id = int(feature[node_id])
        threshold = value[node_id]
        left_child_id = int(left[node_id])
        right_child_id = left_child_id + 1

        this_index = len(attr_pairs["nodes_featureids"])
        add_node(attr_pairs, feat_id, threshold, 0, False, 0, False)

        left_idx, left_is_leaf = process_node(left_child_id)
        right_idx, right_is_leaf = process_node(right_child_id)

        attr_pairs["nodes_truenodeids"][this_index] = left_idx
        attr_pairs["nodes_trueleafs"][this_index] = int(left_is_leaf)
        attr_pairs["nodes_falsenodeids"][this_index] = right_idx
        attr_pairs["nodes_falseleafs"][this_index] = int(right_is_leaf)

        return this_index, False

    attr_pairs["tree_roots"].append(len(attr_pairs["nodes_featureids"]))
    process_node(0)


def convert_aadforest(scope, operator, container):
    model = operator.raw_operator
    evaluator = model.evaluator

    attr_pairs = get_default_attribute_pairs()
    attr_pairs["aggregate_function"] = 1  # SUM

    leaf_offset = 0
    for tree in evaluator.trees:
        add_tree_to_attribute_pairs(attr_pairs, tree, leaf_offset, evaluator)
        leaf_offset += tree.n_leaves

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

    container.add_node(
        "TreeEnsemble",
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        op_version=5,
        name=scope.get_unique_operator_name("TreeEnsemble"),
        **attr_pairs,
    )


register_converter("AADForest", convert_aadforest)
