import numpy as np
from onnxconverter_common.data_types import FloatTensorType
from onnxconverter_common.registration import register_converter

from coniferest.utils import average_path_length

from .coniferest import add_leaf, add_node, get_default_attribute_pairs


def get_leaf_weight(selector, evaluator):
    weight = -np.log(2.0) / average_path_length(evaluator.samples)
    value = selector["value"]

    return weight * value


def add_tree_to_attribute_pairs(attr_pairs, tree_id, evaluator):
    node_offset = evaluator.node_offsets[tree_id]
    node_end = evaluator.node_offsets[tree_id + 1]
    tree_selectors = evaluator.selectors[node_offset:node_end]

    for node_id, selector in enumerate(tree_selectors):
        if selector["feature"] >= 0:
            mode = "BRANCH_LEQ"
            feat_id = int(selector["feature"])
            threshold = selector["value"]
            left_child_id = int(selector["left"])
            right_child_id = int(selector["right"])

            add_node(attr_pairs, tree_id, node_id, feat_id, mode, threshold, left_child_id, right_child_id)
        else:
            mode = "LEAF"
            weight = get_leaf_weight(selector, evaluator)

            add_leaf(attr_pairs, tree_id, node_id, mode, weight)


def convert_isoforest(scope, operator, container):
    model = operator.raw_operator
    evaluator = model.evaluator

    attr_pairs = get_default_attribute_pairs()
    attr_pairs["aggregate_function"] = "AVERAGE"

    for tree_id in range(evaluator.n_trees):
        add_tree_to_attribute_pairs(attr_pairs, tree_id, evaluator)

    # Declare intermediate variable for the tree ensemble output
    leaf_avg_var = scope.declare_local_variable("leaf_avg", FloatTensorType())

    # Add TreeEnsembleRegressor to compute AVERAGE of normalized leaf depths
    container.add_node(
        "TreeEnsembleRegressor",
        operator.input_full_names,
        [leaf_avg_var.full_name],
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("TreeEnsembleRegressor"),
        **attr_pairs,
    )

    # Now compute the score:
    # score = -2 ** (-leaf_avg / average_path_length(n_subsamples))
    # Rewritten as: score = -exp(-leaf_avg * ln(2) / average_path_length(n_subsamples))

    exp_var = scope.declare_local_variable("exp", FloatTensorType())
    container.add_node(
        "Exp",
        [leaf_avg_var.full_name],
        [exp_var.full_name],
        op_domain="",
        name=scope.get_unique_operator_name("Exp"),
    )

    # -(exp(leaf_sum * ln(2)))
    container.add_node(
        "Neg",
        [exp_var.full_name],
        operator.output_full_names,
        op_domain="",
        name=scope.get_unique_operator_name("Neg"),
    )


register_converter("IsolationForest", convert_isoforest)
register_converter("PineForest", convert_isoforest)
