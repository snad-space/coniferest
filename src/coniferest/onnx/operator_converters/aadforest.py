from onnxconverter_common.registration import register_converter

from .coniferest import add_leaf, add_node, get_default_attribute_pairs


def add_tree_to_attribute_pairs(attr_pairs, tree_id, tree, leaf_offset, evaluator):
    left = tree.left
    feature = tree.feature
    value = tree.value

    for node_id in range(tree.n_nodes):
        if left[node_id] > 0:
            mode = "BRANCH_LEQ"
            feat_id = int(feature[node_id])
            threshold = value[node_id]
            left_child_id = int(left[node_id])
            right_child_id = left_child_id + 1

            add_node(attr_pairs, tree_id, node_id, feat_id, mode, threshold, left_child_id, right_child_id)
        else:
            mode = "LEAF"
            # For leaves the `feature` array stores the in-tree leaf index
            leaf_id = leaf_offset + int(feature[node_id])
            # evaluator.leaf_values are mapped with AADForest.map_value
            weight = evaluator.weights[leaf_id] * evaluator.leaf_values[leaf_id]

            add_leaf(attr_pairs, tree_id, node_id, mode, weight)


def convert_aadforest(scope, operator, container):
    model = operator.raw_operator
    evaluator = model.evaluator

    attr_pairs = get_default_attribute_pairs()

    leaf_offset = 0
    for tree_id, tree in enumerate(evaluator.core_forest):
        add_tree_to_attribute_pairs(attr_pairs, tree_id, tree, leaf_offset, evaluator)
        leaf_offset += tree.n_leaves

    container.add_node(
        "TreeEnsembleRegressor",
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("TreeEnsembleRegressor"),
        **attr_pairs,
    )


register_converter("AADForest", convert_aadforest)
