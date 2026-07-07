from onnxconverter_common.registration import register_converter

from .coniferest import add_leaf, add_node, get_default_attribute_pairs


def get_leaf_weight(selector, evaluator):
    value = selector["value"]
    # Generally, selector["leaf"] should be -1 for leafs,
    # but for AADForest it stores related weight index.
    leaf_id = selector["left"]
    weight = evaluator.weights[leaf_id]

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


def convert_aadforest(scope, operator, container):
    model = operator.raw_operator
    evaluator = model.evaluator
    n_trees = evaluator.n_trees

    attr_pairs = get_default_attribute_pairs()

    for tree_id in range(n_trees):
        add_tree_to_attribute_pairs(attr_pairs, tree_id, evaluator)

    container.add_node(
        "TreeEnsembleRegressor",
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        name=scope.get_unique_operator_name("TreeEnsembleRegressor"),
        **attr_pairs,
    )


register_converter("AADForest", convert_aadforest)
