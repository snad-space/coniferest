def get_default_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["n_targets"] = 1
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["target_treeids"] = []
    attrs["target_nodeids"] = []
    attrs["target_ids"] = []
    attrs["target_weights"] = []
    return attrs


def add_node(attr_pairs, tree_id, node_id, feature_id, mode, value, true_child_id, false_child_id):
    attr_pairs["nodes_treeids"].append(tree_id)
    attr_pairs["nodes_nodeids"].append(node_id)
    attr_pairs["nodes_featureids"].append(feature_id)
    attr_pairs["nodes_modes"].append(mode)
    attr_pairs["nodes_values"].append(value)
    attr_pairs["nodes_truenodeids"].append(true_child_id)
    attr_pairs["nodes_falsenodeids"].append(false_child_id)


def add_leaf(attr_pairs, tree_id, node_id, mode, weight):
    add_node(attr_pairs, tree_id, node_id, 0, mode, 0.0, 0, 0)

    attr_pairs["target_treeids"].append(tree_id)
    attr_pairs["target_nodeids"].append(node_id)
    attr_pairs["target_ids"].append(0)
    attr_pairs["target_weights"].append(weight)
