def get_default_attribute_pairs():
    attrs = {}
    attrs["aggregate_function"] = 0
    attrs["n_targets"] = 1
    attrs["tree_roots"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_splits"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_trueleafs"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_falseleafs"] = []
    attrs["leaf_targetids"] = []
    attrs["leaf_weights"] = []
    return attrs


def add_node(
    attr_pairs,
    feature_id,
    threshold,
    true_child_id,
    true_is_leaf,
    false_child_id,
    false_is_leaf,
):
    attr_pairs["nodes_featureids"].append(feature_id)
    attr_pairs["nodes_modes"].append(0)
    attr_pairs["nodes_splits"].append(threshold)
    attr_pairs["nodes_truenodeids"].append(true_child_id)
    attr_pairs["nodes_trueleafs"].append(int(true_is_leaf))
    attr_pairs["nodes_falsenodeids"].append(false_child_id)
    attr_pairs["nodes_falseleafs"].append(int(false_is_leaf))


def add_leaf(attr_pairs, weight):
    attr_pairs["leaf_targetids"].append(0)
    attr_pairs["leaf_weights"].append(weight)
