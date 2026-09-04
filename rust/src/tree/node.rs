use std::num::NonZeroU32;

/// Inner node of a decision tree.
///
/// `T` is the dtype of the training data, f32 or f64.
pub(crate) struct SplitNode<T> {
    /// Index of the left subtree; `right_node_index = left_node_index + 1`.
    pub(crate) left_node_index: NonZeroU32,
    /// Feature index to branch on.
    pub(crate) split_feature: u32,
    /// Feature value to branch on: `<=` goes left, `>` goes right.
    pub(crate) split_value: T,
}

/// Terminal node of a decision tree.
///
/// The struct is 8 bytes, so it fits [Node] outside the niche of
/// [SplitNode::left_node_index], and the enum needs no explicit tag:
/// [Node] is the same size as [SplitNode].
pub(crate) struct Leaf {
    /// Sequential index of the leaf within the tree, in node order.
    pub(crate) leaf_index: u32,
    /// Resulting decision value, the estimated path length by default.
    pub(crate) value: f32,
}

/// Decision tree node: either a split node or a leaf.
///
/// The root is stored at index 0, so no split node can reference it as
/// a child, and `left_node_index` is never zero. Its niche serves as the
/// enum discriminant, which is checked by the assertions below.
pub(crate) enum Node<T> {
    Split(SplitNode<T>),
    Leaf(Leaf),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_size() {
        assert_eq!(size_of::<Node<f32>>(), size_of::<SplitNode<f32>>());
        assert_eq!(size_of::<Node<f64>>(), size_of::<SplitNode<f64>>());
    }
}
