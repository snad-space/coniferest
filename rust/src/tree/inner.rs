use crate::float::Float;
use crate::tree::node::{Leaf, Node, SplitNode};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use std::num::NonZeroU32;

/// A tree built on either f32 or f64 data.
pub(crate) enum TreeVariant {
    F32(TreeInner<f32>),
    F64(TreeInner<f64>),
}

impl TreeVariant {
    /// The numpy name of the dtype the tree was built on.
    pub(crate) fn dtype_name(&self) -> &'static str {
        match self {
            TreeVariant::F32(_) => f32::NAME,
            TreeVariant::F64(_) => f64::NAME,
        }
    }
}

/// The data dtype a tree can be built on: f32 or f64.
pub(crate) trait TreeDtype: Float {
    /// The numpy name of the dtype.
    const NAME: &'static str;

    /// Downcast the tree to this dtype, `None` on mismatch.
    fn tree_inner(variant: &TreeVariant) -> Option<&TreeInner<Self>>;

    fn wrap(inner: TreeInner<Self>) -> TreeVariant;
}

impl TreeDtype for f32 {
    const NAME: &'static str = "float32";

    fn tree_inner(variant: &TreeVariant) -> Option<&TreeInner<Self>> {
        match variant {
            TreeVariant::F32(inner) => Some(inner),
            TreeVariant::F64(_) => None,
        }
    }

    fn wrap(inner: TreeInner<Self>) -> TreeVariant {
        TreeVariant::F32(inner)
    }
}

impl TreeDtype for f64 {
    const NAME: &'static str = "float64";

    fn tree_inner(variant: &TreeVariant) -> Option<&TreeInner<Self>> {
        match variant {
            TreeVariant::F64(inner) => Some(inner),
            TreeVariant::F32(_) => None,
        }
    }

    fn wrap(inner: TreeInner<Self>) -> TreeVariant {
        TreeVariant::F64(inner)
    }
}

/// Decision tree of an isolation forest, generic over the data dtype.
pub(crate) struct TreeInner<T> {
    nodes: Vec<Node<T>>,
    /// Sidecar array: average path length for the number of samples in
    /// each node, used for feature signature/importance computations.
    node_average_path_length: Vec<f32>,
    n_leaves: u32,
    n_subsamples: usize,
    n_features: u32,
}

impl<T> TreeInner<T> {
    pub(super) fn new(
        nodes: Vec<Node<T>>,
        node_average_path_length: Vec<f32>,
        n_leaves: u32,
        n_subsamples: usize,
        n_features: u32,
    ) -> Self {
        Self {
            nodes,
            node_average_path_length,
            n_leaves,
            n_subsamples,
            n_features,
        }
    }

    pub(crate) fn nodes(&self) -> &[Node<T>] {
        &self.nodes
    }

    pub(crate) fn node_average_path_length(&self) -> &[f32] {
        &self.node_average_path_length
    }

    pub(crate) fn n_leaves(&self) -> u32 {
        self.n_leaves
    }

    pub(crate) fn n_subsamples(&self) -> usize {
        self.n_subsamples
    }

    pub(crate) fn n_features(&self) -> u32 {
        self.n_features
    }
}

impl<T> TreeInner<T>
where
    T: Float,
{
    /// Follow the decision path for `sample` and return the reached leaf.
    ///
    /// Safety: relies on the invariants checked in the constructor: child
    /// indices are within the tree and greater than the parent index, and
    /// split features are less than `n_features` (`sample` length).
    #[inline]
    pub(crate) fn find_leaf(&self, sample: &[T]) -> &Leaf {
        self.for_each_split(sample, |_, _, _| {})
    }

    /// Follow the decision path for `sample`, calling `visit` at every split
    /// node visited, and return the reached leaf.
    ///
    /// `visit(node_index, split, child_index)` is called for every split
    /// node visited, where:
    /// - `node_index: usize` is the index of the visited split node;
    /// - `split: &SplitNode<T>` is the split node itself (feature and
    ///   threshold);
    /// - `child_index: usize` is the index of the child chosen for
    ///   `sample`, i.e. the node `for_each_split` will visit or return next.
    ///
    /// Safety: relies on the invariants checked in the constructor: child
    /// indices are within the tree and greater than the parent index, and
    /// split features are less than `n_features` (`sample` length).
    #[inline]
    pub(crate) fn for_each_split(
        &self,
        sample: &[T],
        mut visit: impl FnMut(usize, &SplitNode<T>, usize),
    ) -> &Leaf {
        let mut node_index = 0;
        loop {
            match unsafe { self.nodes.get_unchecked(node_index) } {
                Node::Leaf(leaf) => break leaf,
                Node::Split(split) => {
                    let left = split.left_node_index.get() as usize;
                    let value = *unsafe { sample.get_unchecked(split.split_feature as usize) };
                    let child_index = left + (value > split.split_value) as usize;
                    visit(node_index, split, child_index);
                    node_index = child_index;
                }
            }
        }
    }

    /// Build a tree from per-node arrays, validating the invariants
    /// required for the unchecked traversal.
    pub(super) fn from_arrays(
        left: Vec<u32>,
        feature: Vec<u32>,
        value: Vec<T>,
        node_average_path_length: Vec<f32>,
        n_subsamples: usize,
        n_features: u32,
    ) -> PyResult<Self> {
        let n_nodes = left.len();
        if n_nodes == 0 {
            return Err(PyValueError::new_err("tree must have at least one node"));
        }
        if feature.len() != n_nodes
            || value.len() != n_nodes
            || node_average_path_length.len() != n_nodes
        {
            return Err(PyValueError::new_err(
                "left, feature, value and node_average_path_length must have the same length",
            ));
        }

        let mut nodes = Vec::with_capacity(n_nodes);
        let mut n_leaves: u32 = 0;
        for i in 0..n_nodes {
            match NonZeroU32::new(left[i]) {
                None => {
                    nodes.push(Node::Leaf(Leaf {
                        leaf_index: n_leaves,
                        value: value[i].as_(),
                    }));
                    n_leaves += 1;
                }
                Some(left_node_index) => {
                    let left_usize = left_node_index.get() as usize;
                    // Children must go after the parent: it guarantees that
                    // the tree traversal is safe and finite
                    if left_usize <= i || left_usize + 1 >= n_nodes {
                        return Err(PyValueError::new_err(
                            "left child index must be greater than the node index, \
                             and the right child (left + 1) must be within the tree",
                        ));
                    }
                    if feature[i] >= n_features {
                        return Err(PyValueError::new_err(
                            "split feature must be less than n_features",
                        ));
                    }
                    nodes.push(Node::Split(SplitNode {
                        left_node_index,
                        split_feature: feature[i],
                        split_value: value[i],
                    }));
                }
            }
        }

        Ok(TreeInner {
            nodes,
            node_average_path_length,
            n_leaves,
            n_subsamples,
            n_features,
        })
    }
}
