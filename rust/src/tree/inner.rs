use crate::float::Float;
use crate::tree::node::{Leaf, Node, SplitNode};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use std::num::NonZeroU32;
use std::sync::Arc;

/// A tree built on either f32 or f64 data.
#[derive(Clone)]
pub(crate) enum TreeVariant {
    F32(Arc<TreeInner<f32>>),
    F64(Arc<TreeInner<f64>>),
}

/// Decision tree of an isolation forest, generic over the data dtype.
pub(crate) struct TreeInner<T> {
    nodes: Vec<Node<T>>,
    /// Sidecar array: average path length for the number of samples in
    /// each node, used for feature signature/importance computations.
    node_average_path_length: Vec<f32>,
    n_leaves: u32,
    n_subsamples: usize,
}

impl<T> TreeInner<T> {
    pub(super) fn new(
        nodes: Vec<Node<T>>,
        node_average_path_length: Vec<f32>,
        n_leaves: u32,
        n_subsamples: usize,
    ) -> Self {
        Self {
            nodes,
            node_average_path_length,
            n_leaves,
            n_subsamples,
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

    /// Largest split feature index used by this tree, if it has any splits.
    pub(crate) fn max_split_feature(&self) -> Option<u32> {
        self.nodes
            .iter()
            .filter_map(|node| match node {
                Node::Split(split) => Some(split.split_feature),
                Node::Leaf(_) => None,
            })
            .max()
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
}

impl<T> TreeInner<T>
where
    T: Float,
{
    /// Build a tree from per-node arrays, validating the invariants
    /// required for the unchecked traversal.
    pub(super) fn from_arrays(
        left: Vec<u32>,
        feature: Vec<u32>,
        value: Vec<T>,
        node_average_path_length: Vec<f32>,
        n_subsamples: usize,
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
                        value: value[i].to_f32().ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "failed to convert leaf value (value[{}] = {}) to f32",
                                i, value[i]
                            ))
                        })?,
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
        })
    }
}
