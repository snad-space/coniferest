use num_traits::AsPrimitive;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::PyTypeInfo;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::num::NonZeroU32;

/// Inner node of a decision tree.
pub(crate) struct SplitNode {
    /// Index of the left subtree; `right_node_index = left_node_index + 1`.
    pub(crate) left_node_index: NonZeroU32,
    /// Feature index to branch on.
    pub(crate) split_feature: u32,
    /// Feature value to branch on: `<=` goes left, `>` goes right.
    ///
    /// The value is always stored as f64; it is exactly representable
    /// even when the tree is built on f32 data.
    pub(crate) split_value: f64,
}

/// Terminal node of a decision tree.
pub(crate) struct Leaf {
    /// Sequential index of the leaf within the tree, in node order.
    pub(crate) leaf_index: u32,
    /// Resulting decision value, the estimated path length by default.
    pub(crate) value: f64,
}

/// Decision tree node: either a split node or a leaf.
///
/// The root is stored at index 0, so no split node can reference it as
/// a child, and `left_node_index` is never zero.
pub(crate) enum Node {
    Split(SplitNode),
    Leaf(Leaf),
}

/// Decision tree of an isolation forest.
///
/// The forest is stored as a Python list of trees. All the numpy views of
/// the tree (`left`, `feature`, `value`, `node_average_path_length`) are
/// copies: the tree itself is immutable.
#[pyclass(frozen, module = "coniferest._core")]
pub(crate) struct Tree {
    pub(crate) nodes: Vec<Node>,
    /// Sidecar array: average path length for the number of samples in
    /// each node, used for feature signature/importance computations.
    pub(crate) node_average_path_length: Vec<f32>,
    pub(crate) n_leaves: u32,
    pub(crate) n_subsamples: usize,
    pub(crate) n_features: u32,
}

impl Tree {
    /// Follow the decision path for `sample` and return the reached leaf.
    ///
    /// Safety: relies on the invariants checked in the constructor: child
    /// indices are within the tree and greater than the parent index, and
    /// split features are less than `n_features` (`sample` length).
    #[inline]
    pub(crate) fn find_leaf<T>(&self, sample: &[T]) -> &Leaf
    where
        T: Copy + PartialOrd + 'static,
        f64: AsPrimitive<T>,
    {
        let mut i = 0;
        loop {
            match unsafe { self.nodes.get_unchecked(i) } {
                Node::Leaf(leaf) => break leaf,
                Node::Split(split) => {
                    let threshold: T = split.split_value.as_();
                    let left = split.left_node_index.get() as usize;
                    i = if *unsafe { sample.get_unchecked(split.split_feature as usize) }
                        <= threshold
                    {
                        left
                    } else {
                        left + 1
                    };
                }
            }
        }
    }
}

#[pymethods]
impl Tree {
    /// Build a tree from per-node arrays.
    ///
    /// `left` is the left child index, 0 marks a leaf (`right = left + 1`
    /// and is omitted). `feature` is the split feature, ignored for
    /// leaves. `value` is the split value for split nodes and the decision
    /// value for leaves. Leaves are assigned sequential `leaf_index` in
    /// node order.
    #[new]
    fn new(
        left: PyReadonlyArray1<u32>,
        feature: PyReadonlyArray1<u32>,
        value: PyReadonlyArray1<f64>,
        node_average_path_length: PyReadonlyArray1<f32>,
        n_subsamples: usize,
        n_features: u32,
    ) -> PyResult<Self> {
        let left = left.to_vec()?;
        let feature = feature.to_vec()?;
        let value = value.to_vec()?;
        let node_average_path_length = node_average_path_length.to_vec()?;

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
                        value: value[i],
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

        Ok(Tree {
            nodes,
            node_average_path_length,
            n_leaves,
            n_subsamples,
            n_features,
        })
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[getter]
    fn n_leaves(&self) -> u32 {
        self.n_leaves
    }

    #[getter]
    fn n_subsamples(&self) -> usize {
        self.n_subsamples
    }

    #[getter]
    fn n_features(&self) -> u32 {
        self.n_features
    }

    /// Left child index per node, 0 for leaves.
    #[getter]
    fn left<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_iter(
            py,
            self.nodes.iter().map(|node| match node {
                Node::Split(split) => split.left_node_index.get(),
                Node::Leaf(_) => 0,
            }),
        )
    }

    /// Split feature per node, leaf index for leaves.
    #[getter]
    fn feature<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_iter(
            py,
            self.nodes.iter().map(|node| match node {
                Node::Split(split) => split.split_feature,
                Node::Leaf(leaf) => leaf.leaf_index,
            }),
        )
    }

    /// Split value per node, decision value for leaves.
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_iter(
            py,
            self.nodes.iter().map(|node| match node {
                Node::Split(split) => split.split_value,
                Node::Leaf(leaf) => leaf.value,
            }),
        )
    }

    /// Average path length per node (sidecar array).
    #[getter(node_average_path_length)]
    fn node_average_path_length_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.node_average_path_length)
    }

    /// Decision values of the leaves, ordered by `leaf_index`.
    fn leaf_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let mut values = vec![0.0; self.n_leaves as usize];
        for node in &self.nodes {
            if let Node::Leaf(leaf) = node {
                values[leaf.leaf_index as usize] = leaf.value;
            }
        }
        PyArray1::from_vec(py, values)
    }

    /// Pickle support.
    #[allow(clippy::type_complexity)]
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(
        Bound<'py, PyType>,
        (
            Bound<'py, PyArray1<u32>>,
            Bound<'py, PyArray1<u32>>,
            Bound<'py, PyArray1<f64>>,
            Bound<'py, PyArray1<f32>>,
            usize,
            u32,
        ),
    )> {
        let py = slf.py();
        let this = slf.get();
        Ok((
            Self::type_object(py),
            (
                this.left(py),
                this.feature(py),
                this.value(py),
                this.node_average_path_length_py(py),
                this.n_subsamples,
                this.n_features,
            ),
        ))
    }
}
