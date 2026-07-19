use crate::tree::inner::{TreeInner, TreeVariant};
use crate::tree::node::Node;
use numpy::prelude::*;
use numpy::{PyArray1, PyArrayDescr, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{Bound, PyAny, PyResult, PyTypeInfo, Python};
use std::sync::Arc;

/// Decision tree of an isolation forest.
///
/// The tree is built on f32 or f64 data and can only be applied to data
/// of the same dtype, see the `dtype` attribute; the Python side casts
/// the data when needed.
///
/// The forest is stored as a Python list of trees. All the numpy views of
/// the tree (`left`, `feature`, `value`, `node_average_path_length`) are
/// copies: the tree itself is immutable.
#[derive(Clone)]
#[pyclass(name = "Tree", frozen, module = "coniferest._core", from_py_object)]
pub(crate) struct PyTree(pub(crate) TreeVariant);

/// Dispatch `$body` over the dtype variants of a [TreeVariant].
macro_rules! on_tree_inner {
    ($variant:expr, $tree:ident => $body:expr) => {
        match $variant {
            TreeVariant::F32($tree) => $body,
            TreeVariant::F64($tree) => $body,
        }
    };
}

#[derive(FromPyObject)]
enum ValueArray<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

impl From<Arc<TreeInner<f32>>> for PyTree {
    fn from(inner: Arc<TreeInner<f32>>) -> Self {
        PyTree(TreeVariant::F32(inner))
    }
}

impl From<Arc<TreeInner<f64>>> for PyTree {
    fn from(inner: Arc<TreeInner<f64>>) -> Self {
        PyTree(TreeVariant::F64(inner))
    }
}

#[pymethods]
impl PyTree {
    /// Build a tree from per-node arrays.
    ///
    /// `left` is the left child index, 0 marks a leaf (`right = left + 1`
    /// and is omitted). `feature` is the split feature, ignored for
    /// leaves. `value` is the split value for split nodes and the decision
    /// value for leaves; its dtype, f32 or f64, sets the tree dtype.
    /// Leaves are assigned sequential `leaf_index` in node order.
    #[new]
    fn new(
        left: PyReadonlyArray1<u32>,
        feature: PyReadonlyArray1<u32>,
        value: ValueArray,
        node_average_path_length: PyReadonlyArray1<f32>,
        n_subsamples: usize,
    ) -> PyResult<Self> {
        let left = left.to_vec()?;
        let feature = feature.to_vec()?;
        let node_average_path_length = node_average_path_length.to_vec()?;

        let variant = match value {
            ValueArray::F32(value) => TreeVariant::F32(
                TreeInner::from_arrays(
                    left,
                    feature,
                    value.to_vec()?,
                    node_average_path_length,
                    n_subsamples,
                )?
                .into(),
            ),
            ValueArray::F64(value) => TreeVariant::F64(
                TreeInner::from_arrays(
                    left,
                    feature,
                    value.to_vec()?,
                    node_average_path_length,
                    n_subsamples,
                )?
                .into(),
            ),
        };
        Ok(PyTree(variant))
    }

    /// Data dtype the tree was built on, np.float32 or np.float64.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        match &self.0 {
            TreeVariant::F32(_) => numpy::dtype::<f32>(py),
            TreeVariant::F64(_) => numpy::dtype::<f64>(py),
        }
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        on_tree_inner!(&self.0, tree => tree.nodes().len())
    }

    #[getter]
    fn n_leaves(&self) -> u32 {
        on_tree_inner!(&self.0, tree => tree.n_leaves())
    }

    #[getter]
    fn n_subsamples(&self) -> usize {
        on_tree_inner!(&self.0, tree => tree.n_subsamples())
    }

    /// Left child index per node, 0 for leaves.
    #[getter]
    fn left<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        on_tree_inner!(&self.0, tree => PyArray1::from_iter(
            py,
            tree.nodes().iter().map(|node| match node {
                Node::Split(split) => split.left_node_index.get(),
                Node::Leaf(_) => 0,
            }),
        ))
    }

    /// Split feature per node, leaf index for leaves.
    #[getter]
    fn feature<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        on_tree_inner!(&self.0, tree => PyArray1::from_iter(
            py,
            tree.nodes().iter().map(|node| match node {
                Node::Split(split) => split.split_feature,
                Node::Leaf(leaf) => leaf.leaf_index,
            }),
        ))
    }

    /// Split value per node, decision value for leaves.
    /// The array is of the tree dtype.
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match &self.0 {
            TreeVariant::F32(tree) => PyArray1::from_iter(
                py,
                tree.nodes().iter().map(|node| match node {
                    Node::Split(split) => split.split_value,
                    Node::Leaf(leaf) => leaf.value,
                }),
            )
            .into_any(),
            TreeVariant::F64(tree) => PyArray1::from_iter(
                py,
                tree.nodes().iter().map(|node| match node {
                    Node::Split(split) => split.split_value,
                    Node::Leaf(leaf) => leaf.value as f64,
                }),
            )
            .into_any(),
        }
    }

    /// Average path length per node (sidecar array).
    #[getter(node_average_path_length)]
    fn node_average_path_length_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        on_tree_inner!(&self.0, tree => PyArray1::from_slice(py, tree.node_average_path_length()))
    }

    /// Decision values of the leaves, ordered by `leaf_index`.
    fn leaf_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        on_tree_inner!(&self.0, tree => {
            let mut values = vec![0.0; tree.n_leaves() as usize];
            for node in tree.nodes() {
                if let Node::Leaf(leaf) = node {
                    values[leaf.leaf_index as usize] = leaf.value as f64;
                }
            }
            PyArray1::from_vec(py, values)
        })
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
            Bound<'py, PyAny>,
            Bound<'py, PyArray1<f32>>,
            usize,
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
                this.n_subsamples(),
            ),
        ))
    }
}
