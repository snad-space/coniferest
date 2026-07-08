use num_traits::AsPrimitive;
use numpy::{PyArray1, PyArrayDescr, PyArrayMethods, PyReadonlyArray1};
use pyo3::PyTypeInfo;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
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

const _: () = assert!(
    size_of::<Node<f64>>() == size_of::<SplitNode<f64>>(),
    "Node<f64> must be tag-free via the left_node_index niche",
);
const _: () = assert!(
    size_of::<Node<f32>>() == size_of::<SplitNode<f32>>(),
    "Node<f32> must be tag-free via the left_node_index niche",
);

/// Decision tree of an isolation forest, generic over the data dtype.
pub(crate) struct TreeInner<T> {
    pub(crate) nodes: Vec<Node<T>>,
    /// Sidecar array: average path length for the number of samples in
    /// each node, used for feature signature/importance computations.
    pub(crate) node_average_path_length: Vec<f32>,
    pub(crate) n_leaves: u32,
    pub(crate) n_subsamples: usize,
    pub(crate) n_features: u32,
}

impl<T> TreeInner<T>
where
    T: Copy + PartialOrd,
{
    /// Follow the decision path for `sample` and return the reached leaf.
    ///
    /// Safety: relies on the invariants checked in the constructor: child
    /// indices are within the tree and greater than the parent index, and
    /// split features are less than `n_features` (`sample` length).
    #[inline]
    pub(crate) fn find_leaf(&self, sample: &[T]) -> &Leaf {
        let mut i = 0;
        loop {
            match unsafe { self.nodes.get_unchecked(i) } {
                Node::Leaf(leaf) => break leaf,
                Node::Split(split) => {
                    let left = split.left_node_index.get() as usize;
                    i = if *unsafe { sample.get_unchecked(split.split_feature as usize) }
                        <= split.split_value
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

impl<T> TreeInner<T>
where
    T: Copy + AsPrimitive<f32>,
{
    /// Build a tree from per-node arrays, validating the invariants
    /// required for the unchecked traversal.
    fn from_arrays(
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

/// A tree built on either f32 or f64 data.
pub(crate) enum TreeVariant {
    F32(TreeInner<f32>),
    F64(TreeInner<f64>),
}

/// Dispatch `$body` over the dtype variants of a [TreeVariant].
macro_rules! on_inner {
    ($variant:expr, $tree:ident => $body:expr) => {
        match $variant {
            TreeVariant::F32($tree) => $body,
            TreeVariant::F64($tree) => $body,
        }
    };
}

/// The data dtype a tree can be built on: f32 or f64.
pub(crate) trait TreeDtype: Sized {
    /// The numpy name of the dtype.
    const NAME: &'static str;

    /// Downcast the tree to this dtype, `None` on mismatch.
    fn tree_inner(tree: &Tree) -> Option<&TreeInner<Self>>;

    fn wrap(inner: TreeInner<Self>) -> TreeVariant;
}

impl TreeDtype for f32 {
    const NAME: &'static str = "float32";

    fn tree_inner(tree: &Tree) -> Option<&TreeInner<Self>> {
        match &tree.0 {
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

    fn tree_inner(tree: &Tree) -> Option<&TreeInner<Self>> {
        match &tree.0 {
            TreeVariant::F64(inner) => Some(inner),
            TreeVariant::F32(_) => None,
        }
    }

    fn wrap(inner: TreeInner<Self>) -> TreeVariant {
        TreeVariant::F64(inner)
    }
}

#[derive(FromPyObject)]
enum ValueArray<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

/// Decision tree of an isolation forest.
///
/// The tree is built on f32 or f64 data and can only be applied to data
/// of the same dtype, see the `dtype` attribute; the Python side casts
/// the data when needed.
///
/// The forest is stored as a Python list of trees. All the numpy views of
/// the tree (`left`, `feature`, `value`, `node_average_path_length`) are
/// copies: the tree itself is immutable.
#[pyclass(frozen, module = "coniferest._core")]
pub(crate) struct Tree(pub(crate) TreeVariant);

impl Tree {
    /// The numpy name of the dtype the tree was built on.
    pub(crate) fn dtype_name(&self) -> &'static str {
        match &self.0 {
            TreeVariant::F32(_) => f32::NAME,
            TreeVariant::F64(_) => f64::NAME,
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
    /// value for leaves; its dtype, f32 or f64, sets the tree dtype.
    /// Leaves are assigned sequential `leaf_index` in node order.
    #[new]
    fn new(
        left: PyReadonlyArray1<u32>,
        feature: PyReadonlyArray1<u32>,
        value: ValueArray,
        node_average_path_length: PyReadonlyArray1<f32>,
        n_subsamples: usize,
        n_features: u32,
    ) -> PyResult<Self> {
        let left = left.to_vec()?;
        let feature = feature.to_vec()?;
        let node_average_path_length = node_average_path_length.to_vec()?;

        let variant = match value {
            ValueArray::F32(value) => TreeVariant::F32(TreeInner::from_arrays(
                left,
                feature,
                value.to_vec()?,
                node_average_path_length,
                n_subsamples,
                n_features,
            )?),
            ValueArray::F64(value) => TreeVariant::F64(TreeInner::from_arrays(
                left,
                feature,
                value.to_vec()?,
                node_average_path_length,
                n_subsamples,
                n_features,
            )?),
        };
        Ok(Tree(variant))
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
        on_inner!(&self.0, tree => tree.nodes.len())
    }

    #[getter]
    fn n_leaves(&self) -> u32 {
        on_inner!(&self.0, tree => tree.n_leaves)
    }

    #[getter]
    fn n_subsamples(&self) -> usize {
        on_inner!(&self.0, tree => tree.n_subsamples)
    }

    #[getter]
    fn n_features(&self) -> u32 {
        on_inner!(&self.0, tree => tree.n_features)
    }

    /// Left child index per node, 0 for leaves.
    #[getter]
    fn left<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        on_inner!(&self.0, tree => PyArray1::from_iter(
            py,
            tree.nodes.iter().map(|node| match node {
                Node::Split(split) => split.left_node_index.get(),
                Node::Leaf(_) => 0,
            }),
        ))
    }

    /// Split feature per node, leaf index for leaves.
    #[getter]
    fn feature<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        on_inner!(&self.0, tree => PyArray1::from_iter(
            py,
            tree.nodes.iter().map(|node| match node {
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
                tree.nodes.iter().map(|node| match node {
                    Node::Split(split) => split.split_value,
                    Node::Leaf(leaf) => leaf.value,
                }),
            )
            .into_any(),
            TreeVariant::F64(tree) => PyArray1::from_iter(
                py,
                tree.nodes.iter().map(|node| match node {
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
        on_inner!(&self.0, tree => PyArray1::from_slice(py, &tree.node_average_path_length))
    }

    /// Decision values of the leaves, ordered by `leaf_index`.
    fn leaf_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        on_inner!(&self.0, tree => {
            let mut values = vec![0.0; tree.n_leaves as usize];
            for node in &tree.nodes {
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
                this.n_subsamples(),
                this.n_features(),
            ),
        ))
    }
}
