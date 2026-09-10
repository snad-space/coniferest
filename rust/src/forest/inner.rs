//! Implementation details of the forest traversal

use crate::tree::{PyTree, TreeDtype, TreeInner};
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Borrowed trees of the forest with their global leaf offsets.
///
/// `Tree` is a frozen pyclass, so the borrows are GIL-independent and the
/// forest can be traversed in parallel even when stored as a Python list.
/// All the trees must be built on the data dtype `T`, matching the dtype
/// of the scored data, so the traversal never casts values.
pub(super) struct Forest<'a, T> {
    trees: Vec<&'a TreeInner<T>>,
    /// Global leaf index of the first leaf of each tree
    leaf_offsets: Vec<u32>,
    n_leaves: usize,
}

impl<'a, T> Forest<'a, T>
where
    T: TreeDtype,
{
    pub(super) fn new(trees: &'a [Py<PyTree>], n_features: usize) -> PyResult<Self> {
        let trees: Vec<&TreeInner<T>> = trees
            .iter()
            .map(|tree| {
                T::tree_inner(&tree.get().0).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "data dtype is {}, but the tree was built on {} data",
                        T::NAME,
                        tree.get().0.dtype_name(),
                    ))
                })
            })
            .collect::<PyResult<_>>()?;

        let mut leaf_offsets = Vec::with_capacity(trees.len());
        let mut n_leaves: u32 = 0;
        for tree in &trees {
            if tree.n_features() as usize != n_features {
                return Err(PyValueError::new_err(format!(
                    "data has {} features, but a tree was built on {} features",
                    n_features,
                    tree.n_features(),
                )));
            }
            leaf_offsets.push(n_leaves);
            n_leaves = n_leaves
                .checked_add(tree.n_leaves())
                .ok_or_else(|| PyValueError::new_err("too many leaves in the forest"))?;
        }

        Ok(Self {
            trees,
            leaf_offsets,
            n_leaves: n_leaves as usize,
        })
    }
}

impl<T> Forest<'_, T> {
    pub(super) fn trees(&self) -> &[&TreeInner<T>] {
        &self.trees
    }

    pub(super) fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&&TreeInner<T>, u32)> {
        self.trees.iter().zip(self.leaf_offsets.iter().copied())
    }
}
