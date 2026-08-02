//! What a forest may be indexed with

use numpy::{AllowTypeChange, PyArrayLike1, TypeMustMatch};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PySlice, PySliceMethods};

/// Anything a forest may be indexed with, in numpy's order of preference: the
/// boolean mask is matched before the integer array so that a mask is not
/// silently cast to indices of zeros and ones.
#[derive(FromPyObject)]
pub(super) enum InputForestIndex<'py> {
    One(isize),
    Slice(Bound<'py, PySlice>),
    Mask(PyArrayLike1<'py, bool, TypeMustMatch>),
    Many(PyArrayLike1<'py, i64, AllowTypeChange>),
}

impl<'py> InputForestIndex<'py> {
    /// Bind this index to a forest of `len` trees, so it can resolve itself.
    pub(super) fn resolve(self, len: usize) -> ForestIndex<'py> {
        ForestIndex { index: self, len }
    }
}

/// A [InputForestIndex] that knows how many trees it indexes into, and so can turn
/// itself into tree positions.
pub(super) struct ForestIndex<'py> {
    index: InputForestIndex<'py>,
    len: usize,
}

impl ForestIndex<'_> {
    /// The one tree this selects, or `None` if it selects a whole forest.
    pub(super) fn single(&self) -> Option<PyResult<usize>> {
        match self.index {
            InputForestIndex::One(index) => Some(self.normalize(index)),
            _ => None,
        }
    }

    /// The tree positions this selects, resolved lazily.
    pub(super) fn iter(&self) -> Box<dyn Iterator<Item = PyResult<usize>> + '_> {
        match &self.index {
            InputForestIndex::One(index) => Box::new(std::iter::once(self.normalize(*index))),
            InputForestIndex::Slice(slice) => match slice.indices(self.len as isize) {
                Ok(slice) => Box::new(
                    (0..slice.slicelength)
                        .map(move |step| Ok((slice.start + step as isize * slice.step) as usize)),
                ),
                Err(error) => Box::new(std::iter::once(Err(error))),
            },
            InputForestIndex::Mask(mask) => {
                let mask = mask.as_array();
                if mask.len() != self.len {
                    return Box::new(std::iter::once(Err(PyIndexError::new_err(format!(
                        "boolean index has {} elements, but the forest has {} trees",
                        mask.len(),
                        self.len,
                    )))));
                }
                Box::new(
                    mask.into_iter()
                        .enumerate()
                        .filter_map(|(index, &keep)| keep.then_some(Ok(index))),
                )
            }
            InputForestIndex::Many(indices) => Box::new(
                indices
                    .as_array()
                    .into_iter()
                    .map(move |&index| self.normalize(index as isize)),
            ),
        }
    }

    /// Resolve a possibly negative index, as Python does.
    fn normalize(&self, index: isize) -> PyResult<usize> {
        let normalized = if index < 0 {
            index + self.len as isize
        } else {
            index
        };
        usize::try_from(normalized)
            .ok()
            .filter(|&index| index < self.len)
            .ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "forest tree index {index} is out of range for a forest of {} trees",
                    self.len,
                ))
            })
    }
}
