use numpy::PyUntypedArrayMethods;
use numpy::{PyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::{Bound, PyResult, Python, pyfunction};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A wrapper around `f64` that implements `Ord` via `total_cmp`,
/// panicking on NaN values.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[pyfunction]
#[pyo3(signature = (arr, pos))]
pub(crate) fn argpartial_sort<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<f64>,
    pos: usize,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let mut heap = BinaryHeap::new();
    heap.reserve(pos);

    let slice = arr.as_slice()?;
    for (i, value) in slice.iter().enumerate() {
        heap.push((OrderedF64(*value), i));

        if heap.len() > pos {
            heap.pop();
        }
    }

    let vec = heap.into_sorted_vec();
    Ok(PyArray1::from_iter(py, vec.into_iter().map(|(_, i)| i)))
}
