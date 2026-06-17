use numpy::{PyArray1, PyReadonlyArray1};
use ordered_float::OrderedFloat;
use pyo3::{Bound, PyResult, Python, pyfunction};
use std::collections::BinaryHeap;

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
        heap.push((OrderedFloat(*value), i));

        if heap.len() > pos {
            heap.pop();
        }
    }

    let vec = heap.into_sorted_vec();
    Ok(PyArray1::from_iter(py, vec.into_iter().map(|(_, i)| i)))
}
