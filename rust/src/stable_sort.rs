use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use ordered_float::OrderedFloat;
use pyo3::{Bound, PyResult, Python, pyfunction};
use std::cmp;
use std::collections::BinaryHeap;

#[pyfunction]
#[pyo3(signature = (arr, pos))]
pub(crate) fn argpartial_sort<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<f64>,
    pos: usize,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let capacity = cmp::min(pos, arr.len());
    if capacity == 0 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }

    let slice = arr.as_slice()?;
    let v = Vec::from_iter(
        slice
            .iter()
            .enumerate()
            .map(|(i, x)| (OrderedFloat(*x), i))
            .take(capacity),
    );
    let mut heap = BinaryHeap::from(v);

    for (i, value) in slice.iter().enumerate().skip(capacity) {
        let cur = (OrderedFloat(*value), i);
        let mut val = heap.peek_mut().unwrap();

        if *val > cur {
            *val = cur;
        }
    }

    let vec = heap.into_sorted_vec();
    Ok(PyArray1::from_iter(py, vec.into_iter().map(|(_, i)| i)))
}
