use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use ordered_float::OrderedFloat;
use pyo3::{Bound, PyResult, Python, pyfunction};
use std::cmp;

fn sort_key(arr: &[f64], x: &usize) -> (OrderedFloat<f64>, usize) {
    (OrderedFloat(arr[*x]), *x)
}

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
    let len = slice.len();

    // Create a temporary vector of indices from 0 to len-1
    let mut indices: Vec<usize> = (0..len).collect();
    let head = if capacity < len {
        // Partition so that the first `capacity` elements are the smallest
        indices.select_nth_unstable_by_key(capacity, |x| sort_key(slice, x));

        &mut indices[..capacity]
    } else {
        &mut indices[..]
    };

    head.sort_by_key(|x| sort_key(slice, x));

    // Copy the first `capacity` elements into the result
    Ok(PyArray1::from_vec(py, indices[..capacity].to_vec()))
}
