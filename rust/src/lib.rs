mod mut_slices;
mod selector;
mod tree_traversal;

use crate::selector::Selector;
use crate::tree_traversal::{
    calc_apply, calc_feature_delta_sum, calc_paths_sum, calc_paths_sum_transpose,
};
use pyo3::prelude::*;

#[pymodule]
fn calc_trees(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("selector_dtype", Selector::dtype(py)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(calc_feature_delta_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_apply, m)?)?;
    Ok(())
}
