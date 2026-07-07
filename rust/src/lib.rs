mod build;
mod stable_sort;
mod tree;
mod tree_traversal;
mod utils;

use crate::build::build_trees;
use crate::stable_sort::argpartial_sort;
use crate::tree::Tree;
use crate::tree_traversal::{calc_apply, calc_feature_delta_sum, calc_paths_sum};
use crate::utils::average_path_length_py;
use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Tree>()?;
    m.add_function(wrap_pyfunction!(average_path_length_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_trees, m)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_feature_delta_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_apply, m)?)?;
    m.add_function(wrap_pyfunction!(argpartial_sort, m)?)?;
    Ok(())
}
