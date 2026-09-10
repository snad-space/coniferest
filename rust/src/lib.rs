// `#[bench]` (used by the `bench_alloc` module in tree/builder.rs) is nightly-only;
// gated on cfg(test) so ordinary `cargo build`/maturin builds stay on stable.
#![allow(unexpected_cfgs)]
#![cfg_attr(all(test, coniferest_nightly_bench), feature(test))]
#[cfg(all(test, coniferest_nightly_bench))]
extern crate test;

mod data;
mod float;
mod forest;
mod stable_sort;
mod tree;
mod utils;

use crate::forest::{build_trees, calc_apply, calc_feature_delta_sum, calc_paths_sum};
use crate::stable_sort::argpartial_sort;
use crate::tree::PyTree;
use crate::utils::average_path_length_py;
use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyTree>()?;
    m.add_function(wrap_pyfunction!(average_path_length_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_trees, m)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_feature_delta_sum, m)?)?;
    m.add_function(wrap_pyfunction!(calc_apply, m)?)?;
    m.add_function(wrap_pyfunction!(argpartial_sort, m)?)?;
    Ok(())
}
