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

use crate::forest::{PyCoreForest, build_core_forest};
use crate::stable_sort::argpartial_sort;
use crate::tree::PyTree;
use crate::utils::average_path_length_py;
use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyTree>()?;
    m.add_class::<PyCoreForest>()?;
    m.add_function(wrap_pyfunction!(average_path_length_py, m)?)?;
    m.add_function(wrap_pyfunction!(argpartial_sort, m)?)?;
    m.add_function(wrap_pyfunction!(build_core_forest, m)?)?;
    Ok(())
}
