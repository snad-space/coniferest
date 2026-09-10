//! Forest building

use crate::data::Data;
use crate::tree::{PyTree, TreeDtype, TreeInner};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, Python, pyfunction};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

fn build_trees_impl<T>(
    py: Python<'_>,
    data: &PyReadonlyArray2<'_, T>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    max_depth: usize,
    num_threads: usize,
) -> PyResult<Vec<PyTree>>
where
    T: TreeDtype,
{
    let data_view = data.as_array();
    if !data_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "data must be contiguous and in memory order",
        ));
    }
    if n_subsamples == 0 || n_subsamples > data_view.nrows() {
        return Err(PyValueError::new_err(
            "n_subsamples must be positive and not greater than the number of samples",
        ));
    }
    // Node and feature indices are stored as u32
    if 2 * n_subsamples - 1 > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "n_subsamples exceeds 2^31, IsolationForest usually doesn't require a huge number of samples. Please set a lower value or request support for larger trees",
        ));
    }
    if data_view.ncols() > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "number of features is equal or larger than 2^32, it is likely to be a mistake with data shape",
        ));
    }
    if data_view.is_empty() {
        return Err(PyValueError::new_err("data must not be empty"));
    }
    // A deeper tree is not possible: every split isolates at least one sample
    let max_depth = u16::try_from(max_depth)
        .map_err(|_| PyValueError::new_err(format!("max_depth must not exceed {}", u16::MAX)))?;

    // Sample random seeds for all the tree building jobs in advance, so the
    // result does not depend on the number of threads
    let mut master_rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let child_seeds_iter = (0..n_trees).map(|_| master_rng.next_u64());
    let tree_build_fn = |child_seed| {
        let rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);
        TreeInner::build(&data_view, n_subsamples, max_depth, rng)
    };

    let trees: Vec<TreeInner<T>> = py.detach(|| {
        if num_threads == 1 {
            child_seeds_iter.map(tree_build_fn).collect()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("Cannot build rayon ThreadPool")
                // We have to collect first, the alternative is to use `par_bridge`, but it doesn't
                // keep the order of the trees, so reproducibiliy may be affected.
                .install(|| {
                    child_seeds_iter
                        .collect::<Vec<_>>()
                        .into_par_iter()
                        .map(tree_build_fn)
                        .collect()
                })
        }
    });

    Ok(trees
        .into_iter()
        .map(|inner| T::wrap(inner).into())
        .collect())
}

/// Build isolation trees in parallel.
///
/// `n_trees` trees are built, each from its own random subsample of `data`
/// rows. Per-tree random seeds are derived from `seed` in advance, so the
/// result is reproducible and does not depend on `num_threads` (0 means all
/// available CPUs). Returns a list of `Tree` objects.
#[pyfunction]
#[pyo3(signature = (data, seed, n_trees, n_subsamples, max_depth, *, num_threads))]
pub(crate) fn build_trees<'py>(
    py: Python<'py>,
    data: Data<'py>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    max_depth: usize,
    num_threads: usize,
) -> PyResult<Vec<PyTree>> {
    match &data {
        Data::F32(array) => build_trees_impl(
            py,
            array,
            seed,
            n_trees,
            n_subsamples,
            max_depth,
            num_threads,
        ),
        Data::F64(array) => build_trees_impl(
            py,
            array,
            seed,
            n_trees,
            n_subsamples,
            max_depth,
            num_threads,
        ),
    }
}
