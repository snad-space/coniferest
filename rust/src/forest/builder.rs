//! Forest building

use crate::float::Float;
use crate::forest::inner::ForestInner;
use crate::forest::py::PyCoreForest;
use crate::tree::TreeInner;
use ndarray::ArrayRef2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, Python};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::sync::Arc;

pub(super) fn build_forest_py<'py, T>(
    py: Python<'py>,
    data: &PyReadonlyArray2<'_, T>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    max_depth: usize,
    num_threads: usize,
) -> PyResult<PyCoreForest>
where
    T: Float,
    PyCoreForest: From<ForestInner<T>>,
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
    let n_features = data_view.ncols() as u32;
    if data_view.is_empty() {
        return Err(PyValueError::new_err("data must not be empty"));
    }
    // A deeper tree is not possible: every split isolates at least one sample
    let max_depth = u16::try_from(max_depth)
        .map_err(|_| PyValueError::new_err(format!("max_depth must not exceed {}", u16::MAX)))?;

    let forest_inner = py.detach(|| {
        build_forest_inner(
            &data_view,
            seed,
            n_trees,
            n_subsamples,
            n_features,
            max_depth,
            num_threads,
        )
    });
    Ok(forest_inner.into())
}

fn build_forest_inner<T>(
    data: &ArrayRef2<T>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    n_features: u32,
    max_depth: u16,
    num_threads: usize,
) -> ForestInner<T>
where
    T: Float,
{
    let thread_pool = ForestInner::<T>::init_thread_pool(num_threads);

    let trees = build_trees(
        data,
        seed,
        n_trees,
        n_subsamples,
        max_depth,
        thread_pool.as_ref(),
    );

    ForestInner::with_thread_pool(trees, n_features, thread_pool)
}

fn build_trees<T>(
    data: &ArrayRef2<T>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    max_depth: u16,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Vec<Arc<TreeInner<T>>>
where
    T: Float,
{
    // Sample random seeds for all the tree building jobs in advance, so the
    // result does not depend on the number of threads
    let mut master_rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let child_seeds_iter = (0..n_trees).map(|_| master_rng.next_u64());
    let tree_build_fn = |child_seed| {
        let rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);
        let tree = TreeInner::build(data, n_subsamples, max_depth, rng);
        Arc::new(tree)
    };

    if let Some(pool) = thread_pool {
        // We have to collect first, the alternative is to use `par_bridge`, but it doesn't
        // keep the order of the trees, so reproducibiliy may be affected.
        let child_seeds: Vec<_> = child_seeds_iter.collect();
        pool.install(|| child_seeds.into_par_iter().map(tree_build_fn).collect())
    } else {
        child_seeds_iter.map(tree_build_fn).collect()
    }
}
