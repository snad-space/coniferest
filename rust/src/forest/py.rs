//! Forest class for Python bindings

use crate::data::Data;
use crate::forest::builder::build_forest_py;
use crate::forest::inner::{ForestInner, ForestVariant};
use crate::forest::traversal::{
    calc_apply_py, calc_feature_delta_sum_py, calc_leaf_values_py, calc_paths_sum_py,
};
use crate::tree::{PyTree, TreeVariant};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{PyResult, PyTypeInfo, Python};

#[derive(Clone)]
#[pyclass(
    name = "CoreForest",
    module = "coniferest._core",
    sequence,
    skip_from_py_object
)]
pub(crate) struct PyCoreForest(ForestVariant);

impl From<ForestInner<f32>> for PyCoreForest {
    fn from(inner: ForestInner<f32>) -> Self {
        PyCoreForest(ForestVariant::F32(inner))
    }
}

impl From<ForestInner<f64>> for PyCoreForest {
    fn from(inner: ForestInner<f64>) -> Self {
        PyCoreForest(ForestVariant::F64(inner))
    }
}

/// Dispatch `$body` over variants of [ForestVariant] and [Data].
macro_rules! dispatch_forest_data {
    ($forest_variant:expr, $data_variant:expr, |$forest:ident, $data:ident| => $body:expr) => {
        match ($forest_variant, $data_variant) {
            (ForestVariant::F32($forest), Data::F32($data)) => $body,
            (ForestVariant::F64($forest), Data::F64($data)) => $body,
            (ForestVariant::F32(_), Data::F64(_)) => {
                return Err(PyTypeError::new_err(
                    "Forest dtype float32 does not match data dtype float64. Please either cast the data or rebuild the forest.",
                ));
            }
            (ForestVariant::F64(_), Data::F32(_)) => {
                return Err(PyTypeError::new_err(
                    "Forest dtype float64 does not match data dtype float32. Please either cast the data or rebuild the forest.",
                ));
            }
        }
    };
}

use crate::forest::inner::dispatch_forest_tree;

macro_rules! dispatch_two_forests {
    ($left_variant:expr, $right_variant:expr, |$left:ident, $right:ident| => $body:expr) => {
        match ($left_variant, $right_variant) {
            (ForestVariant::F32($left), ForestVariant::F32($right)) => $body,
            (ForestVariant::F64($left), ForestVariant::F64($right)) => $body,
            (ForestVariant::F32(_), ForestVariant::F64(_)) => {
                return Err(PyTypeError::new_err(
                    "Left forest dtype float32 does not match right forest dtype float64. Please either cast the forests or rebuild the forests.",
                ));
            }
            (ForestVariant::F64(_), ForestVariant::F32(_)) => {
                return Err(PyTypeError::new_err(
                    "Left forest dtype float64 does not match right forest dtype float32. Please either cast the forests or rebuild the forests.",
                ));
            }
        }
    };
}

macro_rules! on_forest_inner {
    ($variant:expr, $forest:ident => $body:expr) => {
        match $variant {
            ForestVariant::F32($forest) => $body,
            ForestVariant::F64($forest) => $body,
        }
    };
}

/// Builds CoreForest object representing an isolation forest.
///
/// Parameters
/// ----------
/// data : <TODO>
/// <TODO>
///
/// Returns
/// -------
/// CoreForest
#[pyfunction]
#[pyo3(signature = (data, *, seed, n_trees, n_subsamples, max_depth, num_threads))]
pub(crate) fn build_core_forest<'py>(
    py: Python<'py>,
    data: Data<'py>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    max_depth: usize,
    num_threads: usize,
) -> PyResult<PyCoreForest> {
    match &data {
        Data::F32(data) => build_forest_py(
            py,
            data,
            seed,
            n_trees,
            n_subsamples,
            max_depth,
            num_threads,
        ),
        Data::F64(data) => build_forest_py(
            py,
            data,
            seed,
            n_trees,
            n_subsamples,
            max_depth,
            num_threads,
        ),
    }
}

pub(crate) type DeltaSumHitCount<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

#[pymethods]
impl PyCoreForest {
    /// Build a forest from a non-empty list of trees, all of the same dtype.
    ///
    /// `n_features` must be supplied explicitly since a `Tree` does not know
    /// the total feature count of the data it was built from.
    #[new]
    #[pyo3(signature = (trees, *, n_features, num_threads))]
    fn new(trees: Vec<PyTree>, n_features: u32, num_threads: usize) -> PyResult<Self> {
        let trees = trees.into_iter().map(|tree| tree.0).collect();
        Ok(PyCoreForest(ForestVariant::from_trees(
            trees,
            n_features,
            num_threads,
        )?))
    }

    /// Pickle support.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (Vec<PyTree>,))> {
        let py = slf.py();
        let this = slf.borrow();
        let trees: Vec<PyTree> = on_forest_inner!(&this.0, forest => forest.trees().iter().cloned().map(Into::into).collect());
        let kwargs = PyDict::new(py);
        kwargs.set_item("n_features", this.n_features())?;
        kwargs.set_item("num_threads", this.num_threads())?;
        let ctor = py
            .import("functools")?
            .getattr("partial")?
            .call((Self::type_object(py),), Some(&kwargs))?;
        Ok((ctor, (trees,)))
    }

    #[getter]
    fn n_features(&self) -> u32 {
        on_forest_inner!(&self.0, forest => forest.n_features())
    }

    #[getter]
    fn num_threads(&self) -> usize {
        on_forest_inner!(&self.0, forest => forest.num_threads())
    }

    #[setter]
    fn set_num_threads(&mut self, num_threads: usize) {
        on_forest_inner!(&mut self.0, forest => forest.set_num_threads(num_threads))
    }

    /// Calculate the sum of path lengths over the forest for every sample.
    ///
    /// If `leaf_values` is given, it is used instead of the values stored in
    /// the leaves, indexed by the global leaf index. If `weights` is given,
    /// every value is multiplied by the weight of the reached leaf.
    #[pyo3(signature = (data, weights = None, leaf_values = None, *, batch_size))]
    fn calc_paths_sum<'py>(
        &self,
        py: Python<'py>,
        data: Data<'py>,
        weights: Option<PyReadonlyArray1<'py, f64>>,
        leaf_values: Option<PyReadonlyArray1<'py, f64>>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        dispatch_forest_data!(&self.0, data, |forest, data| => calc_paths_sum_py(
            py,
            forest,
            &data,
            weights,
            leaf_values,
            batch_size,
        ))
    }

    /// Calculate the sum of path length deltas and the hit count per feature.
    #[pyo3(signature = (data, *, batch_size))]
    fn calc_feature_delta_sum<'py>(
        &self,
        py: Python<'py>,
        data: Data<'py>,
        batch_size: usize,
    ) -> PyResult<DeltaSumHitCount<'py>> {
        dispatch_forest_data!(&self.0, data, |forest, data| =>
            calc_feature_delta_sum_py(py, forest, &data, batch_size)
        )
    }

    /// Find the global leaf index reached by every sample in every tree.
    #[pyo3(signature = (data, *, batch_size))]
    fn calc_apply<'py>(
        &self,
        py: Python<'py>,
        data: Data<'py>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<usize>>> {
        dispatch_forest_data!(&self.0, data, |forest, data| => calc_apply_py(py, forest, &data, batch_size))
    }

    /// Find values of the leaves be every sample
    #[pyo3(signature = (data, *, batch_size))]
    fn calc_leaf_values<'py>(
        &self,
        py: Python<'py>,
        data: Data<'py>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        dispatch_forest_data!(&self.0, data, |forest, data| => calc_leaf_values_py(py, forest, &data, batch_size))
    }

    //
    // Sequence methods
    //

    fn __len__(&self) -> usize {
        on_forest_inner!(&self.0, forest => forest.trees().len())
    }

    /// Returns the tree at the specified index.
    fn __getitem__(&self, index: usize) -> PyResult<PyTree> {
        let tree = on_forest_inner!(&self.0, forest => forest
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("forest tree index out of range"))?
            .into()
        );
        Ok(tree)
    }

    fn __setitem__(&mut self, index: usize, tree: PyTree) -> PyResult<()> {
        dispatch_forest_tree!(&mut self.0, tree.0, |forest, tree| => {
            forest.trees_mut()[index] = tree;
        });
        Ok(())
    }

    fn __delitem__(&mut self, index: usize) -> PyResult<()> {
        on_forest_inner!(&mut self.0, forest => {
            let _tree_inner = forest
                .try_remove_tree(index)
                .ok_or_else(|| PyIndexError::new_err("forest tree index out of range"))?;
        });
        Ok(())
    }

    /// Implements + operator (add) between two forests.
    fn __concat__(&self, other: &Self) -> PyResult<Self> {
        dispatch_two_forests!(&self.0, &other.0, |left, right| => {
            if left.n_features() != right.n_features() {
                return Err(PyValueError::new_err(
                    format!("Cannot concatenate forests with different number of features: {} in the left tree and {} in the right tree", left.n_features(), right.n_features()),
                ));
            }
            let mut concat: ForestInner<_> = left.clone();
            concat.trees_mut().extend(right.trees().iter().cloned());
            Ok(concat.into())
        })
    }

    // https://github.com/PyO3/pyo3/issues/6211
    /*
    /// Implements += operator (iadd) between two forests.
    ///
    /// Uses num_threads of the left forest.
    fn __inplace_concat__<'py>(slf: Bound<'py, Self>, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        println!("__inplace_concat__");
        // If doing `forest += forest`
        if slf.is(&other) {
            Ok(Self::__inplace_repeat__(slf, 2))
        } else {
            {
                let mut slf_mut = slf.borrow_mut();
                dispatch_two_forests!(&mut slf_mut.0, &other.borrow().0, |left, right| => {
                    if left.n_features() != right.n_features() {
                        return Err(PyValueError::new_err(
                            format!("Cannot concatenate forests with different number of features: {} in the left tree and {} in the right tree", left.n_features(), right.n_features()),
                        ));
                    }
                    left.trees_mut().extend(right.trees().iter().cloned());
                });
            }
            Ok(slf)
        }
    }
    */

    fn __repeat__(&self, count: usize) -> Self {
        on_forest_inner!(&self.0, forest => {
            let new_forest = if count == 0 {
                ForestInner::new(forest.n_features(), forest.num_threads())
            } else {
                    let trees = forest.trees();
                    let mut new_forest = forest.clone();
                    new_forest.trees_mut().reserve((count - 1) * trees.len());
                    for _ in 1..count {
                        new_forest.trees_mut().extend(trees.iter().cloned());
                    }
                    new_forest
            };
            new_forest.into()
        })
    }

    // https://github.com/PyO3/pyo3/issues/6211
    /*
    /// Implements the *= operator (imul) for a forest.
    fn __inplace_repeat__<'py>(slf: Bound<'py, Self>, count: usize) -> Bound<'py, Self> {
        {
            let mut slf_mut = slf.borrow_mut();
            on_forest_inner!(&mut slf_mut.0, forest => {
                match count {
                    0 => forest.trees_mut().clear(),
                    1 => {},
                    _ => {
                        let trees = forest.trees().to_vec();
                        forest.trees_mut().reserve((count - 1) * trees.len());
                        for _ in 1..count {
                            forest.trees_mut().extend(trees.iter().cloned());
                        }
                    }
                }
            });
        }
        slf
    }
    */
}
