use crate::float::Float;
use crate::forest::inner::ForestInner;
use crate::forest::py::DeltaSumHitCount;
use ndarray::{ArrayRef1, ArrayRef2, ArrayView1, ArrayViewMut1, Zip};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[inline]
fn check_data<T>(data: &ArrayRef2<T>) -> PyResult<()> {
    if !data.is_standard_layout() {
        return Err(PyValueError::new_err(
            "data must be contiguous and in memory order",
        ));
    }
    Ok(())
}

#[inline]
fn check_leaf_array(array: &ArrayRef1<f64>, n_leaves: usize, name: &str) -> PyResult<()> {
    if array.len() != n_leaves {
        return Err(PyValueError::new_err(format!(
            "{} must have the same length as the total number of leaves in the forest: \
             {} != {}",
            name,
            array.len(),
            n_leaves,
        )));
    }
    Ok(())
}

pub(super) fn calc_paths_sum_py<'py, T>(
    py: Python<'py>,
    forest: &ForestInner<T>,
    data: &PyReadonlyArray2<'py, T>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    leaf_values: Option<PyReadonlyArray1<'py, f64>>,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    T: Float,
{
    let data_view = data.as_array();
    check_data(&data_view)?;

    let weights_view = weights.as_ref().map(|weights| weights.as_array());
    if let Some(weights_view) = weights_view.as_deref() {
        check_leaf_array(weights_view, forest.n_leaves(), "weights")?;
    }

    let leaf_values_view = leaf_values
        .as_ref()
        .map(|leaf_values| leaf_values.as_array());
    if let Some(leaf_values_view) = leaf_values_view.as_deref() {
        check_leaf_array(leaf_values_view, forest.n_leaves(), "leaf_values")?;
    }

    let paths = PyArray1::zeros(py, data_view.nrows(), false);
    // SAFETY: this call invalidates other views, but it is the only view we need
    let mut paths_view_mut = unsafe { paths.as_array_mut() };

    calc_paths_sum_impl(
        forest,
        &data_view,
        weights_view.as_deref(),
        leaf_values_view.as_deref(),
        batch_size,
        &mut paths_view_mut,
    );

    Ok(paths)
}

fn calc_paths_sum_impl<T>(
    forest: &ForestInner<T>,
    data: &ArrayRef2<T>,
    weights: Option<&ArrayRef1<f64>>,
    leaf_values: Option<&ArrayRef1<f64>>,
    batch_size: usize,
    paths: &mut ArrayRef1<f64>,
) where
    T: Float,
{
    let inner_fn = |(path, sample): (&mut f64, ArrayView1<T>)| {
        let sample = sample.as_slice().unwrap();
        for (tree, leaf_offset) in forest.iter() {
            let leaf = tree.find_leaf(sample);
            let leaf_id = leaf_offset + leaf.leaf_index as usize;

            let value = match leaf_values {
                Some(leaf_values) => *unsafe { leaf_values.uget(leaf_id) },
                None => leaf.value as f64,
            };
            match weights {
                Some(weights) => *path += *unsafe { weights.uget(leaf_id) } * value,
                None => *path += value,
            }
        }
    };

    let zip = Zip::from(paths).and(data.rows());
    forest.parallel_process(zip, inner_fn, batch_size);
}

pub(crate) fn calc_feature_delta_sum_py<'py, T>(
    py: Python<'py>,
    forest: &ForestInner<T>,
    data: &PyReadonlyArray2<'py, T>,
    batch_size: usize,
) -> PyResult<DeltaSumHitCount<'py>>
where
    T: Float,
{
    let data_view = data.as_array();
    check_data(&data_view)?;

    let delta_sum = PyArray2::zeros(py, (data_view.nrows(), data_view.ncols()), false);
    let hit_count = PyArray2::zeros(py, (data_view.nrows(), data_view.ncols()), false);

    // SAFETY: this call invalidates other views, but it is the only view we need
    let mut delta_sum_view = unsafe { delta_sum.as_array_mut() };
    // SAFETY: this call invalidates other views, but it is the only view we need
    let mut hit_count_view = unsafe { hit_count.as_array_mut() };

    calc_feature_delta_sum_impl(
        forest,
        &data_view,
        batch_size,
        &mut delta_sum_view,
        &mut hit_count_view,
    );

    Ok((delta_sum, hit_count))
}

fn calc_feature_delta_sum_impl<T>(
    forest: &ForestInner<T>,
    data: &ArrayRef2<T>,
    batch_size: usize,
    delta_sum: &mut ArrayRef2<f64>,
    hit_count: &mut ArrayRef2<i64>,
) where
    T: Float,
{
    let inner_fn = |(sample, mut delta_sum_row, mut hit_count_row): (
        ArrayView1<T>,
        ArrayViewMut1<f64>,
        ArrayViewMut1<i64>,
    )| {
        let sample = sample.as_slice().unwrap();
        for tree in forest.trees() {
            // Sidecar array with the average path length of each node
            let node_apl = tree.node_average_path_length();

            tree.for_each_split(sample, |node_index, split, child_index| {
                // Here we cast to f64 following the original Cython implementation, but
                // it is subject to change.
                *unsafe { delta_sum_row.uget_mut(split.split_feature as usize) } +=
                    1.0 + (node_apl[child_index] - node_apl[node_index]) as f64;
                *unsafe { hit_count_row.uget_mut(split.split_feature as usize) } += 1;
            });
        }
    };

    let zip = Zip::from(data.rows())
        .and(delta_sum.rows_mut())
        .and(hit_count.rows_mut());
    forest.parallel_process(zip, inner_fn, batch_size);
}

pub(crate) fn calc_apply_py<'py, T>(
    py: Python<'py>,
    forest: &ForestInner<T>,
    data: &PyReadonlyArray2<'py, T>,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray2<usize>>>
where
    T: Float,
{
    let data_view = data.as_array();
    check_data(&data_view)?;

    let leaves = PyArray2::zeros(py, (data_view.nrows(), forest.trees().len()), false);
    // SAFETY: this call invalidates other views, but it is the only view we need
    let mut leaves_view = unsafe { leaves.as_array_mut() };

    calc_apply_impl(forest, &data_view, batch_size, &mut leaves_view);

    Ok(leaves)
}

fn calc_apply_impl<T>(
    forest: &ForestInner<T>,
    data: &ArrayRef2<T>,
    batch_size: usize,
    leaves: &mut ArrayRef2<usize>,
) where
    T: Float,
{
    let inner_fn = |(sample, mut sample_leaves): (ArrayView1<T>, ArrayViewMut1<usize>)| {
        let sample = sample.as_slice().unwrap();
        let leaves_slice = sample_leaves.as_slice_mut().unwrap();
        for ((tree, leaf_offset), leaf_id) in forest.iter().zip(leaves_slice.iter_mut()) {
            let leaf = tree.find_leaf(sample);
            *leaf_id = leaf_offset + leaf.leaf_index as usize;
        }
    };

    let zip = Zip::from(data.rows()).and(leaves.rows_mut());
    forest.parallel_process(zip, inner_fn, batch_size);
}

pub(crate) fn calc_leaf_values_py<'py, T>(
    py: Python<'py>,
    forest: &ForestInner<T>,
    data: &PyReadonlyArray2<'py, T>,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>>
where
    T: Float,
{
    let data_view = data.as_array();
    check_data(&data_view)?;

    let values = PyArray2::zeros(py, (data_view.nrows(), forest.trees().len()), false);
    // SAFETY: this call invalidates other views, but it is the only view we need
    let mut values_view = unsafe { values.as_array_mut() };

    calc_leaf_values_impl(forest, &data_view, batch_size, &mut values_view);

    Ok(values)
}

fn calc_leaf_values_impl<T>(
    forest: &ForestInner<T>,
    data: &ArrayRef2<T>,
    batch_size: usize,
    values: &mut ArrayRef2<f32>,
) where
    T: Float,
{
    let inner_fn = |(sample, mut sample_values): (ArrayView1<T>, ArrayViewMut1<f32>)| {
        let sample = sample.as_slice().unwrap();
        let values_slice = sample_values.as_slice_mut().unwrap();
        for ((tree, _leaf_offset), value) in forest.iter().zip(values_slice.iter_mut()) {
            *value = tree.find_leaf(sample).value;
        }
    };

    let zip = Zip::from(data.rows()).and(values.rows_mut());
    forest.parallel_process(zip, inner_fn, batch_size);
}
