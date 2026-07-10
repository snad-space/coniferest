use crate::tree::{Tree, TreeDtype, TreeInner};
use ndarray::parallel::prelude::*;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type DeltaSumHitCount<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>);

/// Input data: 2-D numpy array of features, C-contiguous, one sample per row.
#[derive(FromPyObject)]
pub(crate) enum Data<'py> {
    F64(PyReadonlyArray2<'py, f64>),
    F32(PyReadonlyArray2<'py, f32>),
}

/// Borrowed trees of the forest with their global leaf offsets.
///
/// `Tree` is a frozen pyclass, so the borrows are GIL-independent and the
/// forest can be traversed in parallel even when stored as a Python list.
/// All the trees must be built on the data dtype `T`, matching the dtype
/// of the scored data, so the traversal never casts values.
struct Forest<'a, T> {
    trees: Vec<&'a TreeInner<T>>,
    /// Global leaf index of the first leaf of each tree
    leaf_offsets: Vec<u32>,
    n_leaves: usize,
}

impl<'a, T> Forest<'a, T>
where
    T: TreeDtype,
{
    fn new(trees: &'a [Py<Tree>], n_features: usize) -> PyResult<Self> {
        let trees: Vec<&TreeInner<T>> = trees
            .iter()
            .map(|tree| {
                T::tree_inner(tree.get()).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "data dtype is {}, but the tree was built on {} data",
                        T::NAME,
                        tree.get().dtype_name(),
                    ))
                })
            })
            .collect::<PyResult<_>>()?;

        let mut leaf_offsets = Vec::with_capacity(trees.len());
        let mut n_leaves: u32 = 0;
        for tree in &trees {
            if tree.n_features as usize != n_features {
                return Err(PyValueError::new_err(format!(
                    "data has {} features, but a tree was built on {} features",
                    n_features, tree.n_features,
                )));
            }
            leaf_offsets.push(n_leaves);
            n_leaves = n_leaves
                .checked_add(tree.n_leaves)
                .ok_or_else(|| PyValueError::new_err("too many leaves in the forest"))?;
        }

        Ok(Self {
            trees,
            leaf_offsets,
            n_leaves: n_leaves as usize,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (&&TreeInner<T>, u32)> {
        self.trees.iter().zip(self.leaf_offsets.iter().copied())
    }
}

#[inline]
fn check_data<T>(data: ArrayView2<T>) -> PyResult<()> {
    if !data.is_standard_layout() {
        return Err(PyValueError::new_err(
            "data must be contiguous and in memory order",
        ));
    }
    Ok(())
}

#[inline]
fn check_leaf_array(array: ArrayView1<f64>, n_leaves: usize, name: &str) -> PyResult<()> {
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

#[inline]
fn get_num_threads(nrows: usize, num_threads: usize, batch_size: usize) -> PyResult<usize> {
    if batch_size == 0 {
        Err(PyValueError::new_err("batch_size must be greater than 0"))
    } else {
        let n_jobs = nrows.div_ceil(batch_size);
        Ok(usize::min(num_threads, n_jobs))
    }
}

/// Calculate the sum of path lengths over the forest for every sample.
///
/// If `leaf_values` is given, it is used instead of the values stored in
/// the leaves, indexed by the global leaf index. If `weights` is given,
/// every value is multiplied by the weight of the reached leaf.
#[pyfunction]
#[pyo3(signature = (trees, data, weights = None, leaf_values = None, *, num_threads, batch_size))]
pub(crate) fn calc_paths_sum<'py>(
    py: Python<'py>,
    trees: Vec<Py<Tree>>,
    data: Data<'py>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    leaf_values: Option<PyReadonlyArray1<'py, f64>>,
    num_threads: usize,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    match &data {
        Data::F64(array) => calc_paths_sum_generic(
            py,
            &trees,
            array,
            weights,
            leaf_values,
            num_threads,
            batch_size,
        ),
        Data::F32(array) => calc_paths_sum_generic(
            py,
            &trees,
            array,
            weights,
            leaf_values,
            num_threads,
            batch_size,
        ),
    }
}

fn calc_paths_sum_generic<'py, T>(
    py: Python<'py>,
    trees: &[Py<Tree>],
    data: &PyReadonlyArray2<'py, T>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    leaf_values: Option<PyReadonlyArray1<'py, f64>>,
    num_threads: usize,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    T: Element + Copy + Send + Sync + PartialOrd + TreeDtype,
{
    let data_view = data.as_array();
    check_data(data_view)?;

    let forest = Forest::new(trees, data_view.ncols())?;

    let weights_view = weights.as_ref().map(|weights| weights.as_array());
    if let Some(weights_view) = weights_view {
        check_leaf_array(weights_view, forest.n_leaves, "weights")?;
    }

    let leaf_values_view = leaf_values
        .as_ref()
        .map(|leaf_values| leaf_values.as_array());
    if let Some(leaf_values_view) = leaf_values_view {
        check_leaf_array(leaf_values_view, forest.n_leaves, "leaf_values")?;
    }

    let num_threads = get_num_threads(data_view.nrows(), num_threads, batch_size)?;

    let paths = PyArray1::zeros(py, data_view.nrows(), false);
    // SAFETY: this call invalidates other views, but it is the only view we need
    let paths_view_mut = unsafe { paths.as_array_mut() };

    calc_paths_sum_impl(
        &forest,
        data_view,
        weights_view,
        leaf_values_view,
        num_threads,
        batch_size,
        paths_view_mut,
    );

    Ok(paths)
}

fn calc_paths_sum_impl<T>(
    forest: &Forest<T>,
    data: ArrayView2<T>,
    weights: Option<ArrayView1<f64>>,
    leaf_values: Option<ArrayView1<f64>>,
    num_threads: usize,
    batch_size: usize,
    paths: ArrayViewMut1<f64>,
) where
    T: Copy + Send + Sync + PartialOrd + TreeDtype,
{
    let inner_fn = |path: &mut f64, sample: ArrayView1<T>| {
        let sample = sample.as_slice().unwrap();
        for (tree, leaf_offset) in forest.iter() {
            let leaf = tree.find_leaf(sample);
            let leaf_id = (leaf_offset + leaf.leaf_index) as usize;

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

    if num_threads == 1 {
        zip.for_each(inner_fn);
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Cannot build rayon ThreadPool")
            .install(|| {
                zip.into_par_iter()
                    .with_min_len(batch_size)
                    .for_each(|(path, sample)| inner_fn(path, sample));
            });
    }
}

/// Calculate the sum of path length deltas and the hit count per feature.
#[pyfunction]
#[pyo3(signature = (trees, data, *, num_threads, batch_size))]
pub(crate) fn calc_feature_delta_sum<'py>(
    py: Python<'py>,
    trees: Vec<Py<Tree>>,
    data: Data<'py>,
    num_threads: usize,
    batch_size: usize,
) -> PyResult<DeltaSumHitCount<'py>> {
    match &data {
        Data::F64(array) => {
            calc_feature_delta_sum_generic(py, &trees, array, num_threads, batch_size)
        }
        Data::F32(array) => {
            calc_feature_delta_sum_generic(py, &trees, array, num_threads, batch_size)
        }
    }
}

fn calc_feature_delta_sum_generic<'py, T>(
    py: Python<'py>,
    trees: &[Py<Tree>],
    data: &PyReadonlyArray2<'py, T>,
    num_threads: usize,
    batch_size: usize,
) -> PyResult<DeltaSumHitCount<'py>>
where
    T: Element + Copy + Send + Sync + PartialOrd + TreeDtype,
{
    let data_view = data.as_array();
    check_data(data_view)?;

    let forest = Forest::new(trees, data_view.ncols())?;

    let num_threads = get_num_threads(data_view.nrows(), num_threads, batch_size)?;

    let delta_sum = PyArray2::zeros(py, (data_view.nrows(), data_view.ncols()), false);
    let hit_count = PyArray2::zeros(py, (data_view.nrows(), data_view.ncols()), false);

    // SAFETY: this call invalidates other views, but it is the only view we need
    let delta_sum_view = unsafe { delta_sum.as_array_mut() };
    // SAFETY: this call invalidates other views, but it is the only view we need
    let hit_count_view = unsafe { hit_count.as_array_mut() };

    calc_feature_delta_sum_impl(
        &forest,
        data_view,
        num_threads,
        batch_size,
        delta_sum_view,
        hit_count_view,
    );

    Ok((delta_sum, hit_count))
}

fn calc_feature_delta_sum_impl<T>(
    forest: &Forest<T>,
    data: ArrayView2<T>,
    num_threads: usize,
    batch_size: usize,
    mut delta_sum: ArrayViewMut2<f64>,
    mut hit_count: ArrayViewMut2<i64>,
) where
    T: Copy + Send + Sync + PartialOrd + TreeDtype,
{
    let inner_fn = |sample: ArrayView1<T>,
                    mut delta_sum_row: ArrayViewMut1<f64>,
                    mut hit_count_row: ArrayViewMut1<i64>| {
        let sample = sample.as_slice().unwrap();
        for tree in &forest.trees {
            // Sidecar array with the average path length of each node
            let node_apl = &tree.node_average_path_length;

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

    if num_threads == 1 {
        zip.for_each(inner_fn);
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Cannot build rayon ThreadPool")
            .install(|| {
                zip.into_par_iter().with_min_len(batch_size).for_each(
                    |(sample, delta_sum_row, hit_count_row)| {
                        inner_fn(sample, delta_sum_row, hit_count_row)
                    },
                );
            });
    }
}

/// Find the global leaf index reached by every sample in every tree.
#[pyfunction]
#[pyo3(signature = (trees, data, *, num_threads, batch_size))]
pub(crate) fn calc_apply<'py>(
    py: Python<'py>,
    trees: Vec<Py<Tree>>,
    data: Data<'py>,
    num_threads: usize,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    match &data {
        Data::F64(array) => calc_apply_generic(py, &trees, array, num_threads, batch_size),
        Data::F32(array) => calc_apply_generic(py, &trees, array, num_threads, batch_size),
    }
}

fn calc_apply_generic<'py, T>(
    py: Python<'py>,
    trees: &[Py<Tree>],
    data: &PyReadonlyArray2<'py, T>,
    num_threads: usize,
    batch_size: usize,
) -> PyResult<Bound<'py, PyArray2<u32>>>
where
    T: Element + Copy + Send + Sync + PartialOrd + TreeDtype,
{
    let data_view = data.as_array();
    check_data(data_view)?;

    let forest = Forest::new(trees, data_view.ncols())?;

    let num_threads = get_num_threads(data_view.nrows(), num_threads, batch_size)?;

    let leafs = PyArray2::zeros(py, (data_view.nrows(), forest.trees.len()), false);
    // SAFETY: this call invalidates other views, but it is the only view we need
    let leafs_view = unsafe { leafs.as_array_mut() };

    calc_apply_impl(&forest, data_view, num_threads, batch_size, leafs_view);

    Ok(leafs)
}

fn calc_apply_impl<T>(
    forest: &Forest<T>,
    data: ArrayView2<T>,
    num_threads: usize,
    batch_size: usize,
    mut leafs: ArrayViewMut2<u32>,
) where
    T: Copy + Send + Sync + PartialOrd + TreeDtype,
{
    let inner_fn = |sample: ArrayView1<T>, mut sample_leafs: ArrayViewMut1<u32>| {
        let sample = sample.as_slice().unwrap();
        let leafs_slice = sample_leafs.as_slice_mut().unwrap();
        for ((tree, leaf_offset), leaf_id) in forest.iter().zip(leafs_slice.iter_mut()) {
            let leaf = tree.find_leaf(sample);
            *leaf_id = leaf_offset + leaf.leaf_index;
        }
    };

    let zip = Zip::from(data.rows()).and(leafs.rows_mut());

    if num_threads == 1 {
        zip.for_each(inner_fn);
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Cannot build rayon ThreadPool")
            .install(|| {
                zip.into_par_iter()
                    .with_min_len(batch_size)
                    .for_each(|(sample, sample_leafs)| inner_fn(sample, sample_leafs));
            });
    }
}
