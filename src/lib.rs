use itertools::Itertools;
use ndarray::{Array1, ArrayView1, ArrayView2, Zip};
use num_traits::AsPrimitive;
use numpy::{Element, PyArray, PyArrayDescr};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::py_run;
use pyo3::types::PyDict;

/// Selector is the representation of decision tree nodes: either branches or leafs.
///
/// We use "C"-representation with standard alignment (np.dtype(align=True)), but "packed"
/// (dtype(aligh=False)) would work as well.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub(crate) struct Selector {
    /// Feature index to branch on, -1.0 if leaf
    feature: i32,
    /// Index of left subtree, leaf_id if leaf
    left: i32,
    /// Feature value to branch on, resulting decision score if leaf
    value: f64,
    /// Index of right subtree, -1 if leaf
    right: i32,
}

impl Selector {
    pub(crate) fn dtype(py: Python) -> PyResult<&PyArrayDescr> {
        let locals = PyDict::new(py);
        py_run!(
            py,
            *locals,
            r#"
            dtype = __import__('numpy').dtype(
                [('feature', 'i4'), ('left', 'i4'), ('value', 'f8'), ('right', 'i4')],
                align=True,
            )
            "#
        );
        Ok(locals
            .get_item("dtype")
            .expect("Error in built-in Python code for dtype initialization")
            .downcast::<PyArrayDescr>()?)
    }

    #[inline(always)]
    pub(crate) fn is_leaf(&self) -> bool {
        self.feature == -1
    }
}

/// Implementation of [numpy::Element] for [Selector]
///
/// Safety: we guarantee that [Selector] has the same layout as it would have in numpy with
/// [Selector::dtype]
unsafe impl Element for Selector {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python) -> &PyArrayDescr {
        Self::dtype(py).unwrap()
    }
}

// It looks like the performance is not affected by returning a copy of Selector, not reference.
#[inline]
fn find_leaf<T>(tree: &[Selector], sample: &[T]) -> Selector
where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let mut i = 0;
    loop {
        let selector = *unsafe { tree.get_unchecked(i) };
        if selector.is_leaf() {
            break selector;
        }

        // TODO: do opposite type casting: what if we trained on huge f64 and predict on f32?
        let threshold: T = selector.value.as_();
        i = if *unsafe { sample.get_unchecked(selector.feature as usize) } <= threshold {
            selector.left as usize
        } else {
            selector.right as usize
        };
    }
}

#[pyfunction]
#[pyo3(signature = (selectors, indices, data, weights = None, num_threads = 0))]
pub(crate) fn calc_paths_sum<'py>(
    py: Python<'py>,
    selectors: &PyArray1<Selector>,
    indices: &PyArray1<i64>,
    // TODO: support f32 data
    data: &PyArray2<f64>,
    weights: Option<&PyArray1<f64>>,
    num_threads: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let selectors = selectors.readonly();
    let indices = indices.readonly();
    let data = data.readonly();
    let weights = weights.map(|weights| weights.readonly());
    let weights_view = weights.as_ref().map(|weights| weights.as_array());

    // TODO: add indices check

    // TODO: check arrays are contiguous and in memory order

    // Here we need to dispatch `data` and run the template function
    let paths = calc_paths_sum_impl(
        selectors.as_array(),
        indices.as_array(),
        data.as_array(),
        weights_view,
        num_threads,
    );
    Ok(PyArray::from_owned_array(py, paths))
}

fn calc_paths_sum_impl<T>(
    selectors: ArrayView1<Selector>,
    indices: ArrayView1<i64>,
    data: ArrayView2<T>,
    weights: Option<ArrayView1<f64>>,
    num_threads: usize,
) -> Array1<f64>
where
    T: Copy + Send + Sync + PartialOrd + 'static,
    f64: AsPrimitive<T>,
{
    let mut paths = Array1::zeros(data.nrows());

    let indices = indices.as_slice().unwrap();
    let selectors = selectors.as_slice().unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Cannot build rayon ThreadPool")
        .install(|| {
            Zip::from(paths.view_mut())
                .and(data.rows())
                .par_for_each(|path, sample| {
                    for (tree_start, tree_end) in
                        indices.iter().map(|i| *i as usize).tuple_windows()
                    {
                        let tree_selectors =
                            unsafe { selectors.get_unchecked(tree_start..tree_end) };

                        let leaf = find_leaf(tree_selectors, sample.as_slice().unwrap());

                        if let Some(weights) = weights {
                            *path += *unsafe { weights.uget(leaf.left as usize) } * leaf.value;
                        } else {
                            *path += leaf.value;
                        }
                    }
                })
        });

    paths
}

// #[pyfunction]
// pub fn calc_paths_sum_transpose(
//     selectors: Vec<Selector>,
//     indices: Array1<i64>,
//     data: Array2<f64>,
//     leaf_count: usize,
//     weights: Option<Array1<f64>>,
//     num_threads: usize,
// ) -> PyResult<Array1<f64>> {
//     let values = Array1::zeros(leaf_count);
//
//     paths_sum_transpose(
//         &selectors,
//         &indices,
//         &data,
//         &mut values.view_mut(),
//         weights.as_deref(),
//         num_threads,
//     );
//     Ok(values)
// }
//
// fn paths_sum_transpose(
//     selectors: &[Selector],
//     indices: &[i64],
//     data: &Array2<f64>,
//     values: &mut Array1<f64>,
//     weights: Option<&Array1<f64>>,
//     num_threads: usize,
// ) {
//     values
//         .par_chunks_mut(1)
//         .enumerate()
//         .for_each(|(tree_index, mut values)| {
//             if tree_index >= indices.len() - 1 {
//                 return;
//             }
//             for x_index in 0..data.nrows() {
//                 let mut i: i64 = 0;
//                 let mut selector: &Selector;
//                 loop {
//                     selector = &selectors[(indices[tree_index] + i) as usize];
//                     if selector.is_leaf() {
//                         break;
//                     }
//
//                     if data[[x_index, selector.feature as usize]] <= selector.value {
//                         i = selector.left;
//                     } else {
//                         i = selector.right;
//                     }
//                 }
//                 if let Some(weights) = weights {
//                     values[selector.get_leaf_id()] += weights[x_index] * selector.value;
//                 } else {
//                     values[selector.get_leaf_id()] += selector.value;
//                 }
//             }
//         });
// }

#[pymodule]
#[pyo3(name = "calc_paths_sum")]
fn rust_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("selector_dtype", Selector::dtype(_py)?)?;
    m.add_function(wrap_pyfunction!(calc_paths_sum, m)?)?;
    // m.add_function(wrap_pyfunction!(calc_paths_sum_transpose, m)?)?;
    Ok(())
}
