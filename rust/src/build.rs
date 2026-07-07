use crate::tree::{Leaf, Node, SplitNode, Tree};
use crate::tree_traversal::Data;
use crate::utils::average_path_length;
use itertools::Itertools;
use ndarray::ArrayView2;
use num_traits::AsPrimitive;
use numpy::{Element, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::{Bound, PyResult, Python, pyfunction};
use rand::distr::uniform::SampleUniform;
use rand::distr::{Distribution, Uniform};
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::num::NonZeroU32;

/// Node splitting logic, customizable per the tree kind.
pub(crate) trait Splitter<T> {
    /// Choose the split feature and value for the subsample rows `indices`,
    /// or return `None` to make the node a leaf.
    ///
    /// The returned value must partition `indices` into two non-empty parts.
    fn choose_split(&mut self, data: &ArrayView2<T>, indices: &[usize]) -> Option<(u32, T)>;
}

/// The original Isolation Forest splitter (Liu et al. 2008): sample a random
/// feature, then sample the split value from a uniform distribution between
/// the minimum and maximum values of the feature in the current subsample.
///
/// Features are drawn without replacement until a splittable (non-constant)
/// one is found; this is why a mutable buffer of feature indices is kept:
/// it is partially reshuffled (Fisher-Yates) on each call, which keeps the
/// draws uniform while avoiding re-initialization.
pub(crate) struct ItreeSplitter {
    rng: Xoshiro256PlusPlus,
    features: Vec<u32>,
}

impl ItreeSplitter {
    pub(crate) fn new(rng: Xoshiro256PlusPlus, n_features: u32) -> Self {
        Self {
            rng,
            features: (0..n_features).collect(),
        }
    }
}

impl<T> Splitter<T> for ItreeSplitter
where
    T: Copy + PartialOrd + SampleUniform,
{
    fn choose_split(&mut self, data: &ArrayView2<T>, indices: &[usize]) -> Option<(u32, T)> {
        for k in 0..self.features.len() {
            let j = self.rng.random_range(k..self.features.len());
            self.features.swap(k, j);
            let feature = self.features[k];

            // SAFETY: indices are valid row indices and feature < data.ncols()
            let (min, max) = indices
                .iter()
                .map(|&i| *unsafe { data.uget([i, feature as usize]) })
                .minmax()
                .into_option()
                .expect("indices must not be empty");

            if min < max {
                // Sample from [min, max); both partitions are then non-empty
                let value = Uniform::new(min, max)
                    .expect("min < max is guaranteed")
                    .sample(&mut self.rng);
                return Some((feature, value));
            }
        }
        None
    }
}

/// A pending node-splitting job: build a node at `depth` from the subsample
/// indices in `slice`, a disjoint part of the tree's subsample index array.
///
/// The task queue is FIFO and node indices are allocated in the enqueue
/// order, so tasks are processed in the node index order and the node built
/// by the i-th task is the i-th node of the tree.
struct Task<'a> {
    depth: u16,
    slice: &'a mut [usize],
}

impl Tree {
    /// Build a single isolation tree from a random subsample of `data` rows.
    pub(crate) fn build<T>(
        data: &ArrayView2<T>,
        seed: u64,
        n_subsamples: usize,
        max_depth: u16,
    ) -> Tree
    where
        T: Copy + PartialOrd + SampleUniform + AsPrimitive<f64>,
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        // Subsample without replacement
        let indices: Vec<usize> =
            rand::seq::index::sample(&mut rng, data.nrows(), n_subsamples).into_vec();
        let splitter = ItreeSplitter::new(rng, data.ncols() as u32);

        Self::build_with_splitter(data, indices, max_depth, splitter)
    }

    /// Build a single tree from the given subsample of `data` rows, using
    /// a custom splitting logic.
    pub(crate) fn build_with_splitter<T, S>(
        data: &ArrayView2<T>,
        mut indices: Vec<usize>,
        max_depth: u16,
        mut splitter: S,
    ) -> Tree
    where
        T: Copy + PartialOrd + AsPrimitive<f64>,
        S: Splitter<T>,
    {
        let n_subsamples = indices.len();
        let n_nodes_max = 2 * n_subsamples - 1;

        let mut nodes: Vec<Node> = Vec::with_capacity(n_nodes_max);
        // Kept on the side during the build, converted to the average path
        // length sidecar array afterwards
        let mut n_node_samples: Vec<u32> = Vec::with_capacity(n_nodes_max);
        let mut n_leaves: u32 = 0;
        // The root takes index 0, its task is the first in the queue
        let mut next_node_index: u32 = 1;

        let mut queue: VecDeque<Task> = VecDeque::new();
        queue.push_back(Task {
            depth: 0,
            slice: indices.as_mut_slice(),
        });

        // Tasks are processed in the node index order (see `Task` docs),
        // so the nodes are simply pushed one by one
        while let Some(Task { depth, slice }) = queue.pop_front() {
            n_node_samples.push(slice.len() as u32);

            let split = if depth < max_depth && slice.len() >= 2 {
                splitter.choose_split(data, slice)
            } else {
                None
            };

            match split {
                None => {
                    nodes.push(Node::Leaf(Leaf {
                        leaf_index: n_leaves,
                        value: depth as f64 + average_path_length::<_, f64>(slice.len()),
                    }));
                    n_leaves += 1;
                }
                Some((feature, value)) => {
                    nodes.push(Node::Split(SplitNode {
                        left_node_index: NonZeroU32::new(next_node_index)
                            .expect("node indices start from 1"),
                        split_feature: feature,
                        // Exact value of T is representable in f64 for both f32 and f64
                        split_value: value.as_(),
                    }));
                    next_node_index += 2;

                    // SAFETY: indices are valid row indices and feature < data.ncols()
                    let mid = itertools::partition(slice.iter_mut(), |&i| {
                        *unsafe { data.uget([i, feature as usize]) } <= value
                    });
                    let (left_slice, right_slice) = slice.split_at_mut(mid);
                    queue.push_back(Task {
                        depth: depth + 1,
                        slice: left_slice,
                    });
                    queue.push_back(Task {
                        depth: depth + 1,
                        slice: right_slice,
                    });
                }
            }
        }

        // The tree may be shallower than the upper limit we reserved for
        nodes.shrink_to_fit();

        let node_average_path_length = n_node_samples
            .iter()
            .map(|&n| average_path_length(n))
            .collect();

        Tree {
            nodes,
            node_average_path_length,
            n_leaves,
            n_subsamples,
            n_features: data.ncols() as u32,
        }
    }
}

fn build_trees_impl<T>(
    py: Python<'_>,
    data: &Bound<'_, PyArray2<T>>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    max_depth: usize,
    num_threads: usize,
) -> PyResult<Vec<Tree>>
where
    T: Element + Copy + Send + Sync + PartialOrd + SampleUniform + AsPrimitive<f64>,
{
    let data = data.readonly();
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
    if 2 * n_subsamples - 1 > u32::MAX as usize || data_view.ncols() > u32::MAX as usize {
        return Err(PyValueError::new_err("data is too large"));
    }
    if data_view.ncols() == 0 {
        return Err(PyValueError::new_err("data must have at least one feature"));
    }
    // A deeper tree is not possible: every split isolates at least one sample
    let max_depth = u16::try_from(max_depth)
        .map_err(|_| PyValueError::new_err(format!("max_depth must not exceed {}", u16::MAX)))?;

    // Sample random seeds for all the tree building jobs in advance, so the
    // result does not depend on the number of threads
    let mut master_rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let seeds: Vec<u64> = (0..n_trees).map(|_| master_rng.next_u64()).collect();

    let trees: Vec<Tree> = py.detach(|| {
        if num_threads == 1 {
            seeds
                .iter()
                .map(|&seed| Tree::build(&data_view, seed, n_subsamples, max_depth))
                .collect()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("Cannot build rayon ThreadPool")
                .install(|| {
                    seeds
                        .par_iter()
                        .map(|&seed| Tree::build(&data_view, seed, n_subsamples, max_depth))
                        .collect()
                })
        }
    });

    Ok(trees)
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
) -> PyResult<Vec<Tree>> {
    match &data {
        Data::F64(array) => build_trees_impl(
            py,
            array,
            seed,
            n_trees,
            n_subsamples,
            max_depth,
            num_threads,
        ),
        Data::F32(array) => build_trees_impl(
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
