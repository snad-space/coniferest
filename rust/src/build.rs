use crate::tree::{Leaf, Node, SplitNode, Tree, TreeDtype, TreeInner};
use crate::tree_traversal::Data;
use crate::utils::average_path_length;
use itertools::Itertools;
use ndarray::ArrayView2;
use numpy::{Element, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, Python, pyfunction};
use rand::distr::uniform::SampleUniform;
use rand::distr::{Distribution, Uniform};
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::VecDeque;
use std::num::NonZeroU32;

/// Everything a [SplitAlgorithm] needs from `T` to choose a split.
pub(crate) trait SplitterValue: Copy + PartialOrd + SampleUniform {}

impl<T> SplitterValue for T where T: Copy + PartialOrd + SampleUniform {}

/// Node splitting logic, customizable per the tree kind.
pub(crate) trait SplitAlgorithm<T> {
    /// Choose the split feature and value for the subsample rows `indices`,
    /// or return `None` to make the node a leaf.
    ///
    /// The returned value must partition `indices` into two non-empty parts.
    fn choose_split(
        &mut self,
        data: &ArrayView2<T>,
        indices: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(u32, T)>;
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
    features: Vec<u32>,
}

impl ItreeSplitter {
    pub(crate) fn new(n_features: u32) -> Self {
        Self {
            features: (0..n_features).collect(),
        }
    }
}

impl<T> SplitAlgorithm<T> for ItreeSplitter
where
    T: SplitterValue,
{
    fn choose_split(
        &mut self,
        data: &ArrayView2<T>,
        indices: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(u32, T)> {
        for k in 0..self.features.len() {
            let j = rng.random_range(k..self.features.len());
            self.features.swap(k, j);
            let feature = self.features[k];

            let (min, max) = indices
                .iter()
                // SAFETY: indices are valid row indices and feature < data.ncols()
                .map(|&i| *unsafe { data.uget([i, feature as usize]) })
                .minmax()
                .into_option()
                .expect("indices must not be empty");

            if min < max {
                // Sample from [min, max); both partitions are then non-empty
                let value = Uniform::new(min, max)
                    .expect("min < max is guaranteed")
                    .sample(rng);
                return Some((feature, value));
            }
        }
        None
    }
}

/// The splitting algorithm to build a tree with.
///
/// One variant per [SplitAlgorithm] implementation; add a new variant here
/// whenever a new splitter is introduced, so tree building can pick the
/// algorithm at runtime instead of being generic over it.
pub(crate) enum Splitter {
    Itree(ItreeSplitter),
}

impl<T> SplitAlgorithm<T> for Splitter
where
    T: SplitterValue,
{
    fn choose_split(
        &mut self,
        data: &ArrayView2<T>,
        indices: &[usize],
        rng: &mut impl Rng,
    ) -> Option<(u32, T)> {
        match self {
            Splitter::Itree(splitter) => splitter.choose_split(data, indices, rng),
        }
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

impl<T> TreeInner<T>
where
    T: Copy + PartialOrd,
{
    /// Build a single isolation tree from a random subsample of `data` rows.
    pub(crate) fn build(
        data: &ArrayView2<T>,
        n_subsamples: usize,
        max_depth: u16,
        mut rng: impl Rng,
    ) -> Self
    where
        T: SampleUniform,
    {
        // Subsample without replacement
        let indices: Vec<usize> =
            rand::seq::index::sample(&mut rng, data.nrows(), n_subsamples).into_vec();
        let splitter = Splitter::Itree(ItreeSplitter::new(data.ncols() as u32));

        Self::build_with_splitter(data, indices, max_depth, splitter, rng)
    }

    /// Build a single tree from the given subsample of `data` rows, using
    /// a custom splitting logic.
    pub(crate) fn build_with_splitter<S>(
        data: &ArrayView2<T>,
        mut indices: Vec<usize>,
        max_depth: u16,
        mut splitter: S,
        mut rng: impl Rng,
    ) -> Self
    where
        S: SplitAlgorithm<T>,
    {
        let n_subsamples = indices.len();
        // The maximum number of nodes is the minimum of those for the balanced binary tree
        // (2^(max_depth+1)-1) and tree needed to store all the subsamples in the leaves,
        // (2 * n_subsamples - 1). Normally, the tree would be shallower than the maximum depth,
        // so we reserve approximately half of this estimate.
        let node_vector_capacity = usize::min(n_subsamples, 1 << max_depth as usize);

        let mut nodes: Vec<Node<T>> = Vec::with_capacity(node_vector_capacity);
        // Kept on the side during the build, converted to the average path
        // length sidecar array afterward
        let mut n_node_samples: Vec<u32> = Vec::with_capacity(node_vector_capacity);
        let mut n_leaves: u32 = 0;
        // The root takes index 0, its task is the first in the queue
        let mut next_node_index: NonZeroU32 = NonZeroU32::new(1).unwrap();

        let mut queue: VecDeque<Task> = VecDeque::with_capacity(node_vector_capacity);
        queue.push_back(Task {
            depth: 0,
            slice: indices.as_mut_slice(),
        });

        // Tasks are processed in the node index order (see `Task` docs),
        // so the nodes are simply pushed one by one
        while let Some(Task { depth, slice }) = queue.pop_front() {
            n_node_samples.push(slice.len() as u32);

            let split = if depth < max_depth && slice.len() >= 2 {
                splitter.choose_split(data, slice, &mut rng)
            } else {
                None
            };

            match split {
                None => {
                    nodes.push(Node::Leaf(Leaf {
                        leaf_index: n_leaves,
                        value: depth as f32 + average_path_length::<_, f32>(slice.len()),
                    }));
                    n_leaves += 1;
                }
                Some((feature, value)) => {
                    nodes.push(Node::Split(SplitNode {
                        left_node_index: next_node_index,
                        split_feature: feature,
                        split_value: value,
                    }));
                    next_node_index = next_node_index.checked_add(2).expect("number of nodes exceeded 2^32, please specify smaller max_depth or request support for larger trees");

                    let mid = itertools::partition(slice.iter_mut(), |&i| {
                        // SAFETY: indices are valid row indices and feature < data.ncols()
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

        TreeInner {
            nodes,
            node_average_path_length,
            n_leaves,
            n_subsamples,
            n_features: data.ncols() as u32,
        }
    }
}

/// Everything needed to build trees for a dtype exposed to Python: dtype
/// dispatch ([TreeDtype]), the numpy/threading plumbing, and the numeric
/// operations the splitters rely on ([SplitterValue]).
pub(crate) trait TreeBuildDtype: TreeDtype + SplitterValue + Element + Send + Sync {}

impl<T> TreeBuildDtype for T where T: TreeDtype + SplitterValue + Element + Send + Sync {}

fn build_trees_impl<T>(
    py: Python<'_>,
    data: &PyReadonlyArray2<'_, T>,
    seed: u64,
    n_trees: usize,
    n_subsamples: usize,
    max_depth: usize,
    num_threads: usize,
) -> PyResult<Vec<Tree>>
where
    T: TreeBuildDtype,
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
                .install(|| child_seeds_iter.map(tree_build_fn).collect())
        }
    });

    Ok(trees
        .into_iter()
        .map(|inner| Tree(T::wrap(inner)))
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
