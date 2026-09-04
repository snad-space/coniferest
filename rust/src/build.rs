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
use rayon::prelude::*;
use std::mem::MaybeUninit;
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

/// A pending node-splitting job: build the node at `node_index` and `depth`
/// from the subsample indices in `slice`, a disjoint part of the tree's
/// subsample index array.
///
/// `node_index` is the node's final position in the tree; it is allocated by
/// the parent (see [NodesBuilder]) before the task is enqueued. Tasks are run
/// from a LIFO stack, so the build is depth-first and tasks are *not* processed
/// in node-index order.
struct Task<'a> {
    node_index: usize,
    depth: u16,
    slice: &'a mut [usize],
}

/// Grow-only arena of tree nodes that are allocated and filled in separate
/// steps. During a depth-first build a node's index is allocated by its parent
/// (via [`push_uninit`](Self::push_uninit)) before the node itself is known, so
/// slots are written out of order by [`insert`](Self::insert). Uninitialized
/// slots are held as [`MaybeUninit`], which avoids requiring `N: Default`.
struct NodesBuilder<T> {
    nodes: Vec<MaybeUninit<Node<T>>>,
    n_node_samples: Vec<f32>,
}

impl<T> NodesBuilder<T> {
    /// Create a builder with room for `capacity` nodes and the root slot
    /// (index 0) already allocated.
    fn with_capacity(capacity: usize) -> Self {
        let mut slf = Self {
            nodes: Vec::with_capacity(capacity),
            n_node_samples: Vec::with_capacity(capacity),
        };
        slf.push_uninit();
        slf
    }

    /// Allocate a new, still-uninitialized node slot and return its index.
    fn push_uninit(&mut self) -> usize {
        let new_last_index = self.len();
        self.nodes.push(MaybeUninit::uninit());
        self.n_node_samples.push(0.0);
        new_last_index
    }

    /// Write `node` into a previously allocated slot.
    ///
    /// # Safety
    /// `index` must be a slot allocated by [`push_uninit`](Self::push_uninit).
    /// Overwriting an already-initialized slot leaks it rather than dropping,
    /// but is otherwise sound.
    unsafe fn insert(&mut self, index: usize, node: Node<T>, n_samples: usize) {
        let node = MaybeUninit::new(node);
        let n_samples = n_samples as f32;
        unsafe { *self.nodes.as_mut_ptr().add(index) = node };
        unsafe { *self.n_node_samples.as_mut_ptr().add(index) = n_samples };
    }

    /// Consume the builder and return the nodes as a plain `Vec<N>`.
    ///
    /// # Safety
    /// Every slot allocated by [`push_uninit`](Self::push_uninit) must have
    /// been initialized by [`insert`](Self::insert).
    unsafe fn build(self) -> (Vec<Node<T>>, Vec<f32>, u32) {
        // `MaybeUninit<N>` has the same layout as `N`, so once every element is
        //  initialized, the two vectors are layout-compatible.
        let mut nodes =
            unsafe { std::mem::transmute::<Vec<MaybeUninit<Node<T>>>, Vec<Node<T>>>(self.nodes) };
        nodes.shrink_to_fit();

        let mut leaf_index = 0;
        for node in nodes.iter_mut() {
            match node {
                Node::Leaf(leaf) => {
                    leaf.leaf_index = leaf_index;
                    leaf_index += 1;
                }
                Node::Split(_) => {}
            }
        }

        (nodes, self.n_node_samples, leaf_index)
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
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

        let mut builder = NodesBuilder::with_capacity(node_vector_capacity);

        // LIFO queue of tasks: the first task is the root node, and tasks are being added in the
        // reverse order. This makes the build happening depth-first, so we need up to
        // max_depth + 1 tasks scheduled
        let mut queue: Vec<Task> = Vec::with_capacity(max_depth as usize + 1);
        queue.push(Task {
            node_index: 0,
            depth: 0,
            slice: indices.as_mut_slice(),
        });

        // Depth-first (see `Task` docs): each node is written to its own
        // pre-allocated `node_index`, not in the order tasks are processed
        while let Some(Task {
            node_index,
            depth,
            slice,
        }) = queue.pop()
        {
            let split = if depth < max_depth && slice.len() >= 2 {
                splitter.choose_split(data, slice, &mut rng)
            } else {
                None
            };

            match split {
                None => {
                    // SAFETY: we pre-allocated this node_index
                    unsafe {
                        builder.insert(
                            node_index,
                            Node::Leaf(Leaf {
                                // Temporary: we will re-assign bellow
                                leaf_index: 0,
                                value: depth as f32 + average_path_length::<_, f32>(slice.len()),
                            }),
                            slice.len(),
                        )
                    };
                }
                Some((feature, value)) => {
                    let left_node_index = builder.push_uninit();
                    let right_node_index = builder.push_uninit();

                    // SAFETY: we pre-allocated this node_index when push_uinit() was called,
                    // either in the previous iteration or in the root node's allocation.
                    unsafe {
                        builder.insert(node_index, Node::Split(SplitNode {
                        left_node_index: NonZeroU32::new(left_node_index.try_into().expect("number of nodes exceeded 2^32, please specify smaller max_depth or request support for larger trees")).unwrap(),
                        split_feature: feature,
                        split_value: value,
                    }), slice.len());
                    };

                    let mid = itertools::partition(slice.iter_mut(), |&i| {
                        // SAFETY: indices are valid row indices and feature < data.ncols()
                        *unsafe { data.uget([i, feature as usize]) } <= value
                    });
                    let (left_slice, right_slice) = slice.split_at_mut(mid);
                    queue.push(Task {
                        node_index: right_node_index,
                        depth: depth + 1,
                        slice: right_slice,
                    });
                    queue.push(Task {
                        node_index: left_node_index,
                        depth: depth + 1,
                        slice: left_slice,
                    });
                }
            }
        }

        // SAFETY: the node vector is fully initialized, and we can safely build it.
        let (nodes, n_node_samples, n_leaves) = unsafe { builder.build() };

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

#[cfg(all(test, coniferest_nightly_bench))]
mod bench_alloc {
    use super::*;
    use ndarray::Array2;
    use test::Bencher;

    fn make_data() -> Array2<f64> {
        let n_samples = 16_384usize;
        let n_features = 16usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let v: Vec<f64> = (0..n_samples * n_features)
            .map(|_| rng.random::<f64>())
            .collect();
        Array2::from_shape_vec((n_samples, n_features), v).unwrap()
    }

    /// Run with
    /// RUSTFLAGS="--cfg coniferest_nightly_bench" cargo +nightly bench --manifest-path rust/Cargo.toml -- --nocapture bench_build_trees
    #[bench]
    fn bench_build_trees(b: &mut Bencher) {
        let n_subsamples = 1024usize;
        let max_depth: u16 = 10;
        let data = make_data();
        let view = data.view();

        let info = allocation_counter::measure(|| {
            let rng = Xoshiro256PlusPlus::seed_from_u64(0);
            let _tree = TreeInner::build(&view, n_subsamples, max_depth, rng);
        });
        println!(
            "\n[alloc] {} allocations, {} bytes total, {} bytes peak",
            info.count_total, info.bytes_total, info.bytes_max,
        );

        b.iter(|| {
            let view = std::hint::black_box(&view);
            let rng = Xoshiro256PlusPlus::seed_from_u64(0);
            TreeInner::build(view, n_subsamples, max_depth, rng)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    struct EqualHalfSplitter;

    impl SplitAlgorithm<f64> for EqualHalfSplitter {
        fn choose_split(
            &mut self,
            data: &ArrayView2<f64>,
            indices: &[usize],
            _rng: &mut impl Rng,
        ) -> Option<(u32, f64)> {
            let feature = 0;
            let (min, max) = indices
                .iter()
                .map(|&i| data[[i, feature as usize]])
                .minmax()
                .into_option()?;
            if min == max {
                return None;
            }
            let split_value = (min + max) / 2.0;
            Some((feature, split_value))
        }
    }

    /// Test that the tree building algorithm works correctly on a simple 1-d dataset and
    /// deterministic equal-half splitter.
    #[test]
    fn build_algorithm() {
        let max_depth = 4;
        let n_samples = 1 << max_depth as usize;
        let data = (0..n_samples)
            .map(|i| i as f64)
            .collect::<Array1<_>>()
            .into_shape_with_order((n_samples, 1))
            .unwrap();
        let tree = TreeInner::build_with_splitter(
            &data.view(),
            (0..n_samples).collect(),
            max_depth,
            EqualHalfSplitter,
            // not used
            rand::rngs::StdRng::seed_from_u64(0),
        );
        assert_eq!(tree.n_leaves, 16);
        assert_eq!(tree.n_subsamples, 16);
        assert_eq!(tree.n_features, 1);
        assert_eq!(tree.nodes.len(), (2 << max_depth as usize) - 1);
        assert_eq!(
            tree.node_average_path_length.len(),
            (2 << max_depth as usize) - 1
        );

        // Using visitor pattern check that paths are correct
        {
            // Smaller than the smallest value
            let mut n_samples_left = n_samples;
            tree.for_each_split(&[-1.0], |_node_index, split, child_index| {
                // We should go left each time
                assert_eq!(child_index % 2, 1);
                assert_eq!(split.split_feature, 0);
                assert_eq!(split.split_value, (n_samples_left as f64 - 1.0) / 2.0);
                n_samples_left /= 2;
            });
        }
        {
            // Larger than the largest value
            let mut n_samples_left = n_samples;
            tree.for_each_split(&[n_samples as f64], |_node_index, split, child_index| {
                // We should go right each time
                assert_eq!(child_index % 2, 0);
                assert_eq!(split.split_feature, 0);
                assert_eq!(
                    split.split_value,
                    (n_samples as f64 - 1.0) - (n_samples_left as f64 - 1.0) / 2.0
                );
                n_samples_left /= 2;
            });
        }
        {
            // Median value
            let mut going_left = true;
            let median = (n_samples as f64 - 1.0) / 2.0;
            tree.for_each_split(&[median], |_node_index, split, child_index| {
                // We go left first time, and go right each time after
                assert_eq!(child_index % 2, going_left as usize);
                going_left = false;
                assert_eq!(split.split_feature, 0);
            });
        }

        // node_average_path_length holds average_path_length of each node's
        // subsample size. This complete tree has 2^d nodes of n_samples >> d
        // samples at every depth d.
        let mut expected: Vec<f32> = (0..=max_depth)
            .flat_map(|depth| {
                let count = n_samples >> depth;
                vec![average_path_length::<_, f32>(count); 1 << depth]
            })
            .collect();
        let mut actual = tree.node_average_path_length.clone();
        expected.sort_by(f32::total_cmp);
        actual.sort_by(f32::total_cmp);
        assert_eq!(actual, expected);
    }

    /// Check that leaf indices are a perturbation of 0..n_samples
    #[test]
    fn leaf_indices() {
        let max_depth = 4;
        let n_samples = 1 << max_depth as usize;
        let data = (0..n_samples)
            .map(|i| i as f64)
            .collect::<Array1<_>>()
            .into_shape_with_order((n_samples, 1))
            .unwrap();
        let tree = TreeInner::build_with_splitter(
            &data.view(),
            (0..n_samples).collect(),
            max_depth,
            EqualHalfSplitter,
            // not used
            rand::rngs::StdRng::seed_from_u64(0),
        );

        let mut leaf_indices: Vec<u32> = tree
            .nodes
            .iter()
            .filter_map(|node| match node {
                Node::Leaf(leaf) => Some(leaf.leaf_index),
                Node::Split(_) => None,
            })
            .collect();
        assert_eq!(leaf_indices.len(), tree.n_leaves as usize);

        leaf_indices.sort_unstable();
        assert_eq!(leaf_indices, (0..tree.n_leaves).collect::<Vec<_>>());
    }
}
