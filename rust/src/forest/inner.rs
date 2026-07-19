//! Implementation details of the Forest class

use crate::tree::TreeInner;
use std::sync::Arc;
use std::sync::OnceLock;

#[derive(Clone)]
pub(super) enum ForestVariant {
    F32(ForestInner<f32>),
    F64(ForestInner<f64>),
}

impl ForestVariant {
    /// Return the numpy dtype name.
    pub(super) fn dtype_str(&self) -> &'static str {
        match self {
            ForestVariant::F32(_) => "float32",
            ForestVariant::F64(_) => "float64",
        }
    }
}

pub(super) struct ForestInner<T> {
    trees: Vec<Arc<TreeInner<T>>>,
    n_features: u32,
    num_threads: usize,
    leaf_offsets: OnceLock<Vec<usize>>,
    thread_pool: OnceLock<Option<rayon::ThreadPool>>,
}

impl<T> ForestInner<T> {
    pub(crate) fn new(n_features: u32, num_threads: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_features,
            num_threads,
            leaf_offsets: OnceLock::new(),
            thread_pool: OnceLock::new(),
        }
    }

    pub(crate) fn with_thread_pool(
        trees: Vec<Arc<TreeInner<T>>>,
        n_features: u32,
        thread_pool: Option<rayon::ThreadPool>,
    ) -> Self {
        let num_threads = match &thread_pool {
            Some(thread_pool) => thread_pool.current_num_threads(),
            None => 1,
        };
        Self {
            trees,
            n_features,
            num_threads,
            leaf_offsets: OnceLock::new(),
            thread_pool: OnceLock::from(thread_pool),
        }
    }

    pub(super) fn trees(&self) -> &[Arc<TreeInner<T>>] {
        &self.trees
    }

    pub(crate) fn trees_mut(&mut self) -> &mut Vec<Arc<TreeInner<T>>> {
        self.leaf_offsets.take();
        &mut self.trees
    }

    pub(crate) fn n_features(&self) -> u32 {
        self.n_features
    }

    pub(crate) fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub(crate) fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;
        self.thread_pool.take();
    }

    fn init_leaf_offsets(trees: &[Arc<TreeInner<T>>]) -> Vec<usize> {
        let mut leaf_offsets = Vec::with_capacity(trees.len() + 1);
        let mut offset = 0;
        for tree in trees {
            leaf_offsets.push(offset);
            offset += tree.n_leaves() as usize;
        }
        leaf_offsets.push(offset);
        leaf_offsets
    }

    fn leaf_offsets(&self) -> &[usize] {
        self.leaf_offsets
            .get_or_init(|| Self::init_leaf_offsets(&self.trees))
    }

    pub(crate) fn n_leaves(&self) -> usize {
        // When no trees are present, offsets are [0]
        *self.leaf_offsets().last().unwrap()
    }
    pub(super) fn init_thread_pool(n_jobs: usize) -> Option<rayon::ThreadPool> {
        if n_jobs == 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_jobs)
                    .build()
                    .unwrap(),
            )
        }
    }

    pub(super) fn thread_pool(&self) -> Option<&rayon::ThreadPool> {
        self.thread_pool
            .get_or_init(|| Self::init_thread_pool(self.num_threads))
            .as_ref()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&Arc<TreeInner<T>>, usize)> {
        // Borrow the trees rather than cloning the `Arc`s: this iterator runs
        // once per sample inside the parallel scoring loop, and cloning would
        // hammer the shared atomic refcounts across threads.
        self.trees.iter().zip(self.leaf_offsets().iter().cloned())
    }

    pub(crate) fn get(&self, index: usize) -> Option<Arc<TreeInner<T>>> {
        self.trees.get(index).cloned()
    }

    pub(crate) fn try_remove_tree(&mut self, index: usize) -> Option<Arc<TreeInner<T>>> {
        self.leaf_offsets.take();
        if index >= self.trees.len() {
            None
        } else {
            Some(self.trees.remove(index))
        }
    }
}

impl<T> Clone for ForestInner<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            trees: self.trees.clone(),
            n_features: self.n_features,
            num_threads: self.num_threads,
            leaf_offsets: self.leaf_offsets.clone(),
            thread_pool: OnceLock::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Build a forest of `n_trees` f64 trees on random data with `n_features`
    /// columns. Uses [ForestInner] directly to keep the test free of the
    /// pyo3-returning constructor (which cannot link into a test binary).
    fn build_forest(n_trees: usize, n_features: u32) -> ForestInner<f64> {
        let n_subsamples = 64;
        let n_rows = 4 * n_subsamples;
        let mut rng = StdRng::seed_from_u64(0);
        let trees = (0..n_trees)
            .map(|_| {
                let data =
                    Array2::<f64>::from_shape_simple_fn((n_rows, n_features as usize), || {
                        rng.random::<f64>()
                    });
                Arc::new(TreeInner::build(&data.view(), n_subsamples, 8, &mut rng))
            })
            .collect();
        ForestInner::with_thread_pool(trees, n_features, None)
    }

    /// Every tree's local leaf indices, shifted by its global offset, must map
    /// onto a distinct slot in `0..n_leaves` with no gaps or overlaps, i.e. the
    /// offsets tile the global leaf-index space (regression test for a
    /// shifted-offset bug where consecutive trees shared the same offset).
    #[test]
    fn leaf_offsets_tile_leaf_index_space() {
        let forest = build_forest(5, 3);

        let total = forest.n_leaves();
        // Sanity: several trees each with several leaves, so tiling is non-trivial.
        assert!(total > forest.trees().len());

        let mut seen = vec![false; total];
        for (tree, offset) in forest.iter() {
            for local in 0..tree.n_leaves() as usize {
                let global = offset + local;
                assert!(
                    global < total,
                    "global leaf id {global} out of range 0..{total}"
                );
                assert!(
                    !seen[global],
                    "global leaf id {global} assigned to two trees"
                );
                seen[global] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "some global leaf ids were never assigned"
        );
    }

    /// An empty forest has no leaves and its offset table is well-formed.
    #[test]
    fn empty_forest_has_no_leaves() {
        let forest = ForestInner::<f64>::new(3, 1);
        assert_eq!(forest.n_leaves(), 0);
        assert_eq!(forest.iter().count(), 0);
    }
}
