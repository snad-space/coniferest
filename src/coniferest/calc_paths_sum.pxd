cimport numpy as np

# Selector is the representation of decision tree nodes:
# either branches or leafs.
cdef packed struct selector_t:
    # Feature index to branch on
    np.int32_t feature
    # Index of left subtree
    np.int32_t left
    # Feature value to branch on
    np.float64_t value
    # Index of right subtree
    np.int32_t right
    # Log of samples count covered by this node
    np.float32_t log_n_node_samples

    # In case of leaf the fields have special values:
    # * feature == -1
    # * left == leaf_id
    # * value == resulting decision score
    # * right == -1


# For bitness polymorphism
ctypedef fused floating:
    np.float32_t
    np.float64_t


# Calculate paths sum on data based on trees,
# that are located by indeces in selectors list
cdef void _paths_sum(selector_t [::1] selectors,
                     np.int64_t [::1] indices,
                     floating [:, ::1] data,
                     np.float64_t [::1] paths,
                     np.float64_t [::1] weights,
                     int num_threads,
                     int chunksize)

cdef void _paths_sum_transpose(selector_t [::1] selectors,
                               np.int64_t [::1] indices,
                               floating [:, ::1] data,
                               np.float64_t [::1] values,
                               np.float64_t [::1] weights,
                               int num_threads,
                               int chunksize)

cdef void _feature_delta_sum(selector_t [::1] selectors,
                             np.int64_t [::1] indices,
                             floating [:, ::1] data,
                             np.float64_t [:, ::1] delta_sum,
                             np.int64_t [:, ::1] hit_count,
                             int num_threads,
                             int chunksize)

cdef void _apply(selector_t [::1] selectors,
                 np.int64_t [::1] indices,
                 floating [:, ::1] data,
                 np.int64_t [:, ::1] leafs,
                 int num_threads,
                 int chunksize)
