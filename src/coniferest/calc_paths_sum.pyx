# cython: profile=True

import numpy as np
from cython.parallel cimport prange, parallel

cimport numpy as np
cimport cython
cimport openmp


def calc_paths_sum(selector_t [::1] selectors,
                   np.int64_t [::1] indices,
                   floating [:, ::1] data,
                   np.float64_t [::1] weights=None,
                   int num_threads=1,
                   int chunksize=0):
    cdef np.ndarray [np.double_t, ndim=1] paths = np.zeros(data.shape[0])
    cdef np.float64_t [::1] paths_view = paths
    cdef Py_ssize_t sellen = selectors.shape[0]

    if np.any(np.diff(indices) < 0):
        raise ValueError('indices should be an increasing sequence')

    if indices[-1] > sellen:
        raise ValueError('indices are out of range of the selectors')

    if num_threads < 0:
        num_threads = openmp.omp_get_max_threads()

    _paths_sum(selectors, indices, data, paths_view, weights, num_threads, chunksize)
    return paths


def calc_paths_sum_transpose(selector_t [::1] selectors,
                             np.int64_t [::1] indices,
                             floating [:, ::1] data,
                             Py_ssize_t leaf_count,
                             np.float64_t [::1] weights=None,
                             int num_threads=1,
                             int chunksize=0):
    cdef np.ndarray [np.double_t, ndim=1] values = np.zeros(leaf_count)
    cdef np.float64_t [::1] values_view = values
    cdef Py_ssize_t sellen = selectors.shape[0]

    if np.any(np.diff(indices) < 0):
        raise ValueError('indices should be an increasing sequence')

    if indices[-1] > sellen:
        raise ValueError('indices are out of range of the selectors')

    if weights is not None and weights.shape[0] != data.shape[0]:
        raise ValueError('data and weights should have the same length')

    _paths_sum_transpose(selectors, indices, data, values_view, weights, num_threads, chunksize)
    return values


def calc_feature_delta_sum(selector_t [::1] selectors,
                   np.int64_t [::1] indices,
                   floating [:, ::1] data,
                   int num_threads=1,
                   int chunksize=0):
    cdef np.ndarray [np.double_t, ndim=2] delta_sum = np.zeros([data.shape[0], data.shape[1]])
    cdef np.float64_t [:, ::1] delta_sum_view = delta_sum
    cdef np.ndarray [np.int64_t, ndim=2] hit_count = np.zeros([data.shape[0], data.shape[1]], dtype=np.int64)
    cdef np.int64_t [:, ::1] hit_count_view = hit_count
    cdef Py_ssize_t sellen = selectors.shape[0]

    if np.any(np.diff(indices) < 0):
        raise ValueError('indices should be an increasing sequence')

    if indices[-1] > sellen:
        raise ValueError('indices are out of range of the selectors')

    _feature_delta_sum(selectors, indices, data, delta_sum_view, hit_count_view, num_threads, chunksize)
    return delta_sum, hit_count

def calc_apply(selector_t [::1] selectors, np.int64_t [::1] indices, floating [:, ::1] data, int num_threads=1, int chunksize=0):
    cdef np.ndarray [np.int64_t, ndim=2] leafs = np.zeros([data.shape[0], indices.shape[0] - 1], dtype=np.int64)
    cdef np.int64_t [:, ::1] leafs_view = leafs
    cdef Py_ssize_t sellen = selectors.shape[0]

    if np.any(np.diff(indices) < 0):
        raise ValueError('indices should be an increasing sequence')

    if indices[-1] > sellen:
        raise ValueError('indices are out of range of the selectors')

    _apply(selectors, indices, data, leafs_view, num_threads, chunksize)
    return leafs



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _paths_sum(selector_t [::1] selectors,
                     np.int64_t [::1] indices,
                     floating [:, ::1] data,
                     np.float64_t [::1] paths,
                     np.float64_t [::1] weights,
                     int num_threads,
                     int chunksize):

    cdef Py_ssize_t trees
    cdef Py_ssize_t tree_index
    cdef Py_ssize_t x_index
    cdef selector_t selector
    cdef Py_ssize_t tree_offset
    cdef np.int32_t feature, i
    cdef int use_threads_if = (2 * num_threads < data.shape[0])

    with nogil, parallel(num_threads=num_threads, use_threads_if=use_threads_if):
        trees = indices.shape[0] - 1

        for x_index in prange(data.shape[0], schedule='static', chunksize=chunksize):
            for tree_index in range(trees):
                tree_offset = indices[tree_index]
                i = 0
                while True:
                    selector = selectors[tree_offset + i]
                    feature = selector.feature
                    if feature < 0:
                        break

                    if data[x_index, feature] <= selector.value:
                        i = selector.left
                    else:
                        i = selector.right

                if weights is None:
                    paths[x_index] += selector.value
                else:
                    paths[x_index] += weights[selector.left] * selector.value


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _paths_sum_transpose(selector_t [::1] selectors,
                               np.int64_t [::1] indices,
                               floating [:, ::1] data,
                               np.float64_t [::1] values,
                               np.float64_t [::1] weights,
                               int num_threads,
                               int chunksize):

    cdef Py_ssize_t trees
    cdef Py_ssize_t tree_index
    cdef Py_ssize_t x_index
    cdef selector_t selector
    cdef Py_ssize_t tree_offset
    cdef np.int32_t feature, i

    with nogil, parallel(num_threads=num_threads):
        trees = indices.shape[0] - 1

        for tree_index in prange(trees, schedule='static', chunksize=chunksize):
            tree_offset = indices[tree_index]
            for x_index in range(data.shape[0]):
                i = 0
                while True:
                    selector = selectors[tree_offset + i]
                    feature = selector.feature
                    if feature < 0:
                        break

                    if data[x_index, feature] <= selector.value:
                        i = selector.left
                    else:
                        i = selector.right

                if weights is None:
                    values[selector.left] += selector.value
                else:
                    values[selector.left] += weights[x_index] * selector.value


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _feature_delta_sum(selector_t [::1] selectors,
                     np.int64_t [::1] indices,
                     floating [:, ::1] data,
                     np.float64_t [:, ::1] delta_sum,
                     np.int64_t [:, ::1] hit_count,
                     int num_threads,
                     int chunksize):

    cdef Py_ssize_t trees
    cdef Py_ssize_t tree_index
    cdef Py_ssize_t x_index
    cdef selector_t selector, child_selector
    cdef Py_ssize_t tree_offset
    cdef np.int32_t feature, i

    with nogil, parallel(num_threads=num_threads):
        trees = indices.shape[0] - 1

        for x_index in prange(data.shape[0], schedule='static', chunksize=chunksize):
            for tree_index in range(trees):
                tree_offset = indices[tree_index]
                i = 0
                while True:
                    selector = selectors[tree_offset + i]
                    feature = selector.feature
                    if feature < 0:
                        break

                    if data[x_index, feature] <= selector.value:
                        i = selector.left
                    else:
                        i = selector.right

                    child_selector = selectors[tree_offset + i]

                    delta_sum[x_index, feature] += 1.0 + 2.0 * (child_selector.log_n_node_samples - selector.log_n_node_samples)
                    hit_count[x_index, feature] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _apply(selector_t [::1] selectors,
                 np.int64_t [::1] indices,
                 floating [:, ::1] data,
                 np.int64_t [:, ::1] leafs,
                 int num_threads,
                 int chunksize):

    cdef Py_ssize_t trees
    cdef Py_ssize_t tree_index
    cdef Py_ssize_t x_index
    cdef selector_t selector
    cdef Py_ssize_t tree_offset
    cdef np.int32_t feature, i

    with nogil, parallel(num_threads=num_threads):
        trees = indices.shape[0] - 1

        for x_index in prange(data.shape[0], schedule='static', chunksize=chunksize):
            for tree_index in range(trees):
                tree_offset = indices[tree_index]
                i = 0
                while True:
                    selector = selectors[tree_offset + i]
                    feature = selector.feature
                    if feature < 0:
                        break

                    if data[x_index, feature] <= selector.value:
                        i = selector.left
                    else:
                        i = selector.right

                leafs[x_index, tree_index] = selector.left
