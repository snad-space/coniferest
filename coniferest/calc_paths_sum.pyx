import numpy as np
from cython.parallel cimport prange, parallel

cimport numpy as np
cimport cython


def calc_paths_sum(selector_t [::1] selectors,
                   np.int64_t [::1] indices,
                   floating [:, ::1] data,
                   np.float64_t [::1] weights=None,
                   int num_threads=1):
    cdef np.ndarray [np.double_t, ndim=1] paths = np.zeros(data.shape[0])
    cdef np.float64_t [::1] paths_view = paths
    cdef Py_ssize_t sellen = selectors.shape[0]

    if np.any(np.diff(indices) < 0):
        raise ValueError('indices should be an increasing sequence')

    if indices[-1] > sellen:
        raise ValueError('indices are out of range of the selectors')

    _paths_sum(selectors, indices, data, paths_view, weights, num_threads)
    return paths


def calc_paths_sum_transpose(selector_t [::1] selectors,
                             np.int64_t [::1] indices,
                             floating [:, ::1] data,
                             Py_ssize_t leaf_count,
                             np.float64_t [::1] weights=None,
                             int num_threads=1):
    cdef np.ndarray [np.double_t, ndim=1] values = np.zeros(leaf_count)
    cdef np.float64_t [::1] values_view = values
    cdef Py_ssize_t sellen = selectors.shape[0]

    if np.any(np.diff(indices) < 0):
        raise ValueError('indices should be an increasing sequence')

    if indices[-1] > sellen:
        raise ValueError('indices are out of range of the selectors')

    if weights is not None and weights.shape[0] != data.shape[0]:
        raise ValueError('data and weights should have the same length')

    _paths_sum_transpose(selectors, indices, data, values_view, weights, num_threads)
    return values


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _paths_sum(selector_t [::1] selectors,
                     np.int64_t [::1] indices,
                     floating [:, ::1] data,
                     np.float64_t [::1] paths,
                     np.float64_t [::1] weights=None,
                     int num_threads=1):

    cdef Py_ssize_t trees
    cdef Py_ssize_t tree_index
    cdef Py_ssize_t x_index
    cdef selector_t selector
    cdef Py_ssize_t tree_offset
    cdef np.int32_t feature, i

    with nogil, parallel(num_threads=num_threads):
        trees = indices.shape[0] - 1

        for x_index in prange(data.shape[0], schedule='static'):
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
                               np.float64_t [::1] weights=None,
                               int num_threads=1):

    cdef Py_ssize_t trees
    cdef Py_ssize_t tree_index
    cdef Py_ssize_t x_index
    cdef selector_t selector
    cdef Py_ssize_t tree_offset
    cdef np.int32_t feature, i

    with nogil, parallel(num_threads=num_threads):
        trees = indices.shape[0] - 1

        for tree_index in prange(trees, schedule='static'):
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
