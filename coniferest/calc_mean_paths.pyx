import numpy as np
from cython.parallel cimport prange, parallel

cimport numpy as np
cimport cython


def calc_mean_paths(selector_t [::1] selectors, np.int64_t [::1] indices, floating [:, ::1] data):
    cdef np.ndarray [np.double_t, ndim=1] paths = np.zeros(data.shape[0])
    cdef np.float64_t [::1] paths_view = paths
    cdef Py_ssize_t sellen = selectors.shape[0]

    if np.any(np.diff(indices) < 0):
        raise ValueError('indices should be an increasing sequence')

    if indices[-1] > sellen:
        raise ValueError('indices are out of range of the selectors')

    _mean_paths(selectors, indices, data, paths_view)
    return paths


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _mean_paths(selector_t [::1] selectors,
                         np.int64_t [::1] indices,
                         floating [:, ::1] data,
                         np.float64_t [::1] paths):

    cdef Py_ssize_t trees
    cdef Py_ssize_t tree_index
    cdef Py_ssize_t x_index
    cdef selector_t selector
    cdef Py_ssize_t tree_offset
    cdef np.int32_t feature, i

    with nogil, parallel():
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

                paths[x_index] += selector.value

            paths[x_index] /= trees
