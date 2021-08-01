cimport numpy as np

cdef packed struct selector_t:
    np.int32_t feature
    np.int32_t left
    np.float64_t value
    np.int32_t right


cdef void _mean_paths(selector_t [::1] selectors,
                         np.int64_t [::1] indices,
                         np.double_t [:, ::1] data,
                         np.double_t [::1] paths)