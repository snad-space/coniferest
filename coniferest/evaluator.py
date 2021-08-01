import numpy as np
from .utils import average_path_length
from .calc_mean_paths import calc_mean_paths  # noqa


class ForestEvaluator:
    selector_dtype = np.dtype([('feature', np.int32), ('left', np.int32), ('value', np.double), ('right', np.int32)])

    def __init__(self, samples, selectors, indices):
        self.samples = samples

        self.selectors = selectors
        self.indices = indices

    @classmethod
    def combine_selectors(cls, selectors_list):
        lens = [len(sels) for sels in selectors_list]
        full_len = sum(lens)

        selectors = np.empty((full_len,), dtype=cls.selector_dtype)

        indices = np.empty((len(selectors_list) + 1,), dtype=np.int64)
        indices[0] = 0
        indices[1:] = np.add.accumulate(lens)

        for i in range(len(selectors_list)):
            selectors[indices[i]:indices[i + 1]] = selectors_list[i]

        return selectors, indices

    def score_samples(self, x):
        return -2 ** (- calc_mean_paths(self.selectors, self.indices, x) / average_path_length(self.samples))