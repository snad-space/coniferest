from typing import Callable

import numpy as np

from coniferest.coniferest import Coniferest
from coniferest.pineforest import PineForest

from .callback import click_decision_callback


## ds = ztf_dataset()
## s = Session(*ds, lambda : ...., )
## s.run()
class Session:
    @staticmethod
    def _validate_callbacks(callbacks):
        return all([isinstance(cb, Callable) for cb in callbacks])

    @staticmethod
    def _invoke_callbacks(callbacks, *args, **kwargs):
        for cb in callbacks:
            cb(*args, **kwargs)


    def __init__(self, data, metadata, decision_callback = click_decision_callback, on_refit_callbacks = [], on_decision_callbacks = [], known_labels = None, model = PineForest()):

        self._data     = np.atleast_2d(data)
        self._metadata = np.atleast_1d(metadata)

        if not isinstance(decision_callback, Callable):
            raise ValueError("decision_callback is not a callable")

        self._decision_cb = decision_callback

        if not isinstance(on_refit_callbacks, list):
            on_refit_callbacks = [on_refit_callbacks,]

        if not self._validate_callbacks(on_refit_callbacks):
            raise ValueError("on_refit_callbacks contains not callable object")

        self._on_refit_cb  = on_refit_callbacks

        if not isinstance(on_decision_callbacks, list):
            on_decision_callbacks = [on_decision_callbacks,]

        if not self._validate_callbacks(on_decision_callbacks):
            raise ValueError("on_decision_callbacks is not callable")

        self._on_decision_cb = on_decision_callbacks

        if known_labels is None:
            self._known_labels = {}
        else:
            self._known_labels = dict(known_labels)

        if not isinstance(model, Coniferest):
            raise ValueError("model is not a Coniferest object")

        self._model = model

        self._scores = None
        self._current = None
        self._terminated = False

    def run(self):
        if self._terminated:
            return self

        self.model.fit(self._data)

        while not self._terminated:
            known_data = self._data[list(self._known_labels.keys())]
            known_labels = list(self._known_labels.values())
            self.model.fit_known(self._data, known_data, known_labels)

            self._invoke_callbacks(self._on_refit_cb, self)

            self._scores = self.model.score_samples(self._data)

            self._current = None
            for ind in np.argsort(self._scores):
                if ind not in self._known_labels:
                    self._current = ind
                    break

            if self._current is None:
                self.terminate()
                break

            decision = self._decision_cb(self._metadata[self._current], self._data[self._current], self)
            self._known_labels[self._current] = decision

            self._invoke_callbacks(self._on_decision_cb, self)

        return self

    def terminate(self):
        self._terminated = True


    @property
    def current(self):
        return self._current

    @property
    def scores(self):
        return self._scores

    @property
    def known_labels(self):
        return self._known_labels

    @property
    def model(self):
        return self._model

    @property
    def terminated(self):
        return self._terminated
