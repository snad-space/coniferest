from typing import Callable, Optional

import numpy as np

from coniferest.coniferest import Coniferest
from coniferest.label import Label
from coniferest.session import Session
from coniferest.session.callback import TerminateAfter, TerminateAfterNAnomalies


class OracleSession(Session):
    """Automated session to run experiments with labeled data.

    Parameters
    ----------
    data : np.ndarray, 2-D
        Array with feature values of objects.
    labels : np.ndarray, 1-D
        Array with true labels, of Label or int type
    model : Coniferest
        Anomaly detection model to use
    max_iterations : int
        Maximum number of asked decisions
    max_anomalies : int
        Maximum number of anomalies to search for
    on_refit_callbacks : list of callable, or callable, or None, optional
        Functions to be called when model is refitted, default is empty list.
    on_decision_callbacks : list of callable, or callable, or None, optional
        Functions to be called when expert decision is made, default is empty list.

    Also see methods and attributes from the base `Session` class
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        model: Coniferest,
        max_iterations: int,
        max_anomalies: int,
        on_refit_callbacks=None,
        on_decision_callbacks=None,
    ):
        if on_decision_callbacks is None:
            on_decision_callbacks = []
        elif isinstance(on_decision_callbacks, Callable):
            on_decision_callbacks = [
                on_decision_callbacks,
            ]

        super().__init__(
            data=data,
            metadata=labels,
            model=model,
            # Session.metadata is labels, so we just use this candidate metadata is the true label
            decision_callback=lambda label, _features, _self: label,
            on_refit_callbacks=on_refit_callbacks,
            on_decision_callbacks=[
                TerminateAfter(max_iterations),
                TerminateAfterNAnomalies(max_anomalies),
            ]
            + on_decision_callbacks,
        )


def create_oracle_session(
    data: np.ndarray,
    labels: np.ndarray[int],
    *,
    model: Coniferest,
    max_iterations: Optional[int] = None,
    on_refit_callbacks=None,
    on_decision_callbacks=None,
) -> OracleSession:
    """Create an automated session to run experiments with labeled data.

    Parameters
    ----------
    data : np.ndarray, 2-D
        Array with feature values of objects.
    labels : np.ndarray, 1-D
        Array with true labels, of Label or int type
    model : Coniferest
        Anomaly detection model to use
    max_iterations : int or None, optional
        Maximum number of asked decisions. Default is 5 times the number of anomalies.
    on_refit_callbacks : list of callable, or callable, or None, optional
        Functions to be called when model is refitted, default is empty list.
    on_decision_callbacks : list of callable, or callable, or None, optional
        Functions to be called when expert decision is made, default is empty list.

    Returns
    -------
    OracleSession
    """
    n_anomalies = np.sum(labels == Label.ANOMALY)
    max_iterations = min(n_anomalies * 5.0, len(labels)) if max_iterations is None else max_iterations

    return OracleSession(
        data,
        labels,
        model=model,
        max_iterations=max_iterations,
        max_anomalies=n_anomalies,
        on_refit_callbacks=on_refit_callbacks,
        on_decision_callbacks=on_decision_callbacks,
    )
