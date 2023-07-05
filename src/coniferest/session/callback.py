import click
import webbrowser

from typing import List, Optional

import numpy as np

from coniferest.datasets import Label


def prompt_decision_callback(metadata, data, session) -> Label:
    """
    Prompt user to label the object as anomaly or regular.

    If user sends keyboard interrupt, terminate the session.
    """
    try:
        result = click.confirm(f"Is {metadata} anomaly?")
        return Label.ANOMALY if result else Label.REGULAR

    except click.Abort:
        session.terminate()
        return Label.UNKNOWN


def viewer_decision_callback(metadata, data, session) -> Label:
    """
    Open SNAD Viewer for ZTF DR object. Metadata must be ZTF DR object ID.
    """
    url = "https://ztf.snad.space/view/{}".format(metadata)

    try:
        webbrowser.get().open_new_tab(url)
    except webbrowser.Error:
        click.echo("Check {} for details".format(url))

    return prompt_decision_callback(metadata, data, session)


class TerminateAfter:
    """
    Terminate session after given number of iterations.

    Parameters
    ----------
    budget : int
        Number of iterations after which session will be terminated.
    """
    def __init__(self, budget: int):
        self.budget = budget
        self.iteration = 0

    def __call__(self, metadata, data, session) -> None:
        self.iteration += 1
        if self.iteration >= self.budget:
            session.terminate()


def autonomous_decider_callback(metadata: Label, data: np.ndarray, session: 'Session') -> Label:
    "Use metadata as already defined labels for making decision."
    return metadata


class AutonomousExperiment:
    """
    A callback for tracing the experiment and terminating it on the fetched
    number of anomalies or experimental points.

    Parameters
    ----------
    budget: int
        Maximum number of iterations to spend in search of anomalies.

    anomalies: int
        Maximum number of anomalies to search for.

    save_trace: bool
        Whether to save the full trace. Default value is True.
    """
    def __init__(self, budget: int, anomalies: int, save_trace: bool = True):
        self._anomalies: int = anomalies
        self._budget: int = budget
        self._trajectory: List[int] = []
        self._labels: List[Label] = []
        self._trace: List[np.ndarray] = []
        self._save_trace: bool = save_trace

    @property
    def trajectory(self) -> 'np.ndarray[np.int_]':
        return np.array(self._trajectory, dtype=np.int_)

    @property
    def labels(self) -> 'np.ndarray[np.int8]':
        return np.array(self._labels, dtype=np.int8)

    @property
    def trace(self) -> List[np.ndarray]:
        return self._trace

    def __call__(self, metadata: Label, _, session: 'Session') -> None:
        self._budget -= 1
        if metadata == Label.A:
            self._anomalies -= 1

        if self._anomalies == 0 or self._budget == 0:
            session.terminate()

        self._trajectory.append(session.current)
        self._labels.append(session.last_decision)

        if not self._save_trace:
            return

        ordering = np.argsort(session.scores)
        index, = np.where(ordering == session.current)
        self._trace.append(ordering[:index[0] + self._anomalies + 1])


def make_autonomous_experiment(
        labels: 'np.ndarray[Label]',
        budget: Optional[int] = None) -> AutonomousExperiment:
    """
    Create an instance of AutonomousExperiment with parameter values based on
    data labels. Metadata array is assumed to be just data labels. Terminates
    after either all anomalies are detected or whole experiment budget is
    exausted.

    Parameters
    ----------
    labels: np.ndarray[Label]
        Array of labels to derive number of anomalies from.

    budget: int | None
        Optional. Experiment budget for anomaly detection. Defaults to what's
        less either number of data points or 5 times the number of anomalies.
    """

    anomalies = sum(labels == Label.A)
    budget = budget or min(anomalies * 5.0, len(labels))

    return AutonomousExperiment(budget=budget, anomalies=anomalies)
