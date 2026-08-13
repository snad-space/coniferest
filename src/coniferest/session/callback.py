from pathlib import Path
import webbrowser

import click

from coniferest.datasets import Label

from coniferest.onnx.convert import save_onnx_model, to_onnx


class _LabelChoice(click.Choice):
    """Label choice class for click

    Accepts (case-insensitive):
        * -1 / a / anomaly / yes
        * 1 / r / regular / no
        * 0 / u / unknown
    """

    def __init__(self):
        super().__init__(Label)

    def normalize_choice(self, choice, ctx):
        del ctx
        if isinstance(choice, Label):
            return choice
        if choice.lower() == "y" or choice.lower() == "yes":
            return Label.ANOMALY
        if choice.lower() == "n" or choice.lower() == "no":
            return Label.REGULAR
        try:
            return int(choice)
        except ValueError:
            pass
        try:
            return getattr(Label, choice.upper())
        except AttributeError:
            pass


def prompt_decision_callback(metadata, data, session) -> Label:
    """
    Prompt user to label the object as anomaly or regular.

    If user sends keyboard interrupt, terminate the session.
    """
    try:
        return click.prompt(
            text=f"Is {metadata} an anomaly? ([A]nomaly / yes, [R]egular / no, [U]nknown)",
            type=_LabelChoice(),
            show_choices=False,
        )
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

    This callback to be used as "on decision callback":
    Session(..., on_decision_callbacks=[TerminateAfter(budget)])

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


class TerminateAfterNAnomalies:
    """
    Terminate session after given number of newly labeled anomalies.

    This callback to be used as "on decision callback":
    Session(..., on_decision_callbacks=[TerminateAfter(budget)])

    Parameters
    ----------
    budget : int
        Number of anomalies to stop after.
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.anomalies_count = 0

    def __call__(self, label, _data, session) -> None:
        self.anomalies_count += label == Label.ANOMALY
        if self.anomalies_count >= self.budget:
            session.terminate()


class SaveToOnnx:
    """
    Callback that periodically saves the current session model to ONNX.

    Use it as an "on decision callback":
    Session(..., on_decision_callbacks=[SaveToOnnx(directory="models")])

    Parameters
    ----------
    directory : str
        Directory where the ONNX file(s) will be saved. Created if missing.

    filename : str, optional
        Base name of the ONNX file. Default is "model.onnx".

    every_n_decisions : int, optional
        Save every N decisions. Default is 1 (save after every decision).

    overwrite : bool, optional
    If True (default), always overwrite the same file.
    If False, keep a new numbered file for each save
    (e.g. model_1.onnx, model_2.onnx, ...). The number in the
    filename corresponds to the decision counter at the time of
    saving, so if ``every_n_decisions`` is greater than 1, the
    numbers will not be consecutive (e.g. with
    ``every_n_decisions=5``, files will be named
    ``model_5.onnx``, ``model_10.onnx``, ``model_15.onnx``, ...).
    """

    def __init__(
        self,
        directory,
        filename="model.onnx",
        every_n_decisions=1,
        overwrite=True,
    ):
        self.directory = Path(directory)
        self.filename = filename
        self.every_n_decisions = every_n_decisions
        self.overwrite = overwrite
        self._counter = 0

        self.directory.mkdir(parents=True, exist_ok=True)

    def _build_path(self):
        if self.overwrite:
            return self.directory / self.filename

        stem = Path(self.filename).stem
        suffix = Path(self.filename).suffix
        numbered_filename = f"{stem}_{self._counter}{suffix}"
        return self.directory / numbered_filename

    def __call__(self, metadata, data, session) -> None:
        self._counter += 1

        if self._counter % self.every_n_decisions != 0:
            return

        model = session.model

        if not hasattr(model, "n_features_in_"):
            raise RuntimeError(
                "SaveToOnnx callback requires a fitted model. "
                "The session model has not been fitted yet, "
                "which can happen if this callback is triggered "
                "before any decision has caused a model refit. "
                "Make sure the model is fitted (e.g. via session.model.fit(...)) "
                "before this callback is called."
            )

        onnx_model = to_onnx(model)
        path = self._build_path()
        save_onnx_model(onnx_model, path)