import click
import webbrowser

from coniferest.datasets import Label


def prompt_decision_callback(metadata, data, session):
    try:
        result = click.confirm(f"Is {metadata} anomaly?")
        return Label.ANOMALY if result else Label.REGULAR

    except click.Abort:
        session.terminate()
        return Label.UNKNOWN

def viewer_decision_callback(metadata, data, session):
    url = "https://ztf.snad.space/view/{}".format(metadata)

    try:
        webbrowser.get().open_new_tab(url)
    except webbrowser.Error:
        click.echo("Check {} for details".format(url))

    return prompt_decision_callback(metadata, data, session)
