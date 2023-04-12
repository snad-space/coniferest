import click
import webbrowser

from coniferest.datasets import Label


def click_decision_callback(metadata, data, session):
    try:
        result = click.confirm("Is {} anomaly?".format(str(metadata)))
        return Label.ANOMALY if result else Label.REGULAR

    except:
        session.terminate()
        return Label.UNKNOWN

def viewer_decision_callback(metadata, data, session):
    url = "https://ztf.snad.space/dr4/view/{}".format(metadata)

    try:
        webbrowser.get().open_new_tab(url)
    except webbrowser.Error:
        click.echo("Check {} for details".format(url))

    return click_decision_callback(metadata, data, session)
