# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import inspect
import pathlib
import subprocess

import coniferest

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "coniferest"
copyright = "2024, SNAD team"
author = "SNAD team"

GITHUB_REPO = "snad-space/coniferest"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "../images/CF_logo_green_sign_universal.svg"
html_title = "coniferest"

html_theme_options = {
    "repository_url": GITHUB_URL,
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
    },
    "show_toc_level": 2,
}

nbsphinx_execute = 'never'


# -- Options for linkcode extension -------------------------------------------
# Point API doc "source" links to GitHub instead of embedding a local copy.

try:
    _GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parent,
    ).decode().strip()
except Exception:
    _GIT_SHA = f"v{coniferest.__version__}"


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    module_name = info["module"]
    fullname = info["fullname"]

    try:
        module = importlib.import_module(module_name)
        obj = module
        for part in fullname.split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
    except Exception:
        return None

    try:
        source_file = inspect.getsourcefile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None

    if source_file is None:
        return None

    try:
        rel_path = pathlib.Path(source_file).resolve().relative_to(
            pathlib.Path(coniferest.__file__).resolve().parent.parent
        )
    except ValueError:
        return None

    end_line = start_line + len(source_lines) - 1
    return f"{GITHUB_URL}/blob/{_GIT_SHA}/src/{rel_path.as_posix()}#L{start_line}-L{end_line}"
