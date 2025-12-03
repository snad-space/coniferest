#!/usr/bin/env python3
"""
Ensure every notebook under docs/pre_executed has metadata:
"nbsphinx": {"execute": "never"}

This script is safe to run as a pre-commit hook. It will modify notebooks in-place
and print modified file paths. It raises exceptions on hard errors (instead of
return codes). If any notebook has an explicit `nbsphinx.execute` value other than
`"never"`, the script fails without making changes to that notebook.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "docs" / "pre_executed"


def find_notebooks(dirpath: Path) -> list[Path]:
    if not dirpath.exists():
        return []
    return sorted(dirpath.glob("*.ipynb"))


def ensure_nbsphinx_execute_never(notebook_path: Path) -> bool:
    """Return True if the file was modified.

    Raises RuntimeError when the notebook explicitly sets `nbsphinx.execute`
    to a value other than "never" (the script should fail with no fix).
    """
    with notebook_path.open("r") as fh:
        nb = json.load(fh)

    if not isinstance(nb, dict):
        raise RuntimeError(f"Unexpected notebook format for {notebook_path}")

    metadata = nb.setdefault("metadata", {})
    nbsphinx = metadata.setdefault("nbsphinx", {})

    if not isinstance(nbsphinx, dict):
        raise RuntimeError(
            f"Notebook {notebook_path} has metadata.nbsphinx of type {type(nbsphinx).__name__}; expected a mapping"
        )

    if "execute" in nbsphinx:
        if nbsphinx["execute"] == "never":
            return False
        raise RuntimeError(
            f"Notebook {notebook_path} has metadata.nbsphinx.execute={nbsphinx.get('execute')!r}; expected 'never'"
        )

    nbsphinx["execute"] = "never"

    with notebook_path.open("w") as fh:
        json.dump(nb, fh, indent=1, sort_keys=True, ensure_ascii=False)

    logging.warning(f"Modified notebook to set nbsphinx.execute='never': {notebook_path}")

    return True


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if argv is None:
        argv = sys.argv[1:]

    if argv:
        notebooks = list(map(Path, argv))
    else:
        notebooks = find_notebooks(TARGET_DIR)

    if len(notebooks) == 0:
        print(f"No notebooks found{' for provided paths' if argv else ''}.")
        return 0

    # If pre-scan passed, perform modifications safely.
    return_code = 0
    for nb in notebooks:
        try:
            if ensure_nbsphinx_execute_never(nb):
                return_code = 1
        except Exception as e:
            logging.error(str(e))
            return_code = 1

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
