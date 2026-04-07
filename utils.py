"""Shared utilities for AUTODS agents."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def load_project_env(anchor_file: str | Path) -> None:
    """Load the .env file located in the same directory as anchor_file (or
    any parent directory up to the filesystem root).  Falls back silently
    when python-dotenv is not installed or no .env file is found."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    search = Path(anchor_file).resolve().parent
    for directory in [search, *search.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return
    # No .env found — still call load_dotenv so any OS-level env vars are used
    load_dotenv(override=False)


def reexec_with_project_venv(script_file: str | Path) -> None:
    """Re-execute the current script inside the project's virtual environment.

    This is a CLI helper used when a script is run directly (``__name__ ==
    '__main__'``).  It is a no-op when the script is imported as a module or
    when the project does not have a venv directory alongside it.
    """
    script_path = Path(script_file).resolve()
    for venv_name in ("venv", ".venv", "env"):
        venv_python = script_path.parent / venv_name / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
            return  # unreachable if execv succeeds


def json_default(obj):
    """JSON serialiser fallback that handles numpy scalar types.

    Pass this as the ``default`` argument to ``json.dump`` / ``json.dumps``
    so that numpy integers, floats, and booleans produced by pandas/sklearn
    are serialised without raising ``TypeError``.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)
