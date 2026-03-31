"""Shared utilities for AUTODS agents."""

from __future__ import annotations

import numpy as np


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
