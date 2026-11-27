"""Equal- or weighted-average ensembles of saved runs (no refitting)."""

from pathlib import Path

import numpy as np
import pandas as pd

from volsurf.io import load_run


def average_runs(
    runs_dir: Path, members: list[str], weights: list[float] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Weighted average of the saved forecasts of ``members`` found in ``runs_dir``.

    Weights default to equal; they must be non-negative, one per member, and sum to 1. All
    members must share the index and columns (same order) of the first one, else
    ``ValueError`` naming the offending member. Returns the averaged forecast and
    ``{"name", "commit", "timestamp"}`` for each member, in order.
    """
    from volsurf.run import validate_run_name  # deferred: run imports this module

    if not members:
        raise ValueError("an ensemble needs at least one member (members is empty)")
    if weights is None:
        weights = [1 / len(members)] * len(members)
    if len(weights) != len(members):
        raise ValueError(f"weights has {len(weights)} entries for {len(members)} members")
    if any(w < 0 for w in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError(f"weights must be non-negative and sum to 1, got {weights}")

    total: pd.DataFrame | None = None
    info: list[dict] = []
    for name, weight in zip(members, weights, strict=True):
        preds, meta = load_run(runs_dir / validate_run_name(name))
        if total is None:
            total, first = preds * weight, name
        else:
            if not preds.index.equals(total.index):
                raise ValueError(f"member {name!r} has a different index from {first!r}")
            if not preds.columns.equals(total.columns):
                raise ValueError(f"member {name!r} has different columns from {first!r}")
            total = total + preds * weight
        info.append({"name": name, "commit": meta.get("commit"), "timestamp": meta.get("timestamp")})
    assert total is not None
    return total, info
