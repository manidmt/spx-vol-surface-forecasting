"""Equal- or weighted-average ensembles of saved runs (no refitting)."""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from volsurf.io import load_run, validate_run_name


def average_runs(
    runs_dir: Path, members: list[str], weights: list[float] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Weighted average of the saved forecasts of ``members`` found in ``runs_dir``.

    Weights default to equal; they must be non-negative, one per member, and sum to 1. All
    members must share the index and columns (same order) of the first one, else
    ``ValueError`` naming the offending member. Members must also agree on ``include_holdout``
    (from ``meta.json``) and ``test_start`` (from ``config.toml``), else ``ValueError``.
    Returns the averaged forecast and ``{"name", "commit", "timestamp", "include_holdout",
    "test_start"}`` for each member, in order.
    """
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
        test_start = tomllib.loads((runs_dir / name / "config.toml").read_text()).get("test_start")
        info.append(
            {
                "name": name,
                "commit": meta.get("commit"),
                "timestamp": meta.get("timestamp"),
                "include_holdout": bool(meta.get("include_holdout", False)),
                "test_start": None if test_start is None else str(test_start),
            }
        )
        for key in ("include_holdout", "test_start"):
            if info[-1][key] != info[0][key]:
                raise ValueError(
                    f"member {name!r} has {key}={info[-1][key]!r} but {info[0]['name']!r} "
                    f"has {info[0][key]!r}; ensemble members must agree"
                )
    assert total is not None
    return total, info
