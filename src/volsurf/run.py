"""Run one experiment: ``python -m volsurf.run CONFIG [--runs-dir runs] [--holdout]``."""

import argparse
import json
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from volsurf.data import load_surface
from volsurf.io import save_run
from volsurf.models import build_model
from volsurf.protocol import walk_forward


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    return value


def main(argv: list[str] | None = None) -> Path:
    """Walk-forward a configured model over the surface and save the run; returns its directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--holdout", action="store_true", help="include the hold-out period")
    args = parser.parse_args(argv)

    source = args.config.read_text()
    config = tomllib.loads(source)
    name = config["name"]
    surface = load_surface(include_holdout=args.holdout)
    model = build_model(config["kind"], name, config.get("params", {}))

    start = time.perf_counter()
    preds = walk_forward(model, surface, pd.Timestamp(config["test_start"]))
    runtime = time.perf_counter() - start

    meta = {
        "commit": _git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime,
        "include_holdout": args.holdout,
        "fit_log": _jsonable(model.fit_log),
    }
    json.dumps(meta)  # fail before touching the run directory if not serialisable
    run_dir = args.runs_dir / name
    if run_dir.exists():
        print(f"Overwriting existing run {run_dir}")
    save_run(run_dir, preds, {"source": source}, meta)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
