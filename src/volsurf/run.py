"""Run one experiment: ``python -m volsurf.run CONFIG [--runs-dir runs] [--holdout]``."""

import argparse
import json
import re
import shutil
import subprocess
import time
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from volsurf.data import load_surface
from volsurf.io import save_run
from volsurf.models import build_model
from volsurf.protocol import walk_forward

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_run_name(name: object) -> str:
    """Return ``name`` if it is a safe single directory name, else raise ``ValueError``.

    Allowed: letters, digits, ``.``, ``_`` and ``-``, not starting with ``.``. This rules out
    empty names, path separators, ``.``/``..`` and absolute paths, so a run always lands
    directly under the runs directory.
    """
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"run name {name!r} must be a single directory name made of letters, digits, "
            "'.', '_' or '-', not starting with '.'"
        )
    return name


def publish_run(runs_dir: Path, name: str, preds: pd.DataFrame, config: dict, meta: dict) -> Path:
    """Write a run to ``runs_dir/name`` so that a failure never leaves a mixed or partial run.

    The files are written to a hidden staging directory next to the target, which then
    replaces any previous run by renaming (the previous run is moved aside first and deleted
    only once the new one is in place). If writing fails, the staging directory is removed
    and the previous run is left untouched.
    """
    run_dir = runs_dir / validate_run_name(name)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    staging = runs_dir / f".{name}.staging-{tag}"
    try:
        save_run(staging, preds, config, meta)
        if run_dir.exists():
            backup = runs_dir / f".{name}.previous-{tag}"
            run_dir.rename(backup)
            try:
                staging.rename(run_dir)
            except BaseException:
                backup.rename(run_dir)
                raise
            shutil.rmtree(backup)
        else:
            staging.rename(run_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # only still there if something failed
    return run_dir


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
    try:
        name = validate_run_name(config.get("name"))
    except ValueError as err:
        parser.error(str(err))
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
    if (args.runs_dir / name).exists():
        print(f"Overwriting existing run {args.runs_dir / name}")
    run_dir = publish_run(args.runs_dir, name, preds, {"source": source}, meta)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
