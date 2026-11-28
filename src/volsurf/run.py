"""Run one experiment: ``python -m volsurf.run CONFIG [--runs-dir runs] [--holdout]``."""

import argparse
import json
import shutil
import subprocess
import time
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from volsurf.data import load_surface
from volsurf.ensemble import average_runs
from volsurf.io import save_run, validate_run_name
from volsurf.models import EXOG_KINDS, build_model
from volsurf.protocol import walk_forward


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


def _split_exog(kind: str, params: dict) -> tuple[list[str] | None, dict]:
    """Split the optional ``exog`` list of input blocks off a config's model ``params``.

    Returns ``(blocks, remaining params)``, with ``blocks`` None when the config has no
    ``exog``. Raises ``ValueError`` if ``exog`` is given for a kind that does not accept it, is not a
    non-empty list of block names, or comes with an explicit ``n_exog`` (set from the panel).
    """
    if "exog" not in params:
        return None, params
    params = dict(params)
    blocks = params.pop("exog")
    if kind not in EXOG_KINDS:
        raise ValueError(f"exog is only supported by the kinds {', '.join(EXOG_KINDS)}, not {kind!r}")
    if not (isinstance(blocks, list) and blocks and all(isinstance(b, str) for b in blocks)):
        raise ValueError(f"exog must be a non-empty list of block names, got {blocks!r}")
    if "n_exog" in params:
        raise ValueError("n_exog is set from the exog panel; remove it from params")
    return blocks, params


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
    params = config.get("params", {})
    start = time.perf_counter()
    if config["kind"] == "ensemble":
        # Averages saved forecasts from the same runs directory; no surface or fitting needed.
        preds, members = average_runs(args.runs_dir, params["members"], params.get("weights"))
        fit_log: object = []
        holdout = members[0]["include_holdout"]
        if args.holdout and not holdout:
            raise ValueError("--holdout contradicts the ensemble members, which exclude the hold-out")
        extra = {"members": members}
    else:
        blocks, params = _split_exog(config["kind"], params)
        holdout = args.holdout
        surface = load_surface(include_holdout=holdout)
        exog, extra = None, {}
        if blocks is not None:
            # imported here so that runs without exogenous inputs do not need the features module
            from volsurf.features import build_exog

            exog = build_exog(blocks, surface.index)
            params = {**params, "n_exog": exog.shape[1]}
            extra = {"exog": {"blocks": blocks, "features": [str(c) for c in exog.columns]}}
        model = build_model(config["kind"], name, params)
        preds = walk_forward(model, surface, pd.Timestamp(config["test_start"]), exog=exog)
        fit_log = _jsonable(model.fit_log)
    runtime = time.perf_counter() - start

    meta = {
        "commit": _git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime,
        "include_holdout": holdout,
        "fit_log": fit_log,
        **extra,
    }
    json.dumps(meta)  # fail before touching the run directory if not serialisable
    if (args.runs_dir / name).exists():
        print(f"Overwriting existing run {args.runs_dir / name}")
    run_dir = publish_run(args.runs_dir, name, preds, {"source": source}, meta)
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
