"""Compare runs: ``python -m volsurf.evaluate RUN_DIR [RUN_DIR ...] [--reference persistence]``."""

import argparse
import tomllib
from pathlib import Path

import pandas as pd

from volsurf.data import load_surface
from volsurf.io import load_run
from volsurf.metrics import dm_table, errors, mae_by_tenor, summary


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Print and return summary, per-tenor MAE and Diebold-Mariano tables for the given runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--reference", default="persistence")
    parser.add_argument("--holdout", action="store_true", help="include the hold-out period")
    args = parser.parse_args(argv)

    preds = {}
    for run_dir in args.run_dirs:
        name = tomllib.loads((run_dir / "config.toml").read_text())["name"]
        preds[name], _ = load_run(run_dir)
    if args.reference not in preds:
        parser.error(f"reference {args.reference!r} is not among the runs: {sorted(preds)}")

    actual = load_surface(include_holdout=args.holdout)
    errs = errors(actual, preds)
    by_tenor = mae_by_tenor(errs)
    others = [m for m in errs if m != args.reference]
    tables = {
        "summary": summary(errs, args.reference),
        "by_tenor": by_tenor,
        "dm": dm_table(errs, [args.reference], others),
    }
    relative = by_tenor / by_tenor.loc[args.reference] - 1
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(f"Days evaluated: {len(next(iter(errs.values())))}\n")
        print(tables["summary"].round(4), "\n")
        print("MAE by tenor\n", by_tenor.round(4), "\n")
        print(f"MAE by tenor relative to {args.reference}\n", relative.round(4), "\n")
        if others:
            print(tables["dm"].round(4))
    return tables


if __name__ == "__main__":
    main()
