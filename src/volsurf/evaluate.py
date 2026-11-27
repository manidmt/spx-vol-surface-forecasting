"""Compare runs: ``python -m volsurf.evaluate RUN_DIR [RUN_DIR ...] [--reference persistence]``.

Every table is computed on one common mask: the days on which the actual surface and every run
are finite. Models are compared against persistence (when a persistence run is loaded) and
against ``--reference`` when it is a different run.
"""

import argparse
import tomllib
from pathlib import Path

import pandas as pd

from volsurf.data import load_surface
from volsurf.io import load_run
from volsurf.metrics import (
    dm_table,
    errors,
    mae_by_point,
    mae_by_tenor,
    rmse_by_point,
    rmse_by_tenor,
    summary,
)

PERSISTENCE = "persistence"


def comparison_tables(errs: dict[str, pd.DataFrame], references: list[str]) -> dict[str, pd.DataFrame]:
    """All evaluation tables for error frames already restricted to a common mask.

    Keys: ``summary`` (models x MAE, RMSE and, for each reference, ``mae_vs_<ref>`` and
    ``r2_oos_vs_<ref>``); ``by_tenor`` and ``by_tenor_rmse`` (models x tenors);
    ``by_tenor_rel`` (MAE relative to each reference, rows ``(reference, model)``);
    ``by_point`` and ``by_point_rmse`` (points x models); ``by_point_rel`` (points x
    ``(reference, model)``); ``dm`` (Diebold-Mariano of every other model against each
    reference, rows ``(reference, model, loss)``).
    """
    overall = summary(errs, references[0])[["mae", "rmse"]]
    for ref in references:
        relative = summary(errs, ref)
        overall[f"mae_vs_{ref}"] = relative["mae_vs_ref"]
        overall[f"r2_oos_vs_{ref}"] = relative["r2_oos_vs_ref"]
    by_tenor, by_point = mae_by_tenor(errs), mae_by_point(errs)
    dm_parts = [dm_table(errs, [ref], [m for m in errs if m != ref]) for ref in references if len(errs) > 1]
    return {
        "summary": overall,
        "by_tenor": by_tenor,
        "by_tenor_rmse": rmse_by_tenor(errs),
        "by_tenor_rel": pd.concat(
            {ref: by_tenor / by_tenor.loc[ref] - 1 for ref in references}, names=["reference", "model"]
        ),
        "by_point": by_point,
        "by_point_rmse": rmse_by_point(errs),
        "by_point_rel": pd.concat(
            {ref: by_point.div(by_point[ref], axis=0) - 1 for ref in references},
            axis=1,
            names=["reference", "model"],
        ),
        "dm": pd.concat(dm_parts) if dm_parts else pd.DataFrame(columns=["t_stat", "p_value"]),
    }


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Print and return the evaluation tables for the given runs (see ``comparison_tables``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--reference", default=PERSISTENCE)
    parser.add_argument("--holdout", action="store_true", help="include the hold-out period")
    args = parser.parse_args(argv)

    preds: dict[str, pd.DataFrame] = {}
    sources: dict[str, Path] = {}
    for run_dir in args.run_dirs:
        name = tomllib.loads((run_dir / "config.toml").read_text())["name"]
        if name in sources:
            parser.error(f"run name {name!r} appears twice: {sources[name]} and {run_dir}")
        sources[name] = run_dir
        preds[name], _ = load_run(run_dir)
    if args.reference not in preds:
        parser.error(f"reference {args.reference!r} is not among the runs: {sorted(preds)}")
    references = [PERSISTENCE] if PERSISTENCE in preds else []
    if args.reference not in references:
        references.append(args.reference)

    actual = load_surface(include_holdout=args.holdout)
    errs = errors(actual, preds)
    tables = comparison_tables(errs, references)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(f"Days evaluated: {len(next(iter(errs.values())))}\n")
        print(tables["summary"].round(4), "\n")
        print("MAE by tenor\n", tables["by_tenor"].round(4), "\n")
        print("RMSE by tenor\n", tables["by_tenor_rmse"].round(4), "\n")
        print("MAE by tenor relative to each reference\n", tables["by_tenor_rel"].round(4), "\n")
        print("MAE by point\n", tables["by_point"].round(4), "\n")
        print("RMSE by point\n", tables["by_point_rmse"].round(4), "\n")
        print("MAE by point relative to each reference\n", tables["by_point_rel"].round(4), "\n")
        if len(tables["dm"]):
            print(tables["dm"].round(4))
    return tables


if __name__ == "__main__":
    main()
