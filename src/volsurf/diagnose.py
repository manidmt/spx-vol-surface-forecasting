"""Diagnose runs against a reference: ``python -m volsurf.diagnose RUN_DIR ... --reference NAME``.

For every run other than the reference, on the days common to all the runs given: relative MAE,
Diebold-Mariano (absolute loss), Clark-West (the reference is taken as the nested model),
Giacomini-White with the default instruments (constant, previous day's loss differential, ex-ante
vol state), the days needed to detect gains of 0.5, 1 and 2 % with 80 % power, and the tables by
year and by tercile of the ex-ante vol state. ``--out DIR`` also writes them as CSV files.
"""

import argparse
import tomllib
from pathlib import Path

import pandas as pd

from volsurf.data import load_surface
from volsurf.diagnostics import (
    by_state,
    by_year,
    clark_west,
    days_for_power,
    default_instruments,
    dm_absolute,
    giacomini_white,
    relative_mae,
    state_instrument,
)
from volsurf.io import load_run
from volsurf.metrics import errors

EFFECTS = (0.005, 0.01, 0.02)


def diagnose(errs: dict[str, pd.DataFrame], reference: str, actual: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables for every model in ``errs`` against ``reference`` (errors on a common mask)."""
    e_ref = errs[reference]
    state = state_instrument(actual)
    overall, years, terciles = {}, {}, {}
    for name, e in errs.items():
        if name == reference:
            continue
        dm_t, dm_p = dm_absolute(e_ref, e, 5)
        cw_t, cw_p = clark_west(e_ref, e)
        gw, gw_p, gw_n = giacomini_white(e_ref, e, default_instruments(e_ref, e, actual, "absolute"))
        row = {
            "n_days": len(e),
            "mae_vs_ref": relative_mae(e_ref, e),
            "dm_t_abs": dm_t,
            "dm_p_abs": dm_p,
            "cw_t": cw_t,
            "cw_p": cw_p,
            "gw_wald_abs": gw,
            "gw_p_abs": gw_p,
            "gw_days": gw_n,
        }
        for effect in EFFECTS:
            row[f"days_for_{effect:.1%}"] = days_for_power(e_ref, e, effect)
        overall[name] = row
        years[name] = by_year(e_ref, e)
        terciles[name] = by_state(e_ref, e, state)
    return {
        "overall": pd.DataFrame.from_dict(overall, orient="index"),
        "by_year": pd.concat(years, names=["model", "year"]),
        "by_state": pd.concat(terciles, names=["model", "state"]),
    }


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Print (and optionally save) the diagnostic tables; returns them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--holdout", action="store_true", help="include the hold-out period")
    args = parser.parse_args(argv)

    preds: dict[str, pd.DataFrame] = {}
    for run_dir in args.run_dirs:
        name = tomllib.loads((run_dir / "config.toml").read_text())["name"]
        if name in preds:
            parser.error(f"run name {name!r} appears twice")
        preds[name], _ = load_run(run_dir)
    if args.reference not in preds:
        parser.error(f"reference {args.reference!r} is not among the runs: {sorted(preds)}")
    actual = load_surface(include_holdout=args.holdout)
    errs = errors(actual, preds)
    tables = diagnose(errs, args.reference, actual)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(f"Days evaluated: {len(errs[args.reference])}; reference {args.reference}\n")
        for key, table in tables.items():
            print(key, "\n", table.round(4), "\n")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for key, table in tables.items():
            table.to_csv(args.out / f"{key}.csv")
    return tables


if __name__ == "__main__":
    main()
