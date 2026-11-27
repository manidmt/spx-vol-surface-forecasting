"""Saving and loading of run directories (predictions, config copy, metadata)."""

import json
from pathlib import Path

import pandas as pd


def save_run(run_dir: Path, preds: pd.DataFrame, config: dict, meta: dict) -> None:
    """Write ``predictions.parquet``, ``config.toml`` and ``meta.json`` into ``run_dir``.

    ``config["source"]`` is the text of the config file, copied verbatim to ``config.toml``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    flat = preds.copy()
    flat.columns = [f"{t}|{m:g}" for t, m in preds.columns]
    flat.to_parquet(run_dir / "predictions.parquet")
    (run_dir / "config.toml").write_text(config["source"])
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load predictions (original column order and MultiIndex) and metadata of a run."""
    flat = pd.read_parquet(run_dir / "predictions.parquet")
    pairs = [c.rsplit("|", 1) for c in flat.columns]
    flat.columns = pd.MultiIndex.from_tuples([(t, float(m)) for t, m in pairs], names=["tenor", "moneyness"])
    meta = json.loads((run_dir / "meta.json").read_text())
    return flat, meta
