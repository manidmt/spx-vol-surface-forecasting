import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsurf import evaluate, run
from volsurf.io import save_run
from volsurf.metrics import (
    dm_table,
    errors,
    mae_by_point,
    mae_by_tenor,
    rmse_by_point,
    rmse_by_tenor,
    summary,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.slow
def test_run_writes_files_and_overwrites(tmp_path, capsys, real_surface):
    config = tmp_path / "toy.toml"
    config.write_text('name = "toy"\nkind = "persistence"\ntest_start = "2021-01-01"\n')
    run_dir = run.main([str(config), "--runs-dir", str(tmp_path / "runs")])
    assert run_dir == tmp_path / "runs" / "toy"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert {"commit", "timestamp", "runtime_seconds", "include_holdout", "fit_log"} <= meta.keys()
    assert (run_dir / "config.toml").read_text() == config.read_text()
    run.main([str(config), "--runs-dir", str(tmp_path / "runs")])
    assert "overwrit" in capsys.readouterr().out.lower()


@pytest.mark.slow
def test_run_and_evaluate_persistence(tmp_path, real_surface):
    run_dir = run.main([str(CONFIGS / "persistence.toml"), "--runs-dir", str(tmp_path)])
    for f in ("predictions.parquet", "config.toml", "meta.json"):
        assert (run_dir / f).exists()
    out = evaluate.main([str(run_dir)])
    assert round(out["summary"].loc["persistence", "mae"], 4) == 0.4597


@pytest.mark.slow
def test_evaluate_missing_reference(tmp_path, real_surface):
    run_dir = run.main([str(CONFIGS / "persistence.toml"), "--runs-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        evaluate.main([str(run_dir), "--reference", "nope"])


def _save(run_dir: Path, name: str, preds: pd.DataFrame) -> Path:
    save_run(run_dir, preds, {"source": f'name = "{name}"\n'}, {})
    return run_dir


@pytest.fixture
def synthetic_runs(tmp_path, monkeypatch, synthetic_surface):
    """Three runs on the synthetic surface, each with a NaN day of its own, and the actual surface."""
    monkeypatch.setattr(evaluate, "load_surface", lambda include_holdout=False: synthetic_surface)
    actual = synthetic_surface.loc["2020-01-01":]
    persistence = synthetic_surface.shift(1).loc[actual.index]
    better = 0.8 * persistence + 0.2 * actual
    noisy = persistence + np.random.default_rng(5).normal(scale=0.3, size=persistence.shape)
    better.iloc[10] = np.nan
    noisy.iloc[20, 3] = np.nan
    preds = {"persistence": persistence, "better": better, "noisy": noisy}
    dirs = [_save(tmp_path / name, name, frame) for name, frame in preds.items()]
    return dirs, preds, synthetic_surface


def test_evaluate_custom_reference_reports_both(synthetic_runs):
    dirs, preds, actual = synthetic_runs
    out = evaluate.main([*map(str, dirs), "--reference", "better"])
    errs = errors(actual, preds)
    # one common mask: the NaN actual row, the day after it (persistence) and each run's NaN day
    dropped = actual.index[[400, 401]].append(preds["better"].index[[10]]).append(preds["noisy"].index[[20]])
    assert errs["better"].index.equals(actual.loc["2020-01-01":].index.drop(dropped))

    s = out["summary"]
    assert list(s.columns) == [
        "mae", "rmse", "mae_vs_persistence", "r2_oos_vs_persistence", "mae_vs_better", "r2_oos_vs_better",
    ]  # fmt: skip
    for ref in ("persistence", "better"):
        expected = summary(errs, ref)
        np.testing.assert_allclose(s[f"mae_vs_{ref}"], expected["mae_vs_ref"])
        np.testing.assert_allclose(s[f"r2_oos_vs_{ref}"], expected["r2_oos_vs_ref"])
    np.testing.assert_allclose(s[["mae", "rmse"]], summary(errs)[["mae", "rmse"]])

    pd.testing.assert_frame_equal(out["by_tenor"], mae_by_tenor(errs))
    pd.testing.assert_frame_equal(out["by_tenor_rmse"], rmse_by_tenor(errs))
    pd.testing.assert_frame_equal(out["by_point"], mae_by_point(errs))
    pd.testing.assert_frame_equal(out["by_point_rmse"], rmse_by_point(errs))
    assert out["by_point"].index.equals(actual.columns)
    for ref in ("persistence", "better"):
        np.testing.assert_allclose(
            out["by_tenor_rel"].loc[ref], mae_by_tenor(errs) / mae_by_tenor(errs).loc[ref] - 1
        )
        rel = out["by_point_rel"][ref]
        np.testing.assert_allclose(rel, mae_by_point(errs).div(mae_by_point(errs)[ref], axis=0) - 1)
        assert list(rel.columns) == ["persistence", "better", "noisy"]

    dm = out["dm"]
    assert list(dm.index.get_level_values("reference").unique()) == ["persistence", "better"]
    pd.testing.assert_frame_equal(
        dm.loc[["persistence"]], dm_table(errs, ["persistence"], ["better", "noisy"])
    )
    pd.testing.assert_frame_equal(dm.loc[["better"]], dm_table(errs, ["better"], ["persistence", "noisy"]))


def test_evaluate_default_reference_is_persistence_only(synthetic_runs):
    dirs, _, _ = synthetic_runs
    out = evaluate.main(list(map(str, dirs)))
    assert list(out["summary"].columns) == ["mae", "rmse", "mae_vs_persistence", "r2_oos_vs_persistence"]
    assert set(out["dm"].index.get_level_values("reference")) == {"persistence"}


def test_evaluate_custom_reference_without_persistence(synthetic_runs):
    dirs, _, _ = synthetic_runs
    out = evaluate.main([str(dirs[1]), str(dirs[2]), "--reference", "better"])
    assert list(out["summary"].columns) == ["mae", "rmse", "mae_vs_better", "r2_oos_vs_better"]
    assert set(out["dm"].index.get_level_values("reference")) == {"better"}


def test_evaluate_rejects_duplicate_run_names(tmp_path, synthetic_runs, capsys):
    dirs, preds, _ = synthetic_runs
    twin = _save(tmp_path / "other_root" / "persistence", "persistence", preds["persistence"])
    with pytest.raises(SystemExit):
        evaluate.main([str(dirs[0]), str(dirs[1]), str(twin)])
    err = capsys.readouterr().err
    assert "persistence" in err and str(dirs[0]) in err and str(twin) in err
