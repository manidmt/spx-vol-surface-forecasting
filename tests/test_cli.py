import json
from pathlib import Path

import pytest

from volsurf import evaluate, run

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
