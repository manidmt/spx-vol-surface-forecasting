import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from volsurf import run
from volsurf.ensemble import average_runs
from volsurf.io import load_run, save_run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _save(
    runs_dir: Path,
    name: str,
    preds: pd.DataFrame,
    commit: str = "abc",
    holdout: bool = False,
    test_start: str = "2021-01-04",
) -> None:
    meta = {"commit": commit, "timestamp": f"ts-{name}", "include_holdout": holdout}
    source = f'name = "{name}"\ntest_start = "{test_start}"\n'
    save_run(runs_dir / name, preds, {"source": source}, meta)


@pytest.fixture
def two_members(tmp_path, synthetic_surface):
    a = synthetic_surface.loc["2020-01-01":]
    b = a + 1.0
    _save(tmp_path, "a", a, "c1")
    _save(tmp_path, "b", b, "c2")
    return tmp_path, a, b


def test_average_is_mean_and_reports_members(two_members):
    runs_dir, a, b = two_members
    avg, members = average_runs(runs_dir, ["a", "b"])
    assert_frame_equal(avg, (a + b) / 2, check_freq=False)
    assert members == [
        {
            "name": "a",
            "commit": "c1",
            "timestamp": "ts-a",
            "include_holdout": False,
            "test_start": "2021-01-04",
        },
        {
            "name": "b",
            "commit": "c2",
            "timestamp": "ts-b",
            "include_holdout": False,
            "test_start": "2021-01-04",
        },
    ]


def test_average_weights(two_members):
    runs_dir, a, b = two_members
    avg, _ = average_runs(runs_dir, ["a", "b"], [0.75, 0.25])
    assert_frame_equal(avg, 0.75 * a + 0.25 * b, check_freq=False)


def test_average_rejects_mismatched_index_or_columns(two_members):
    runs_dir, a, _ = two_members
    _save(runs_dir, "short", a.iloc[1:])
    with pytest.raises(ValueError, match="short"):
        average_runs(runs_dir, ["a", "short"])
    _save(runs_dir, "cols", a.iloc[:, ::-1])
    with pytest.raises(ValueError, match="cols"):
        average_runs(runs_dir, ["a", "cols"])


@pytest.mark.parametrize("weights", [[0.5, 0.6], [1.5, -0.5], [1.0], [0.5, 0.25, 0.25]])
def test_average_rejects_bad_weights(two_members, weights):
    runs_dir, _, _ = two_members
    with pytest.raises(ValueError, match="weights"):
        average_runs(runs_dir, ["a", "b"], weights)


def test_average_rejects_no_members(tmp_path):
    with pytest.raises(ValueError, match="members"):
        average_runs(tmp_path, [])


def test_ensemble_config_runs_through_main(two_members, monkeypatch):
    runs_dir, a, b = two_members

    def no_surface(include_holdout: bool = False):
        raise AssertionError("an ensemble does not need the surface")

    monkeypatch.setattr(run, "load_surface", no_surface)
    config = runs_dir / "ens.toml"
    config.write_text(
        'name = "ens"\nkind = "ensemble"\n\n[params]\nmembers = ["a", "b"]\nweights = [0.5, 0.5]\n'
    )
    run_dir = run.main([str(config), "--runs-dir", str(runs_dir)])
    assert run_dir == runs_dir / "ens"
    preds, meta = load_run(run_dir)
    assert_frame_equal(preds, (a + b) / 2, check_freq=False)
    assert [m["name"] for m in meta["members"]] == ["a", "b"]
    assert {"commit", "timestamp", "runtime_seconds", "include_holdout", "fit_log"} <= meta.keys()
    assert (run_dir / "config.toml").read_text() == config.read_text()
    json.dumps(meta)


def test_shipped_ensemble_configs():
    cnn_ar1 = tomllib.loads((CONFIGS / "ens_cnn_ar1.toml").read_text())
    assert cnn_ar1["kind"] == "ensemble" and cnn_ar1["name"] == "ens_cnn_ar1"
    assert cnn_ar1["params"]["members"] == ["cnn_conv_head", "ar1_rolling252_huber"]
    assert cnn_ar1["params"]["weights"] == [0.5, 0.5]
    all_ = tomllib.loads((CONFIGS / "ens_all.toml").read_text())
    assert all_["name"] == "ens_all"
    assert all_["params"]["members"] == ["cnn_conv_head", "lstm", "convlstm", "ar1_rolling252_huber"]
    assert all_["params"]["weights"] == [0.25] * 4


def test_ensemble_missing_member_publishes_nothing(tmp_path):
    config = tmp_path / "ens.toml"
    config.write_text('name = "ens"\nkind = "ensemble"\n[params]\nmembers = ["nope"]\n')
    with pytest.raises(FileNotFoundError):
        run.main([str(config), "--runs-dir", str(tmp_path / "runs")])
    assert not (tmp_path / "runs" / "ens").exists()


def test_average_rejects_mixed_include_holdout(two_members):
    runs_dir, a, _ = two_members
    _save(runs_dir, "h", a, holdout=True)
    with pytest.raises(ValueError, match="include_holdout"):
        average_runs(runs_dir, ["a", "h"])


def test_average_rejects_mixed_test_start(two_members):
    runs_dir, a, _ = two_members
    _save(runs_dir, "t", a, test_start="2022-01-03")
    with pytest.raises(ValueError, match="test_start"):
        average_runs(runs_dir, ["a", "t"])


def _ens_config(runs_dir: Path, members: list[str]) -> Path:
    config = runs_dir / "ens.toml"
    names = ", ".join(f'"{m}"' for m in members)
    config.write_text(f'name = "ens"\nkind = "ensemble"\n[params]\nmembers = [{names}]\n')
    return config


def test_ensemble_records_members_include_holdout(tmp_path, synthetic_surface):
    a = synthetic_surface.loc["2020-01-01":]
    _save(tmp_path, "a", a, holdout=True)
    _save(tmp_path, "b", a, holdout=True)
    config = _ens_config(tmp_path, ["a", "b"])
    _, meta = load_run(run.main([str(config), "--runs-dir", str(tmp_path)]))
    assert meta["include_holdout"] is True  # taken from the members, without --holdout
    _, meta = load_run(run.main([str(config), "--runs-dir", str(tmp_path), "--holdout"]))
    assert meta["include_holdout"] is True


def test_ensemble_holdout_flag_contradicting_members_raises(two_members):
    runs_dir, _, _ = two_members
    config = _ens_config(runs_dir, ["a", "b"])
    with pytest.raises(ValueError, match="holdout"):
        run.main([str(config), "--runs-dir", str(runs_dir), "--holdout"])
    assert not (runs_dir / "ens").exists()
