"""The runner's exogenous branch and the round 2 configs."""

import json
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsurf import run
from volsurf.io import load_run
from volsurf.models.cnn import make_cnn
from volsurf.protocol import walk_forward

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TEST_START = "2021-03-01"
TINY_CNN = 'head = "conv"\nlookback = 5\nseeds = 1\nepochs = 2\npatience = 1\nchannels = 4\ndevice = "cpu"\n'


@pytest.fixture
def synthetic_data(monkeypatch, synthetic_surface):
    monkeypatch.setattr(run, "load_surface", lambda include_holdout=False: synthetic_surface)


@pytest.fixture
def fake_features(monkeypatch):
    """Stand-in for ``volsurf.features`` whose ``build_exog`` records its calls.

    Installed in ``sys.modules``, so the runner's lazy import finds it whether or not the real
    module exists.
    """
    calls: list[tuple[list[str], pd.DatetimeIndex]] = []

    def build_exog(blocks: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
        calls.append((list(blocks), index.copy()))
        rng = np.random.default_rng(4)
        return pd.DataFrame(rng.normal(size=(len(index), 2)), index=index, columns=["f1", "f2"])

    module = types.ModuleType("volsurf.features")
    module.build_exog = build_exog
    monkeypatch.setitem(sys.modules, "volsurf.features", module)
    return calls, build_exog


def _write(tmp_path: Path, kind: str, params: str, name: str = "exog_run") -> Path:
    config = tmp_path / f"{name}.toml"
    config.write_text(f'name = "{name}"\nkind = "{kind}"\ntest_start = "{TEST_START}"\n\n[params]\n{params}')
    return config


@pytest.mark.parametrize(
    "settings",
    [{}, {"exog_scaling": "robust", "exog_clip": 3.0, "exog_l2": [0.0, 0.1]}],
)
def test_run_with_exog(tmp_path, synthetic_data, synthetic_surface, fake_features, settings):
    calls, build_exog = fake_features
    params = f'{TINY_CNN}deterministic = true\nexog = ["calendar", "underlying"]\n'
    params += "".join(f"{key} = {json.dumps(value)}\n" for key, value in settings.items())
    run_dir = run.main([str(_write(tmp_path, "cnn", params)), "--runs-dir", str(tmp_path / "runs")])
    assert len(calls) == 1
    assert calls[0][0] == ["calendar", "underlying"]
    pd.testing.assert_index_equal(calls[0][1], synthetic_surface.index)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["exog"] == {"blocks": ["calendar", "underlying"], "features": ["f1", "f2"]}
    model = make_cnn(
        "exog_run", head="conv", lookback=5, seeds=1, epochs=2, patience=1, channels=4, device="cpu",
        deterministic=True, n_exog=2, **settings,
    )  # fmt: skip
    panel = build_exog([], synthetic_surface.index)
    expected = walk_forward(model, synthetic_surface, pd.Timestamp(TEST_START), exog=panel)
    preds, _ = load_run(run_dir)
    np.testing.assert_array_equal(preds.to_numpy(), expected.to_numpy())


def test_run_without_exog_records_none(tmp_path, synthetic_data, fake_features):
    calls, _ = fake_features
    run_dir = run.main([str(_write(tmp_path, "persistence", "")), "--runs-dir", str(tmp_path / "runs")])
    assert calls == []
    assert "exog" not in json.loads((run_dir / "meta.json").read_text())


@pytest.mark.parametrize(
    ("kind", "params", "match"),
    [
        ("persistence", 'exog = ["calendar"]\n', "exog.*neural"),
        ("ar1", 'exog = ["calendar"]\n', "exog.*neural"),
        ("cnn", f"{TINY_CNN}exog = []\n", "exog.*non-empty list"),
        ("cnn", f'{TINY_CNN}exog = "calendar"\n', "exog.*non-empty list"),
        ("cnn", f'{TINY_CNN}n_exog = 2\nexog = ["calendar"]\n', "n_exog"),
    ],
)
def test_run_rejects_bad_exog(tmp_path, synthetic_data, fake_features, kind, params, match):
    with pytest.raises(ValueError, match=match):
        run.main([str(_write(tmp_path, kind, params)), "--runs-dir", str(tmp_path / "runs")])
    assert fake_features[0] == []
    assert not (tmp_path / "runs").exists()


ROUND2 = {
    "cnn_conv_head": {"_cal": ["calendar"], "_und": ["underlying"], "_calund": ["calendar", "underlying"]},
    "convlstm": {"_cal": ["calendar"], "_und": ["underlying"], "_calund": ["calendar", "underlying"]},
}


@pytest.mark.parametrize(
    ("base", "suffix"), [(base, suffix) for base, runs in ROUND2.items() for suffix in runs]
)
def test_round2_configs_add_only_exog(base, suffix):
    reference = tomllib.loads((CONFIGS / f"{base}.toml").read_text())
    config = tomllib.loads((CONFIGS / f"{base}{suffix}.toml").read_text())
    assert config["name"] == f"{base}{suffix}"
    params = dict(config["params"])
    assert params.pop("exog") == ROUND2[base][suffix]
    assert {**config, "params": params, "name": base} == reference


@pytest.mark.parametrize("base", ["cnn_conv_head", "convlstm"])
@pytest.mark.parametrize("block", ["rates", "all"])
@pytest.mark.parametrize("variant", ["robust", "ridge"])
def test_regularised_configs_add_only_exog_settings(base, block, variant):
    name = f"{base}_{block}_{variant}"
    reference = tomllib.loads((CONFIGS / f"{base}.toml").read_text())
    config = tomllib.loads((CONFIGS / f"{name}.toml").read_text())
    exog = (
        ["rates"]
        if block == "rates"
        else ["calendar", "underlying", "rates", "credit", "cross_vol", "sectors"]
    )
    extra = {"exog": exog, "exog_scaling": "robust", "exog_clip": 3.0}
    if variant == "ridge":
        extra["exog_l2"] = [0.0, 0.001, 0.01, 0.1, 1.0]
    assert config == {**reference, "name": name, "params": reference["params"] | extra}
