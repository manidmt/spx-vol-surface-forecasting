import pytest
from pandas.testing import assert_frame_equal

from volsurf.io import load_run, save_run
from volsurf.models import build_model


def test_run_roundtrip_preserves_columns(tmp_path, synthetic_surface):
    run_dir = tmp_path / "run"
    save_run(run_dir, synthetic_surface, {"source": 'name = "x"\n'}, {"commit": "abc"})
    preds, meta = load_run(run_dir)
    assert_frame_equal(preds, synthetic_surface, check_freq=False)
    assert meta == {"commit": "abc"}
    assert preds.columns.names == ["tenor", "moneyness"]
    assert preds.columns.get_level_values("moneyness").dtype == float


def test_build_model_unknown_kind():
    with pytest.raises(ValueError, match="persistence"):
        build_model("nope", "x", {})


def test_build_model_kinds():
    assert build_model("persistence", "p", {}).name == "persistence"
    ar1 = build_model("ar1", "my_ar1", {"window": 100})
    assert ar1.name == "my_ar1" and ar1.window == 100
    with pytest.raises(ValueError):
        build_model("persistence", "p", {"window": 3})
