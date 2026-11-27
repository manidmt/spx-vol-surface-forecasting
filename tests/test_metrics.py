import numpy as np
import pandas as pd
import pytest

from volsurf.metrics import (
    common_mask,
    diebold_mariano,
    dm_table,
    errors,
    mae_by_point,
    mae_by_tenor,
    rmse_by_point,
    rmse_by_tenor,
    summary,
)

PREDS_PATH = "data/processed/surface_preds.pkl"


def _frame(index, value=1.0):
    cols = pd.MultiIndex.from_product([["30DAY", "60DAY"], [90.0, 100.0]], names=["tenor", "moneyness"])
    return pd.DataFrame(value, index=index, columns=cols)


def test_common_mask_intersects_dates():
    actual = _frame(pd.bdate_range("2021-01-01", periods=10))
    p1 = _frame(pd.bdate_range("2021-01-04", periods=6))
    p2 = _frame(pd.bdate_range("2021-01-06", periods=8))
    p2.iloc[3] = np.nan  # 2021-01-11
    mask = common_mask(actual, {"a": p1, "b": p2})
    expected = pd.bdate_range("2021-01-06", "2021-01-11").drop(pd.Timestamp("2021-01-11"))
    assert list(mask) == list(expected)


def test_summary_hand_case():
    idx = pd.bdate_range("2021-01-04", periods=2)
    cols = pd.MultiIndex.from_product([["30DAY"], [90.0, 100.0]], names=["tenor", "moneyness"])
    e = pd.DataFrame([[1.0, -1.0], [2.0, 0.0]], index=idx, columns=cols)
    ref = pd.DataFrame(2.0, index=idx, columns=cols)
    out = summary({"persistence": ref, "m": e})
    assert out.loc["m", "mae"] == pytest.approx(1.0)
    assert out.loc["m", "rmse"] == pytest.approx(np.sqrt(1.5))
    assert out.loc["m", "mae_vs_ref"] == pytest.approx(-0.5)
    assert out.loc["m", "r2_oos_vs_ref"] == pytest.approx(1 - 6 / 16)


def test_by_tenor_and_point_preserve_order():
    idx = pd.bdate_range("2021-01-04", periods=3)
    cols = pd.MultiIndex.from_product([["60DAY", "30DAY"], [100.0, 90.0]], names=["tenor", "moneyness"])
    errs = {"m": pd.DataFrame(1.0, index=idx, columns=cols)}
    assert list(mae_by_tenor(errs).columns) == ["60DAY", "30DAY"]
    assert mae_by_point(errs).index.equals(cols)


def test_rmse_by_tenor_and_point_hand_case():
    idx = pd.bdate_range("2021-01-04", periods=2)
    cols = pd.MultiIndex.from_product([["60DAY", "30DAY"], [100.0, 90.0]], names=["tenor", "moneyness"])
    e = pd.DataFrame([[1.0, 3.0, 0.0, 2.0], [1.0, -1.0, 0.0, 0.0]], index=idx, columns=cols)
    by_tenor = rmse_by_tenor({"m": e})
    assert list(by_tenor.columns) == ["60DAY", "30DAY"]
    assert by_tenor.loc["m", "60DAY"] == pytest.approx(np.sqrt(12 / 4))
    assert by_tenor.loc["m", "30DAY"] == pytest.approx(np.sqrt(4 / 4))
    by_point = rmse_by_point({"m": e})
    assert by_point.index.equals(cols)
    np.testing.assert_allclose(by_point["m"], np.sqrt([1.0, 5.0, 0.0, 2.0]))


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-04", periods=300)
    ref = _frame(idx) * rng.normal(size=(300, 1)) * 2
    better = ref * 0.5
    t, p = diebold_mariano(ref, better, "absolute")
    assert t > 0 and p < 0.05
    with pytest.raises(ValueError):
        diebold_mariano(ref, better, "cubic")  # type: ignore[arg-type]


def test_dm_table_shape():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-04", periods=100)
    errs = {n: _frame(idx) * rng.normal(size=(100, 1)) for n in ["persistence", "a", "b"]}
    t = dm_table(errs, ["persistence"], ["a", "b"])
    assert t.index.names == ["reference", "model", "loss"]
    assert list(t.columns) == ["t_stat", "p_value"]
    assert len(t) == 4


def _actual_and_pred():
    idx = pd.bdate_range("2021-01-04", periods=5)
    rng = np.random.default_rng(2)
    actual = _frame(idx) + rng.normal(size=(5, 4))
    pred = _frame(idx) + rng.normal(size=(5, 4))
    return actual, pred


def test_errors_reject_duplicate_prediction_dates():
    actual, pred = _actual_and_pred()
    duplicated = pd.concat([pred, pred.iloc[[2]]]).sort_index()
    with pytest.raises(ValueError, match="m.*duplicate dates"):
        errors(actual, {"m": duplicated})
    with pytest.raises(ValueError, match="m.*duplicate dates"):
        common_mask(actual, {"m": duplicated})


def test_errors_reject_duplicate_actual_dates():
    actual, pred = _actual_and_pred()
    with pytest.raises(ValueError, match="actual.*duplicate dates"):
        errors(pd.concat([actual, actual.iloc[[0]]]).sort_index(), {"m": pred})


def test_errors_reject_duplicate_columns():
    actual, pred = _actual_and_pred()
    doubled = pd.concat([pred, pred.iloc[:, [0]]], axis=1)
    with pytest.raises(ValueError, match="m.*duplicate column"):
        errors(actual, {"m": doubled})
    with pytest.raises(ValueError, match="actual.*duplicate column"):
        errors(pd.concat([actual, actual.iloc[:, [0]]], axis=1), {"m": pred})


def test_errors_reject_missing_grid_point():
    actual, pred = _actual_and_pred()
    with pytest.raises(ValueError, match=r"m.*missing.*60DAY.*100"):
        errors(actual, {"m": pred.drop(columns=[("60DAY", 100.0)])})
    with pytest.raises(ValueError, match="m.*missing"):
        common_mask(actual, {"m": pred.drop(columns=[("60DAY", 100.0)])})


def test_errors_reject_extra_grid_point():
    actual, pred = _actual_and_pred()
    extra = pred.copy()
    extra[("90DAY", 100.0)] = 1.0
    with pytest.raises(ValueError, match=r"m.*extra.*90DAY"):
        errors(actual, {"m": extra})


def test_errors_align_reordered_grid_to_actual_order():
    actual, pred = _actual_and_pred()
    reordered = pred[pred.columns[::-1]]
    out = errors(actual, {"m": reordered, "ref": pred})
    assert out["m"].columns.equals(actual.columns)
    pd.testing.assert_frame_equal(out["m"], out["ref"])


def test_errors_reject_empty_common_sample():
    actual, pred = _actual_and_pred()
    later = pred.copy()
    later.index = later.index + pd.Timedelta(days=30)
    with pytest.raises(ValueError, match="no common"):
        errors(actual, {"m": later})
    all_nan = pred.copy()
    all_nan.iloc[:, 0] = np.nan
    with pytest.raises(ValueError, match="no common"):
        errors(actual, {"m": all_nan})


@pytest.mark.slow
def test_reproduces_notebook_tables(real_surface):
    preds = pd.read_pickle(PREDS_PATH)
    mask = common_mask(real_surface, preds)
    assert len(mask) == 1444
    assert mask[0] == pd.Timestamp("2021-01-04")
    assert mask[-1] == pd.Timestamp("2026-10-07")
    errs = errors(real_surface, preds)
    s = summary(errs)
    expected = {
        "persistence": (0.4597, 0.8277, 0.0, 0.0),
        "ar1_rolling252_huber": (0.4571, 0.8218, -0.0057, 0.0143),
        "cnn_conv_head": (0.4546, 0.8241, -0.0112, 0.0089),
        "cnn_dense_head": (0.4546, 0.8231, -0.0112, 0.0111),
    }
    for name, vals in expected.items():
        np.testing.assert_allclose(s.loc[name].to_numpy(), vals, atol=5e-5)
    t, p = diebold_mariano(errs["persistence"], errs["cnn_conv_head"], "absolute")
    assert t == pytest.approx(1.894, abs=1e-3)
    assert p == pytest.approx(0.058, abs=1e-3)
    t2, _ = diebold_mariano(errs["persistence"], errs["cnn_conv_head"], "squared")
    assert t2 == pytest.approx(0.500, abs=1e-3)
    t3, p3 = diebold_mariano(errs["persistence"], errs["ar1_rolling252_huber"], "absolute")
    assert (t3, p3) == (pytest.approx(1.241, abs=1e-3), pytest.approx(0.215, abs=1e-3))
