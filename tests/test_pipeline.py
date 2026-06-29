"""
End-to-end smoke test of the pipeline:

    synthetic data -> region series -> features -> temporal split
                   -> train (sklearn stand-in) -> evaluate

Deliberately uses a scikit-learn classifier instead of XGBoost / PyTorch so the
test runs anywhere (the real models are exercised by ``train.py``). Also checks
the two things most likely to be silently wrong in EO time-series work:

  1. the temporal split never leaks future dates into training, and
  2. region subsetting is robust to longitude convention and latitude ordering.

Runnable two ways::

    python tests/test_pipeline.py      # plain script: prints PASS/FAIL
    pytest tests/test_pipeline.py      # discovered as test_* functions
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import features as F  # noqa: E402
import evaluate as E  # noqa: E402
from synthetic_data import synthetic_region_cube  # noqa: E402
from data_download import cube_to_region_dataframe, subset_region  # noqa: E402


def test_synthetic_cube_is_well_formed():
    cube = synthetic_region_cube("florida_keys")
    for v in ("sst", "ssta", "hotspot", "dhw", "baa"):
        assert v in cube, f"missing variable {v}"
    assert set(cube.dims) >= {"time", "latitude", "longitude"}
    baa = cube["baa"].values
    assert baa.min() >= 0 and baa.max() <= 4, "BAA outside 0..4"
    assert (cube["hotspot"].values >= 0).all(), "HotSpot must be non-negative"
    print("  [ok] synthetic cube well-formed:", dict(cube.sizes))


def test_region_dataframe_columns():
    cube = synthetic_region_cube("hawaii")
    df = cube_to_region_dataframe(cube)
    for col in ("date", "sst", "ssta", "hotspot", "dhw", "baa"):
        assert col in df.columns, f"missing column {col}"
    assert df["date"].is_monotonic_increasing
    assert len(df) > 1000, "expected a multi-year daily series"
    print(f"  [ok] region dataframe: {len(df)} rows, cols={list(df.columns)}")


def test_features_have_no_nans_and_align():
    cube = synthetic_region_cube("great_barrier_reef")
    df = cube_to_region_dataframe(cube)
    X, y, dates = F.build_feature_table(df, horizon=28, target_mode="binary")
    assert len(X) == len(y) == len(dates) > 0
    assert not X.isna().any().any(), "features contain NaNs after dropping"
    assert set(np.unique(y)) <= {0, 1}, "binary target must be 0/1"
    # current-BAA must NOT be a feature (would leak the threshold rule)
    assert not any(c.startswith("baa") for c in X.columns), "BAA leaked into features"
    print(f"  [ok] features: X={X.shape}, positive_rate={y.mean():.3f}")


def test_temporal_split_has_no_leakage():
    cube = synthetic_region_cube("florida_keys")
    df = cube_to_region_dataframe(cube)
    X, y, dates = F.build_feature_table(df, horizon=28, target_mode="binary")
    Xtr, Xte, ytr, yte, dtr, dte, split_date = F.temporal_split(X, y, dates)
    assert dtr.max() <= dte.min(), "train dates must precede all test dates"
    assert len(Xte) > 0 and len(Xtr) > 0
    frac = len(Xte) / (len(Xtr) + len(Xte))
    assert abs(frac - config.TEST_FRACTION) < 0.02, "test fraction off target"
    print(f"  [ok] temporal split clean: split at {split_date.date()}, test_frac={frac:.2f}")


def test_train_and_evaluate_with_sklearn_standin():
    """sklearn stand-in keeps the test independent of xgboost/torch."""
    from sklearn.ensemble import RandomForestClassifier

    cube = synthetic_region_cube("florida_keys")
    df = cube_to_region_dataframe(cube)
    X, y, dates = F.build_feature_table(df, horizon=28, target_mode="binary")
    Xtr, Xte, ytr, yte, *_ = F.temporal_split(X, y, dates)

    clf = RandomForestClassifier(
        n_estimators=120, class_weight="balanced", random_state=config.RANDOM_STATE
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    metrics = E.evaluate(yte, pred, "binary")
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    cm = E.confusion_df(yte, pred)
    assert cm.values.sum() == len(yte), "confusion matrix must cover all test rows"
    print(
        f"  [ok] sklearn stand-in: macro_f1={metrics['macro_f1']:.3f} "
        f"alert_recall={metrics.get('alert_recall', float('nan')):.3f}"
    )


def test_subset_region_robust_to_conventions():
    """subset_region must handle 0..360 longitudes and descending latitudes."""
    # Build a tiny grid in 0..360 longitude and north->south latitude.
    lat = np.array([30.0, 25.0, 20.0, 15.0])          # descending
    lon = np.array([275.0, 280.0, 285.0])             # == -85..-75 in -180..180
    data = np.zeros((1, lat.size, lon.size), dtype="float32")
    ds = xr.Dataset(
        {"sst": (["time", "latitude", "longitude"], data)},
        coords={"time": [0], "latitude": lat, "longitude": lon},
    )
    # Request the Florida-ish box with NEGATIVE longitudes and ascending lats.
    out = subset_region(ds, lat_bounds=(20.0, 28.0), lon_bounds=(-82.0, -78.0))
    assert out["latitude"].size > 0 and out["longitude"].size > 0
    assert float(out["latitude"].min()) >= 20.0 and float(out["latitude"].max()) <= 28.0
    print(f"  [ok] subset robust: lat={out['latitude'].values}, lon={out['longitude'].values}")


# --------------------------------------------------------------------------- #
# Plain-script runner (no pytest required)
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [
        test_synthetic_cube_is_well_formed,
        test_region_dataframe_columns,
        test_features_have_no_nans_and_align,
        test_temporal_split_has_no_leakage,
        test_train_and_evaluate_with_sklearn_standin,
        test_subset_region_robust_to_conventions,
    ]
    failures = 0
    print(f"Running {len(tests)} pipeline tests...\n")
    for t in tests:
        print(f"- {t.__name__}")
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  [FAIL] {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"FAILED: {failures}/{len(tests)} test(s) failed.")
        return 1
    print(f"PASSED: all {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
