"""
Feature engineering and the (crucial) temporal train/test split.

From the region-mean daily series of SST, SST anomaly, HotSpot and DHW we build
predictors that capture *recent history* and *seasonality*:

* **Lags** — the value 1, 3, 7, ... days ago.
* **Rolling mean / max / std** over several windows — level, peak and volatility
  of recent conditions.
* **Short-term deltas** — rate of change (warming/cooling) over 1, 3, 7 days.
* **Seasonality** — sin/cos of day-of-year, so the model knows where in the
  annual heat cycle "today" sits without treating the date as a magnitude.

The label is the Bleaching Alert Area ``FORECAST_HORIZON_DAYS`` in the *future*
(``baa`` shifted backwards), so every feature uses only information available on
the prediction date. See the README for *why* this framing avoids leaking
NOAA's own threshold rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Variables we engineer features from (BAA is the label source, not a feature —
# including current BAA would leak the answer through autocorrelation).
BASE_VARS = ("sst", "ssta", "hotspot", "dhw")
DELTA_DAYS = (1, 3, 7)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame (indexed like ``df``) with no target column."""
    df = df.sort_values("date").reset_index(drop=True)
    feats: dict[str, np.ndarray | pd.Series] = {}

    for var in BASE_VARS:
        if var not in df:
            continue
        s = df[var]
        feats[f"{var}_now"] = s
        for lag in config.LAG_DAYS:
            feats[f"{var}_lag{lag}"] = s.shift(lag)
        for w in config.ROLL_WINDOWS:
            roll = s.rolling(window=w, min_periods=w)
            feats[f"{var}_rmean{w}"] = roll.mean()
            feats[f"{var}_rmax{w}"] = roll.max()
            feats[f"{var}_rstd{w}"] = roll.std()
        for d in DELTA_DAYS:
            feats[f"{var}_delta{d}"] = s - s.shift(d)

    # Day-of-year seasonality (continuous, wrap-around safe).
    doy = df["date"].dt.dayofyear.to_numpy().astype(float)
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return pd.DataFrame(feats, index=df.index)


def make_target(df: pd.DataFrame, horizon: int, target_mode: str) -> pd.Series:
    """
    Build the forecast label: BAA ``horizon`` days ahead.

    * ``binary``     -> 1 if future BAA >= ``BAA_ALERT_THRESHOLD`` else 0.
    * ``multiclass`` -> the future BAA level itself (0..4).
    """
    future_baa = df.sort_values("date")["baa"].shift(-horizon)
    if target_mode == "binary":
        y = (future_baa >= config.BAA_ALERT_THRESHOLD).astype("float")
        y[future_baa.isna()] = np.nan  # keep horizon tail as NaN for dropping
        return y
    return future_baa  # NaN tail preserved


def build_feature_table(df: pd.DataFrame, horizon: int | None = None,
                        target_mode: str | None = None):
    """
    Assemble aligned ``(X, y, dates)``.

    Rows with NaNs introduced by lags/rolling windows (early period) or by the
    forecast shift (final ``horizon`` days, which have no known future) are
    dropped, so every returned row is fully observed.

    Returns
    -------
    X : pd.DataFrame   feature matrix
    y : pd.Series      integer labels
    dates : pd.Series  the prediction ("today") date for each row
    """
    horizon = config.FORECAST_HORIZON_DAYS if horizon is None else horizon
    target_mode = config.TARGET_MODE if target_mode is None else target_mode

    df = df.sort_values("date").reset_index(drop=True)
    X = make_features(df)
    y = make_target(df, horizon, target_mode)
    dates = df["date"]

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    dates = dates.loc[valid].reset_index(drop=True)
    return X, y, dates


def temporal_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                   test_fraction: float | None = None):
    """
    Split by **time**, not at random: the most-recent ``test_fraction`` of the
    timeline becomes the test set. Ocean fields are strongly autocorrelated, so a
    random split would place near-identical neighbouring days on both sides and
    massively inflate the score. We train on the past and forecast the future.

    Returns ``(Xtr, Xte, ytr, yte, dtr, dte, split_date)``.
    """
    test_fraction = config.TEST_FRACTION if test_fraction is None else test_fraction
    n = len(X)
    n_test = max(1, int(round(n * test_fraction)))
    split = n - n_test  # rows are already in chronological order

    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    dtr, dte = dates.iloc[:split], dates.iloc[split:]
    split_date = dates.iloc[split]
    return Xtr, Xte, ytr, yte, dtr, dte, split_date
