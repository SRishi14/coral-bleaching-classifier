"""
Offline synthetic generator for NOAA Coral Reef Watch (CRW)-like products.

The goal is *faithfulness*, not realism for its own sake: the generated cube has
the same variables, dimensions, units and physical relationships as the real CRW
5 km product suite, so the entire downstream pipeline (features -> models ->
dashboard) is identical whether you run on synthetic or real data.

Physics we reproduce
---------------------
* **SST** = base climate + latitude gradient + seasonal cycle + slow warming
  trend + an ENSO-like low-frequency oscillation + injected marine-heatwave
  events + small daily noise.
* **MMM** (Maximum Monthly Mean climatology) = the warmest climatological month,
  computed per pixel from the deterministic seasonal climatology (no trend /
  anomalies), exactly as NOAA defines its bleaching-threshold baseline.
* **SST anomaly** = SST - daily climatology.
* **HotSpot** = max(SST - MMM, 0)  — positive only when warmer than the warmest
  *normal* month.
* **DHW** (Degree Heating Weeks) = trailing 84-day (12-week) accumulation of
  HotSpots >= 1 degC, divided by 7 to convert degC-days to degC-weeks.
* **BAA** (Bleaching Alert Area, 0..4) derived from HotSpot/DHW by NOAA's rule.

Because heat stress (HotSpot >= 1) only occurs during anomalously warm summers,
bleaching alerts are *rare* — which is exactly the class-imbalance challenge the
modeling code is built to handle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import config


# --------------------------------------------------------------------------- #
# Grid + time construction
# --------------------------------------------------------------------------- #
def _region_grid(region_key: str, approx_res_deg: float = 0.6):
    """Build a modest lat/lon grid (<= ~10 cells per axis) for a region box."""
    reg = config.REGIONS[region_key]
    lat0, lat1 = sorted(reg["lat"])
    lon0, lon1 = sorted(reg["lon"])

    def _axis(a0: float, a1: float) -> np.ndarray:
        n = int(round((a1 - a0) / approx_res_deg)) + 1
        n = max(4, min(10, n))
        return np.linspace(a0, a1, n)

    return _axis(lat0, lat1), _axis(lon0, lon1)


def _seasonal_peak_doy(lat: float) -> float:
    """Day-of-year of warmest water: late summer in each hemisphere."""
    return 50.0 if lat < 0 else 240.0  # ~mid-Feb (S) vs ~late-Aug (N)


# --------------------------------------------------------------------------- #
# Core generator
# --------------------------------------------------------------------------- #
def synthetic_region_cube(region_key: str) -> xr.Dataset:
    """
    Generate a CRW-like :class:`xarray.Dataset` for ``region_key``.

    Returns a cube with dims ``(time, latitude, longitude)`` and data variables
    ``sst``, ``ssta``, ``hotspot``, ``dhw`` (float) and ``baa`` (int 0..4).
    """
    rng = np.random.default_rng(config.RANDOM_STATE)

    lats, lons = _region_grid(region_key)
    times = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    nT, nY, nX = len(times), len(lats), len(lons)

    doy = times.dayofyear.to_numpy().astype(float)
    years_elapsed = (times - times[0]).days.to_numpy().astype(float) / 365.25

    # Basin-scale ENSO-like signal: shared across the region (random phase/period).
    enso_period_yr = rng.uniform(3.0, 4.5)
    enso_phase = rng.uniform(0, 2 * np.pi)
    enso = 0.7 * np.sin(2 * np.pi * years_elapsed / enso_period_yr + enso_phase)

    # Marine-heatwave events: a handful of warm summers, magnitude 2-3.5 degC.
    # Seeded years deliberately span the timeline so both the (past) train split
    # and the (recent) test split contain alert events.
    start_year = times[0].year
    end_year = times[-1].year
    candidate_years = list(range(start_year + 1, end_year + 1))
    n_events = max(3, len(candidate_years) // 3)
    event_years = np.sort(rng.choice(candidate_years, size=n_events, replace=False))

    heatwave = np.zeros(nT)
    for yr in event_years:
        # Centre the bump near that hemisphere's warmest day, that calendar year.
        peak = _seasonal_peak_doy(float(np.mean(lats)))
        center = pd.Timestamp(year=int(yr), month=1, day=1) + pd.Timedelta(days=peak)
        t_days = (times - center).days.to_numpy().astype(float)
        sigma = rng.uniform(28, 55)
        mag = rng.uniform(2.0, 3.5)
        heatwave += mag * np.exp(-0.5 * (t_days / sigma) ** 2)

    # ----- Deterministic per-pixel climatology + full SST field ----- #
    sst = np.empty((nT, nY, nX), dtype=np.float32)
    clim = np.empty((nT, nY, nX), dtype=np.float32)   # daily climatology (no trend/noise)
    mmm = np.empty((nY, nX), dtype=np.float32)         # max monthly mean per pixel

    warming_per_yr = 0.025  # degC/yr long-term trend

    # A representative climatological year (for MMM via monthly means).
    clim_doy = np.arange(1, 366, dtype=float)
    clim_months = (pd.Timestamp("2001-01-01") + pd.to_timedelta(clim_doy - 1, "D")).month

    for j, lat in enumerate(lats):
        base = 28.0 - 0.15 * (abs(lat) - 15.0)          # warmer near the tropics
        amp = 1.5 + 0.10 * abs(lat)                     # bigger swing at higher lat
        peak = _seasonal_peak_doy(lat)
        for i, lon in enumerate(lons):
            # tiny spatial offset so the map isn't perfectly uniform
            spatial = 0.10 * np.sin(np.radians(lon * 3)) + 0.05 * (i - nX / 2)

            seasonal = amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
            clim_ij = base + spatial + seasonal
            clim[:, j, i] = clim_ij

            noise = rng.normal(0, 0.25, size=nT)
            sst[:, j, i] = (
                clim_ij
                + warming_per_yr * years_elapsed
                + enso
                + heatwave
                + noise
            ).astype(np.float32)

            # MMM = warmest monthly mean of the deterministic climatology
            clim_year = base + spatial + amp * np.cos(2 * np.pi * (clim_doy - peak) / 365.25)
            monthly = pd.Series(clim_year).groupby(clim_months).mean()
            mmm[j, i] = float(monthly.max())

    # ----- Derived CRW products ----- #
    ssta = (sst - clim).astype(np.float32)                  # anomaly vs climatology
    hotspot = np.clip(sst - mmm[None, :, :], 0.0, None).astype(np.float32)

    # DHW: trailing 84-day sum of HotSpots >= 1 degC, in degC-weeks (sum/7).
    hs_acc = np.where(hotspot >= 1.0, hotspot, 0.0)
    window = 84
    cs = np.cumsum(hs_acc, axis=0)
    dhw = cs.copy()
    dhw[window:] = cs[window:] - cs[:-window]
    dhw = (dhw / 7.0).astype(np.float32)

    baa = _baa_from_stress(hotspot, dhw)

    ds = xr.Dataset(
        data_vars=dict(
            sst=(["time", "latitude", "longitude"], sst),
            ssta=(["time", "latitude", "longitude"], ssta),
            hotspot=(["time", "latitude", "longitude"], hotspot),
            dhw=(["time", "latitude", "longitude"], dhw),
            baa=(["time", "latitude", "longitude"], baa),
        ),
        coords=dict(time=times, latitude=lats, longitude=lons),
        attrs=dict(
            title="Synthetic NOAA CRW-like product suite",
            source="src/synthetic_data.py (offline generator)",
            region=config.REGIONS[region_key]["name"],
            note="HotSpot=max(SST-MMM,0); DHW=84-day accumulation/7; BAA per NOAA rule.",
            marine_heatwave_years=", ".join(map(str, event_years.tolist())),
        ),
    )
    # Per-variable metadata (units), mirroring CRW conventions.
    ds["sst"].attrs.update(units="degree_C", long_name="Sea Surface Temperature")
    ds["ssta"].attrs.update(units="degree_C", long_name="SST Anomaly")
    ds["hotspot"].attrs.update(units="degree_C", long_name="Coral Bleaching HotSpot")
    ds["dhw"].attrs.update(units="degree_C-weeks", long_name="Degree Heating Weeks")
    ds["baa"].attrs.update(units="1", long_name="Bleaching Alert Area (0..4)")
    ds["latitude"].attrs.update(units="degrees_north")
    ds["longitude"].attrs.update(units="degrees_east")
    return ds


def _baa_from_stress(hotspot: np.ndarray, dhw: np.ndarray) -> np.ndarray:
    """
    NOAA Bleaching Alert Area rule (0..4):

    ===  ============  ===========================================
    BAA  Name          Condition
    ===  ============  ===========================================
    0    No Stress     HotSpot <= 0
    1    Watch         0 < HotSpot < 1
    2    Warning       HotSpot >= 1 and 0 < DHW < 4
    3    Alert Level 1 HotSpot >= 1 and 4 <= DHW < 8
    4    Alert Level 2 HotSpot >= 1 and DHW >= 8
    ===  ============  ===========================================
    """
    baa = np.zeros(hotspot.shape, dtype=np.int8)
    watch = (hotspot > 0) & (hotspot < 1)
    stressed = hotspot >= 1.0
    baa[watch] = 1
    baa[stressed & (dhw < 4)] = 2
    baa[stressed & (dhw >= 4) & (dhw < 8)] = 3
    baa[stressed & (dhw >= 8)] = 4
    return baa
