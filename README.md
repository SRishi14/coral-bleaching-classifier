# 🪸 Coral Bleaching Risk Classifier

**Forecasting satellite-derived coral-bleaching heat-stress alerts from NOAA Earth-observation data.**

When ocean water stays too warm for too long, corals expel their symbiotic algae
and bleach — often fatally. NOAA's [Coral Reef Watch](https://coralreefwatch.noaa.gov)
turns satellite sea-surface temperature into operational heat-stress products
(SST anomaly, *HotSpot*, *Degree Heating Weeks*, and a 0–4 *Bleaching Alert Area*).

This project builds an end-to-end machine-learning pipeline that learns from a
reef's recent temperature history and **forecasts its bleaching alert level ~4 weeks
ahead** — an early-warning signal for reef managers.

---

## Why this is a forecasting problem (and not a trivial one)

NOAA derives *today's* alert level directly from *today's* Degree Heating Weeks by a
fixed rule. Predicting today's alert from today's DHW would therefore be a data leak,
not science. Instead this project predicts the alert level **`FORECAST_HORIZON_DAYS`
in the future** using only information available now. That is genuinely useful and
genuinely hard.

Two modeling decisions follow from the nature of ocean data:

- **Temporal train/test split.** Ocean fields are strongly autocorrelated in time, so
  a random split would leak near-identical days across train and test and massively
  inflate the score. We train on the past and test on the future.
- **Imbalance-aware evaluation.** Bleaching-level stress is rare (most days are "no
  stress"), so accuracy is misleading. We weight the rare class during training and
  report **macro-F1** and the **recall on actual alerts**.

---

## What it does, end to end

1. **Data** — pulls the NOAA Coral Reef Watch 5 km product suite (SST, SST anomaly,
   HotSpot, DHW, Bleaching Alert Area) for a reef region from **CoastWatch ERDDAP**
   and loads it with **Xarray** (NetCDF). The real-data path is verified against the
   live NOAA servers — it reconstructs the catastrophic **2023 Florida Keys marine
   heatwave** (DHW climbing past 15 °C-weeks, Bleaching Alert Area 4). A faithful
   **synthetic generator** lets you build and demo everything offline.
2. **Features** — engineers lagged values, rolling mean/max/std, short-term rates of
   change, and seasonality terms from SST, SST anomaly, HotSpot, and DHW.
3. **Models** — trains a **gradient-boosting** model (XGBoost) *and* a small
   **PyTorch** neural network, both with class weighting.
4. **Evaluation** — macro-F1, per-class precision/recall, and confusion matrices on a
   held-out future period.
5. **Dashboard** — a **Streamlit + Plotly** app: a regional heat-stress map, the
   temperature/DHW time series with alert shading, and forecast-vs-actual.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

python train.py                 # offline demo on synthetic data (default)
streamlit run app.py            # open the dashboard
```

Use the **real NOAA data** (live CoastWatch ERDDAP):

```bash
python train.py --source erddap --region florida_keys
```

The live path downloads the recent window set by `ERDDAP_START_DATE/END_DATE` in
`config.py` (default 2023–2024, which captures the 2023 Florida Keys heatwave).
NOAA throttles griddap, so expect ~30 s per variable-year; widen the window in
`config.py` for a longer record. If a dataset ID has drifted or the network is
down, the run stops with a clear error pointing at the ERDDAP search page — it
never silently falls back to synthetic data.

Other options:

```bash
python train.py --region great_barrier_reef     # great_barrier_reef | florida_keys | hawaii
python train.py --target multiclass             # full 0..4 alert scale (default: binary)
python train.py --horizon 14                     # shorter lead time = easier
```

Run the tests:

```bash
python tests/test_pipeline.py
```

---

## Project layout

```
coral-bleaching-classifier/
├── config.py            # regions, ERDDAP ids, feature & model settings (all knobs here)
├── train.py             # end-to-end pipeline CLI
├── app.py               # Streamlit dashboard
├── src/
│   ├── synthetic_data.py  # CRW-like data generator (offline path)
│   ├── data_download.py   # NOAA CRW download via ERDDAP/OPeNDAP
│   ├── features.py        # feature engineering + temporal split
│   ├── models.py          # XGBoost + PyTorch MLP
│   └── evaluate.py        # imbalance-aware metrics
├── tests/test_pipeline.py
└── requirements.txt
```

The code is intentionally **modular and documented** — each stage is a small,
testable unit, and every tunable lives in `config.py`.

---

