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

## How this maps to the WHOI role

| JD requirement | Where it shows up |
|---|---|
| Python, Git, modular & maintainable code | whole repo: small documented modules, config-driven |
| Statistics, time series, non-linear modeling | `features.py` (lags/rolling/anomalies), both models |
| Gradient boosting | `models.train_xgboost` |
| Deep learning (PyTorch / TensorFlow) | `models.TorchMLP` |
| Geospatial & scientific Python for multi-dim EO data | Xarray cubes, region subsetting in `data_download.py` |
| NetCDF / HDF5 | reads/writes NetCDF throughout |
| Large datasets + technical reporting | multi-year daily grids; this README + dashboard |
| Interactive visualization tools | `app.py` (Streamlit + Plotly) |
| *Advanced architectures (GNN/PINN)* | see "Extensions" — discussed, not yet built |

---

## Talking points for the interview

- *"I framed it as a 4-week forecast on purpose — predicting the current alert from
  current DHW would just relearn NOAA's threshold rule and leak the answer."*
- *"I split by time, not randomly, because ocean data is autocorrelated and a random
  split inflates scores. Train on the past, test on the future."*
- *"Alerts are rare, so I used class weighting and judged the model on macro-F1 and
  alert recall, not accuracy."*
- *"The same products are built from satellites plus ships and buoys — a real
  heterogeneous, multi-platform observational dataset, handled with Xarray."*
- *"The real-data plumbing had real-world failure modes: latitude grids that run
  north→south, a 0–360 vs −180–180 longitude convention, one product whose OPeNDAP
  stream is flaky (so I fetch via ERDDAP's `.nc` endpoint), and a proxy that 502s on
  multi-year requests (so I chunk by year and concatenate). Handling those is most of
  what working with operational Earth-observation data actually is."*

## Extensions (good answers to "what would you do next?")

- **Per-pixel spatial prediction** to produce a full forecast risk *map* (and a
  natural place for a **U-Net** / ConvLSTM).
- A **graph neural network** over reef sites, or a **physics-informed** term tying SST
  evolution to a heat-budget constraint — the "strong plus" architectures in the JD.
- Spatial block cross-validation; probability calibration; SHAP for interpretability.

---

## Data & credits

NOAA Coral Reef Watch daily global 5 km product suite (v3.1), distributed via
CoastWatch ERDDAP. NOAA CRW data are public domain. This is an independent learning
project and is not affiliated with NOAA or WHOI.
