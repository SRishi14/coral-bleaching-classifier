"""
Interactive dashboard for the Coral Bleaching Risk Classifier.

Run after training:
    python train.py
    streamlit run app.py

Shows: a map of the reef region's heat stress, the temperature/DHW time series
with alert shading, and the model's forecast vs. what actually happened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

import config

st.set_page_config(page_title="Coral Bleaching Risk", layout="wide")
ART = config.ARTIFACTS_DIR

st.title("🪸 Coral Bleaching Risk Classifier")
st.caption("Forecasting satellite-derived heat-stress alerts on coral reefs · NOAA Coral Reef Watch")

# --------------------------------------------------------------------------- #
# Load artifacts produced by train.py
# --------------------------------------------------------------------------- #
if not (ART / "metrics.json").exists():
    st.warning("No trained artifacts found. Run `python train.py` first.")
    st.stop()

meta = json.loads((ART / "metrics.json").read_text())
series = pd.read_csv(ART / "region_series.csv", parse_dates=["date"])
preds = pd.read_csv(ART / "predictions.csv", parse_dates=["date"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Region", config.REGIONS[meta["region"]]["name"].split("(")[0].strip())
c2.metric("Data source", meta["source"])
c3.metric("Forecast horizon", f"{meta['horizon_days']} days")
c4.metric("XGBoost macro-F1", f"{meta['metrics']['xgboost']['macro_f1']:.2f}")

# --------------------------------------------------------------------------- #
# Map of latest heat stress (uses the saved NetCDF cube)
# --------------------------------------------------------------------------- #
cube_path = config.DATA_DIR / f"{meta['region']}_{meta['source']}.nc"
st.subheader("Latest heat stress across the region")
if cube_path.exists():
    ds = xr.open_dataset(cube_path)
    var = st.selectbox("Layer", ["dhw", "ssta", "sst", "hotspot"], index=0,
                       format_func=lambda v: {"dhw": "Degree Heating Weeks",
                                              "ssta": "SST Anomaly",
                                              "sst": "Sea Surface Temp",
                                              "hotspot": "HotSpot"}[v])
    snap = ds[var].isel(time=-1)
    fig_map = px.imshow(
        snap.values,
        x=ds.longitude.values, y=ds.latitude.values,
        origin="lower", aspect="auto",
        color_continuous_scale="thermal" if var != "ssta" else "RdBu_r",
        labels={"color": var.upper()},
    )
    fig_map.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Longitude", yaxis_title="Latitude")
    st.plotly_chart(fig_map, width="stretch")
else:
    st.info("NetCDF cube not found — map skipped (time series still available).")

# --------------------------------------------------------------------------- #
# Time series with alert shading
# --------------------------------------------------------------------------- #
st.subheader("Temperature & accumulated heat stress (region mean)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=series["date"], y=series["sst"], name="SST (°C)",
                         line=dict(color="#1f77b4", width=1)))
fig.add_trace(go.Scatter(x=series["date"], y=series["dhw"], name="DHW (°C-weeks)",
                         yaxis="y2", line=dict(color="#d62728", width=1.5)))
fig.add_hline(y=4, line_dash="dot", line_color="orange", yref="y2",
              annotation_text="DHW=4 (Alert 1)")
fig.update_layout(
    height=360, margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(title="SST (°C)"),
    yaxis2=dict(title="DHW (°C-weeks)", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Forecast vs. actual on the held-out (future) test period
# --------------------------------------------------------------------------- #
st.subheader("Forecast vs. actual (held-out future period)")
model_choice = st.radio("Model", ["xgb_pred", "mlp_pred"],
                        format_func=lambda m: "XGBoost" if m == "xgb_pred" else "PyTorch MLP",
                        horizontal=True)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=preds["date"], y=preds["y_true"], name="Actual",
                          mode="lines", line=dict(color="#2ca02c", width=2)))
fig2.add_trace(go.Scatter(x=preds["date"], y=preds[model_choice], name="Predicted",
                          mode="lines", line=dict(color="#9467bd", width=1, dash="dot")))
fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                   yaxis_title="Bleaching risk", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2, width="stretch")

with st.expander("Model metrics (test set)"):
    st.json(meta["metrics"])
