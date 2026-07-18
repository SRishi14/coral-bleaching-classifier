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
# Map of heat stress on a chosen date (uses the saved NetCDF cube)
# --------------------------------------------------------------------------- #
cube_path = config.DATA_DIR / f"{meta['region']}_{meta['source']}.nc"
st.subheader("Study region — geographic heat-stress map")
if cube_path.exists():
    ds = xr.open_dataset(cube_path)
    map_dates = pd.to_datetime(ds["time"].values)

    layer_labels = {"baa": "Bleaching Alert Area (0-4)",
                    "dhw": "Degree Heating Weeks",
                    "ssta": "SST Anomaly",
                    "sst": "Sea Surface Temp",
                    "hotspot": "HotSpot"}
    # BAA first so the map opens on the actual alert level.
    layer_options = [v for v in ("baa", "dhw", "ssta", "sst", "hotspot") if v in ds]
    c_layer, c_date = st.columns([1, 2])
    var = c_layer.selectbox("Layer", layer_options, index=0,
                            format_func=lambda v: layer_labels[v])
    # Default the slider to the most heat-stressed day, so the map shows
    # something meaningful on load (the last calendar day is usually winter = 0).
    peak_idx = int(ds["dhw"].max(dim=["latitude", "longitude"]).values.argmax())
    picked = c_date.slider(
        "Date", min_value=map_dates[0].to_pydatetime(),
        max_value=map_dates[-1].to_pydatetime(),
        value=map_dates[peak_idx].to_pydatetime(), format="YYYY-MM-DD",
    )
    ti = int(np.abs(map_dates - pd.Timestamp(picked)).argmin())
    snap = ds[var].isel(time=ti)

    baa_colors = ["#3b93c4", "#f2e63d", "#f5a623", "#e63329", "#8b1a1a"]

    # ----- Geographic map: real coastlines + the exact box we trained/tested ----- #
    reg = config.REGIONS[meta["region"]]
    lat_b, lon_b = sorted(reg["lat"]), sorted(reg["lon"])
    LON, LAT = np.meshgrid(ds["longitude"].values, ds["latitude"].values)
    dfm = pd.DataFrame({"lat": LAT.ravel(), "lon": LON.ravel(),
                        "value": np.asarray(snap.values, dtype=float).ravel()}).dropna()

    extent = max(lat_b[1] - lat_b[0], lon_b[1] - lon_b[0])
    zoom = 7 if extent < 3 else 6 if extent < 7 else 5
    center = {"lat": float(np.mean(lat_b)), "lon": float(np.mean(lon_b))}
    npts = len(dfm)
    msize = 24 if npts <= 60 else 12 if npts <= 900 else 7

    if var == "baa":
        dfm["Alert"] = dfm["value"].round().clip(0, 4).astype(int).map(
            lambda k: f"{k} · {config.BAA_LABELS[k]}")
        cmap = {f"{k} · {config.BAA_LABELS[k]}": c for k, c in enumerate(baa_colors)}
        order = [f"{k} · {config.BAA_LABELS[k]}" for k in range(5)]
        geo = px.scatter_map(dfm, lat="lat", lon="lon", color="Alert",
                             color_discrete_map=cmap, category_orders={"Alert": order},
                             map_style="carto-darkmatter", zoom=zoom, center=center)
    else:
        geo = px.scatter_map(dfm, lat="lat", lon="lon", color="value",
                             color_continuous_scale="thermal" if var != "ssta" else "RdBu_r",
                             map_style="carto-darkmatter", zoom=zoom, center=center,
                             labels={"value": var.upper()})
    geo.update_traces(marker=dict(size=msize, opacity=0.8))
    # Outline the exact bounding box the model was trained/tested on.
    geo.add_trace(go.Scattermap(
        lat=[lat_b[0], lat_b[1], lat_b[1], lat_b[0], lat_b[0]],
        lon=[lon_b[0], lon_b[0], lon_b[1], lon_b[1], lon_b[0]],
        mode="lines", line=dict(width=2, color="#00e5ff"),
        name="Study region", hoverinfo="skip"))
    geo.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.02, x=0,
                                  bgcolor="rgba(0,0,0,0)"))
    st.caption(f"**{layer_labels[var]}** on **{map_dates[ti].date()}** over "
               f"{reg['name']} · box lat {lat_b} lon {lon_b} "
               f"(region peak DHW falls on {map_dates[peak_idx].date()}).")
    st.plotly_chart(geo, width="stretch")

    # ----- Plain lat/lon grid (kept for a clean pixel view) ----- #
    with st.expander("Show as a plain lat/lon grid"):
        if var == "baa":
            discrete = []
            for i, col in enumerate(baa_colors):
                discrete += [[i / 5, col], [(i + 1) / 5, col]]
            fig_map = px.imshow(snap.values, x=ds.longitude.values, y=ds.latitude.values,
                                origin="lower", aspect="auto",
                                color_continuous_scale=discrete, range_color=(-0.5, 4.5),
                                labels={"color": "BAA"})
            fig_map.update_coloraxes(colorbar=dict(
                title="Alert", tickvals=[0, 1, 2, 3, 4],
                ticktext=[f"{k} · {config.BAA_LABELS[k]}" for k in range(5)]))
        else:
            zmin = 0 if var in ("dhw", "hotspot") else None
            fig_map = px.imshow(snap.values, x=ds.longitude.values, y=ds.latitude.values,
                                origin="lower", aspect="auto", zmin=zmin,
                                color_continuous_scale="thermal" if var != "ssta" else "RdBu_r",
                                labels={"color": var.upper()})
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
