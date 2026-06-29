"""
End-to-end pipeline: data -> features -> temporal split -> train -> evaluate -> save.

Examples
--------
    python train.py                              # offline, synthetic data (default)
    python train.py --region great_barrier_reef  # different reef
    python train.py --source erddap              # real NOAA Coral Reef Watch data
    python train.py --target multiclass          # full 0..4 alert scale

Outputs land in artifacts/ and are read by the dashboard (app.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import features as F          # noqa: E402
import models as M            # noqa: E402
import evaluate as E          # noqa: E402


def load_data(source: str, region_key: str):
    """Return (cube_or_None, region_dataframe)."""
    if source == "erddap":
        from data_download import download_region_cube, cube_to_region_dataframe
        print(f"[data] downloading NOAA CRW data for '{region_key}' ...")
        cube = download_region_cube(region_key)
        df = cube_to_region_dataframe(cube)
    else:
        from synthetic_data import synthetic_region_cube
        from data_download import cube_to_region_dataframe
        print(f"[data] generating synthetic CRW-like data for '{region_key}' ...")
        cube = synthetic_region_cube(region_key)
        df = cube_to_region_dataframe(cube)
    return cube, df


def main():
    ap = argparse.ArgumentParser(description="Coral Bleaching Risk Classifier")
    ap.add_argument("--source", choices=["synthetic", "erddap"], default="synthetic")
    ap.add_argument("--region", default=config.DEFAULT_REGION, choices=list(config.REGIONS))
    ap.add_argument("--target", choices=["binary", "multiclass"], default=config.TARGET_MODE)
    ap.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON_DAYS)
    args = ap.parse_args()

    config.TARGET_MODE = args.target  # honor CLI override downstream

    cube, df = load_data(args.source, args.region)

    # Persist the cube (for the dashboard's maps) and the region series
    cube_path = config.DATA_DIR / f"{args.region}_{args.source}.nc"
    try:
        cube.to_netcdf(cube_path)
        print(f"[data] saved cube -> {cube_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[data] (could not write NetCDF cube: {e})")
    df.to_csv(config.ARTIFACTS_DIR / "region_series.csv", index=False)

    # Features + temporal split
    X, y, dates = F.build_feature_table(df, horizon=args.horizon, target_mode=args.target)
    Xtr, Xte, ytr, yte, dtr, dte, split_date = F.temporal_split(X, y, dates)
    print(f"[split] train={len(Xtr)} test={len(Xte)} | split at {pd.Timestamp(split_date).date()}")
    print(f"[split] positive rate -> train {ytr.mean():.3f} | test {yte.mean():.3f}")

    results = {}
    preds = pd.DataFrame({"date": dte.to_numpy(), "y_true": yte.to_numpy()})

    # ---- XGBoost ----
    print("[train] XGBoost ...")
    xgb = M.train_xgboost(Xtr, ytr)
    xgb_pred = xgb.predict(Xte)
    preds["xgb_pred"] = xgb_pred
    results["xgboost"] = E.evaluate(yte, xgb_pred, args.target)
    E.confusion_df(yte, xgb_pred).to_csv(config.ARTIFACTS_DIR / "confusion_xgb.csv")
    print(E.text_report(yte, xgb_pred, args.target))

    # ---- PyTorch MLP ----
    print("[train] PyTorch MLP ...")
    mlp = M.train_torch_mlp(Xtr, ytr)
    mlp_pred = mlp.predict(Xte)
    preds["mlp_pred"] = mlp_pred
    results["torch_mlp"] = E.evaluate(yte, mlp_pred, args.target)
    E.confusion_df(yte, mlp_pred).to_csv(config.ARTIFACTS_DIR / "confusion_mlp.csv")
    print(E.text_report(yte, mlp_pred, args.target))

    # Save artifacts
    preds.to_csv(config.ARTIFACTS_DIR / "predictions.csv", index=False)
    meta = {"region": args.region, "source": args.source, "target": args.target,
            "horizon_days": args.horizon, "split_date": str(pd.Timestamp(split_date).date()),
            "metrics": results}
    (config.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(meta, indent=2))

    print("\n=== Macro-F1 (higher is better) ===")
    for name, m in results.items():
        print(f"  {name:10s}  macro_f1={m['macro_f1']:.3f}")
    print(f"\nArtifacts written to {config.ARTIFACTS_DIR}/  ->  run:  streamlit run app.py")


if __name__ == "__main__":
    main()
