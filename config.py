"""
Central configuration for the Coral Bleaching Risk Classifier.

Keeping all tunable settings in one place keeps the rest of the codebase
modular and easy to maintain (no magic numbers buried in the logic).
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # downloaded / generated NetCDF cubes
ARTIFACTS_DIR = ROOT / "artifacts"  # trained models, metrics, predictions
for _d in (DATA_DIR, ARTIFACTS_DIR):
    _d.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Reef regions (lat/lon bounding boxes, degrees; lon in -180..180)
# Pick one with --region <key>. Add your own freely.
# --------------------------------------------------------------------------- #
REGIONS: dict[str, dict] = {
    "great_barrier_reef": {
        "name": "Great Barrier Reef (Australia)",
        "lat": (-20.0, -14.0),
        "lon": (145.0, 150.0),
    },
    "florida_keys": {
        "name": "Florida Keys / Caribbean (USA)",
        "lat": (24.0, 26.5),
        "lon": (-82.5, -80.0),
    },
    "hawaii": {
        "name": "Main Hawaiian Islands (USA)",
        "lat": (19.0, 22.5),
        "lon": (-160.5, -154.5),
    },
}
DEFAULT_REGION = "florida_keys"

# --------------------------------------------------------------------------- #
# Date range for the historical record (CRW data starts 1985-01-01)
# --------------------------------------------------------------------------- #
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# The real ERDDAP download is network-bound (NOAA throttles griddap to a few
# hundred KB/s), so the live path defaults to a shorter recent window than the
# synthetic generator. This 2-year span includes the catastrophic 2023 Florida
# Keys marine heatwave. Widen it for a longer real record (expect ~30 s of
# download per variable-year).
ERDDAP_START_DATE = "2023-01-01"
ERDDAP_END_DATE = "2024-12-31"

# --------------------------------------------------------------------------- #
# NOAA Coral Reef Watch ERDDAP (CoastWatch).  These dataset IDs were correct at
# build time; if a download fails, open the ERDDAP search page printed by
# data_download.py and confirm the current IDs, or just use --source synthetic.
# --------------------------------------------------------------------------- #
ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap/griddap"
# Dataset IDs + variable names verified live against CoastWatch ERDDAP.
# Note: CoastWatch exposes the CRW suite under CF-standard variable names
# (e.g. "degree_heating_week"), NOT the CRW_* names used in NOAA's own NetCDF
# files. The Bleaching Alert Area is served as a 7-day product (baa7d).
ERDDAP_DATASETS = {
    "sst": "noaacrwsstDaily",            # CoralTemp SST
    "ssta": "noaacrwsstanomalyDaily",    # SST Anomaly
    "hotspot": "noaacrwhotspotDaily",    # Coral Bleaching HotSpot
    "dhw": "noaacrwdhwDaily",            # Degree Heating Weeks
    "baa": "noaacrwbaa7dDaily",          # Bleaching Alert Area (7-day)
}
ERDDAP_VARS = {
    "sst": "analysed_sst",
    "ssta": "sea_surface_temperature_anomaly",
    "hotspot": "hotspot",
    "dhw": "degree_heating_week",
    "baa": "bleaching_alert_area",
}

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
LAG_DAYS = [1, 3, 7, 14, 30, 60]          # how far back to look for each variable
ROLL_WINDOWS = [7, 30, 90]                 # rolling mean/max windows (days)
FORECAST_HORIZON_DAYS = 28                 # predict risk ~4 weeks ahead

# --------------------------------------------------------------------------- #
# Target definition
#   "binary"     -> 0 = no/low stress, 1 = bleaching-level heat stress (BAA >= 2)
#   "multiclass" -> the full 0..4 NOAA Bleaching Alert Area scale
# --------------------------------------------------------------------------- #
TARGET_MODE = "binary"
BAA_ALERT_THRESHOLD = 2  # BAA >= this counts as "bleaching risk" in binary mode

BAA_LABELS = {
    0: "No Stress",
    1: "Watch",
    2: "Warning",
    3: "Alert Level 1",
    4: "Alert Level 2",
}

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_FRACTION = 0.25      # most-recent 25% of the timeline = test set (temporal split)
TORCH_EPOCHS = 40
TORCH_LR = 1e-3
TORCH_HIDDEN = (64, 32)
