"""
Download the real NOAA Coral Reef Watch (CRW) 5 km daily product suite and turn a
gridded region cube into the region-mean time series the models consume.

Real data path
--------------
CRW products are served as ERDDAP *griddap* datasets by NOAA CoastWatch. Each
griddap dataset is also an OPeNDAP endpoint, so :func:`xarray.open_dataset` can
open it lazily over the network and we subset *before* anything is downloaded.

Robustness we care about (these bite people constantly with EO data):

* **Latitude ordering** — some grids run north->south, some south->north. We
  sort the requested bounds against the actual coordinate direction.
* **Longitude convention** — grids may use ``-180..180`` or ``0..360``. We detect
  which and translate the requested box accordingly.
* **Clear failures** — dataset IDs drift over time. If a fetch fails we raise a
  message pointing at the ERDDAP search page rather than silently faking data.
"""
from __future__ import annotations

import os
import tempfile
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

import config

# A browser-like UA — CoastWatch ERDDAP returns 403 to some default clients.
_HTTP_HEADERS = {"User-Agent": "coral-bleaching-classifier/1.0 (research)"}

ERDDAP_SEARCH_URL = (
    "https://coastwatch.noaa.gov/erddap/griddap/index.html?searchFor=coral+reef+watch"
)


# --------------------------------------------------------------------------- #
# Coordinate-name + subsetting helpers (work for either convention)
# --------------------------------------------------------------------------- #
def _coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f"None of {candidates} found in dataset coords {list(ds.coords)}")


def _slice_bounds(coord_vals: np.ndarray, lo: float, hi: float):
    """A slice from lo..hi that respects whether the coordinate ascends/descends."""
    ascending = coord_vals[0] <= coord_vals[-1]
    return slice(lo, hi) if ascending else slice(hi, lo)


def subset_region(ds: xr.Dataset, lat_bounds, lon_bounds) -> xr.Dataset:
    """
    Subset ``ds`` to a lat/lon box, robust to latitude ordering and to
    ``-180..180`` vs ``0..360`` longitude conventions.
    """
    latn = _coord_name(ds, ("latitude", "lat"))
    lonn = _coord_name(ds, ("longitude", "lon"))

    lat0, lat1 = sorted(lat_bounds)
    lon0, lon1 = sorted(lon_bounds)

    lons = np.asarray(ds[lonn].values, dtype=float)
    if float(np.nanmax(lons)) > 180.0:
        # Dataset is 0..360 -> translate any negative requested longitudes.
        lon0 = lon0 % 360.0
        lon1 = lon1 % 360.0
        lon0, lon1 = sorted((lon0, lon1))

    lat_sel = _slice_bounds(np.asarray(ds[latn].values, dtype=float), lat0, lat1)
    lon_sel = _slice_bounds(lons, lon0, lon1)
    out = ds.sel({latn: lat_sel, lonn: lon_sel})

    if out[latn].size == 0 or out[lonn].size == 0:
        raise ValueError(
            f"Region box lat={lat_bounds} lon={lon_bounds} selected an empty grid. "
            f"Check the dataset's coverage and longitude convention.\n{ERDDAP_SEARCH_URL}"
        )
    return out.rename({latn: "latitude", lonn: "longitude"})


# --------------------------------------------------------------------------- #
# Real download
# --------------------------------------------------------------------------- #
def _open_griddap(dataset_id: str) -> xr.Dataset:
    """
    Open one CRW griddap dataset over OPeNDAP.

    We try the ``netcdf4`` engine first (its built-in DAP client is the most
    reliable against CoastWatch ERDDAP) and fall back to ``pydap``.
    """
    url = f"{config.ERDDAP_BASE}/{dataset_id}"
    last_err: Exception | None = None
    for engine in ("netcdf4", "pydap"):
        try:
            return xr.open_dataset(url, engine=engine)
        except Exception as e:  # noqa: BLE001 -- fall through to next engine
            last_err = e
    raise RuntimeError(
        f"Could not open CRW dataset '{dataset_id}' over OPeNDAP ({config.ERDDAP_BASE}).\n"
        f"Underlying error: {last_err}\n"
        f"Confirm the dataset ID / your network, or browse: {ERDDAP_SEARCH_URL}"
    )


def _grid_metadata(dataset_id: str, var: str) -> dict:
    """
    Open a griddap dataset *once* and capture the small facts needed to build
    subset queries for it: coordinate names, the variable's dimension order, and
    each axis's stored direction / longitude convention. Reused across all the
    yearly chunks of that variable so we don't re-read coordinates every request.
    """
    meta = _open_griddap(dataset_id)
    try:
        if var not in meta:
            raise KeyError(
                f"Variable '{var}' not in dataset '{dataset_id}'. "
                f"Available: {list(meta.data_vars)}"
            )
        latn = _coord_name(meta, ("latitude", "lat"))
        lonn = _coord_name(meta, ("longitude", "lon"))
        timn = _coord_name(meta, ("time",))
        latv = np.asarray(meta[latn].values, dtype=float)
        lonv = np.asarray(meta[lonn].values, dtype=float)
        return {
            "latn": latn, "lonn": lonn, "timn": timn,
            "var_dims": tuple(meta[var].dims),
            "lat_ascending": bool(latv[0] <= latv[-1]),
            "lon_ascending": bool(lonv[0] <= lonv[-1]),
            "lon_is_360": bool(float(np.nanmax(lonv)) > 180.0),
        }
    finally:
        meta.close()


def _build_griddap_url(dataset_id: str, var: str, grid: dict,
                       lat_bounds, lon_bounds, time_bounds: tuple[str, str]) -> str:
    """
    Build an ERDDAP ``.nc`` griddap query URL for one space/time subset.

    griddap selects by coordinate *value* as ``[(start):stride:(stop)]`` per
    dimension, where ``start`` must sit at a lower index than ``stop`` — so each
    range is ordered to match how the dataset stores that axis (latitude often
    descends; longitude may be 0..360).
    """
    lat0, lat1 = sorted(lat_bounds)
    lon0, lon1 = sorted(lon_bounds)
    if grid["lon_is_360"]:
        lon0, lon1 = sorted((lon0 % 360.0, lon1 % 360.0))

    lat_q = (lat0, lat1) if grid["lat_ascending"] else (lat1, lat0)
    lon_q = (lon0, lon1) if grid["lon_ascending"] else (lon1, lon0)
    t0, t1 = time_bounds

    ranges = {
        grid["timn"]: f"[({t0}):1:({t1})]",
        grid["latn"]: f"[({lat_q[0]}):1:({lat_q[1]})]",
        grid["lonn"]: f"[({lon_q[0]}):1:({lon_q[1]})]",
    }
    # Constraint in the variable's own dim order; pin any extra singleton
    # dimension (e.g. altitude) to its first index.
    constraint = var + "".join(ranges.get(d, "[0:1:0]") for d in grid["var_dims"])
    query = urllib.parse.quote(constraint, safe="[]():,.-")
    return f"{config.ERDDAP_BASE}/{dataset_id}.nc?{query}"


def _http_download(url: str, dest: str, attempts: int = 3) -> None:
    """Download ``url`` to ``dest``, retrying transient server errors (502/timeout)."""
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            req = urllib.request.Request(url, headers=_HTTP_HEADERS)
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
                fh.write(resp.read())
            return
        except Exception as e:  # noqa: BLE001 -- transient 5xx / timeouts: retry
            last_err = e
    raise last_err  # type: ignore[misc]


def _yearly_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into <=1-year (start, end) windows (ISO date strings)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    windows: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        wend = min(pd.Timestamp(year=cur.year, month=12, day=31), e)
        windows.append((cur.strftime("%Y-%m-%d"), wend.strftime("%Y-%m-%d")))
        cur = wend + pd.Timedelta(days=1)
    return windows


def _fetch_variable(dataset_id: str, var: str, lat_bounds, lon_bounds,
                    full_bounds: tuple[str, str]) -> xr.Dataset:
    """
    Fetch a variable over the full date range by concatenating yearly chunks.

    ERDDAP's proxy rejects a single multi-year subset request (HTTP 502), so we
    request one year at a time and concatenate along ``time`` — the standard way
    to pull a long griddap series reliably. Dataset metadata is read once and
    reused for every chunk.
    """
    grid = _grid_metadata(dataset_id, var)
    tmp = os.path.join(tempfile.gettempdir(), f"crw_{dataset_id}_{var}.nc")
    parts = []
    for win in _yearly_windows(*full_bounds):
        url = _build_griddap_url(dataset_id, var, grid, lat_bounds, lon_bounds, win)
        _http_download(url, tmp)
        ds = xr.open_dataset(tmp).load()
        parts.append(subset_region(ds[[var]], lat_bounds, lon_bounds))
    combined = xr.concat(parts, dim="time")
    # Drop any duplicate timestamps at chunk boundaries, keep chronological order.
    _, keep = np.unique(combined["time"].values, return_index=True)
    return combined.isel(time=np.sort(keep))


def download_region_cube(region_key: str) -> xr.Dataset:
    """
    Fetch the five CRW variables for ``region_key`` and merge into one cube with
    dims ``(time, latitude, longitude)`` and vars ``sst/ssta/hotspot/dhw/baa``.

    Raises a clear, actionable error (pointing at the ERDDAP search page) if any
    product cannot be fetched — never fabricates data on failure.
    """
    reg = config.REGIONS[region_key]
    lat_bounds, lon_bounds = reg["lat"], reg["lon"]
    # The live download uses its own (shorter, recent) date window — see config.
    full_bounds = (config.ERDDAP_START_DATE, config.ERDDAP_END_DATE)

    merged = xr.Dataset()
    for key, dataset_id in config.ERDDAP_DATASETS.items():
        var = config.ERDDAP_VARS[key]
        try:
            print(f"[data]   fetching {key:8s} ({dataset_id}) ...", flush=True)
            sub = _fetch_variable(dataset_id, var, lat_bounds, lon_bounds, full_bounds)
            merged[key] = sub[var].astype("float32")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download CRW product '{key}' (var '{var}', dataset "
                f"'{dataset_id}') for region '{region_key}'.\n  -> {e}\n"
                f"Tip: dataset IDs change; verify current IDs at {ERDDAP_SEARCH_URL} "
                f"or run with --source synthetic."
            ) from e

    # BAA is an integer alert level; restore the dtype after the float cast above.
    if "baa" in merged:
        merged["baa"] = merged["baa"].round().clip(0, 4).astype("int8")
    merged.attrs.update(
        title="NOAA Coral Reef Watch 5km daily product suite (subset)",
        source=config.ERDDAP_BASE,
        region=reg["name"],
    )
    return merged


# --------------------------------------------------------------------------- #
# Cube -> region-mean time series (shared by both data paths)
# --------------------------------------------------------------------------- #
def cube_to_region_dataframe(cube: xr.Dataset) -> pd.DataFrame:
    """
    Collapse the spatial grid to one daily record per date.

    Continuous fields (SST, SSTA, HotSpot, DHW) are spatially averaged. The
    Bleaching Alert Area is reduced with the spatial **max**: a region is "in
    alert" if *any* part of it is — matching how reef managers read the maps.
    """
    spatial_dims = [d for d in ("latitude", "longitude") if d in cube.dims]
    df = pd.DataFrame({"date": pd.to_datetime(cube["time"].values)})

    for var in ("sst", "ssta", "hotspot", "dhw"):
        if var in cube:
            df[var] = cube[var].mean(dim=spatial_dims, skipna=True).values
    if "baa" in cube:
        df["baa"] = cube["baa"].max(dim=spatial_dims, skipna=True).values.astype(int)

    return df.sort_values("date").reset_index(drop=True)
