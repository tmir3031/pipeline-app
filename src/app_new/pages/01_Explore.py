from __future__ import annotations
import os, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from matplotlib import colormaps as mpl_cmaps

from data_loader import load_parquet_from_folder, list_drive_files

# ============== CONFIG ==============
GEO_DIR = "data/geo"
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
DEFAULT_VIEW = (44.437, 26.097)     # fallback if nothing else is found (Bucharest)
MAP_HEIGHT = 720

CMAP_NAME = "cividis"                # perceptually uniform, colour-blind friendly
SCALING_MODE = "robust"              # 'robust' (p5–p95) or 'minmax'
INCLUDE_ZERO = False                 # hide zero-value cells by default
# ====================================

# ---------- PAGE ----------
st.set_page_config(page_title="Explore", layout="wide")
st.title("Explore — mobility on H3")

# ---------- DRIVE FINGERPRINT & MAINTENANCE ----------
ALL_DRIVE_FILES = list_drive_files()  # cached inside data_loader
FILES_SIG = hash(tuple(sorted(f.lower() for f in ALL_DRIVE_FILES)))

m1, m2 = st.sidebar.columns(2)
if m1.button("🔄 Clear cache"):
    st.cache_data.clear()
    st.rerun()
if m2.button("🧹 Redownload (.cache)"):
    shutil.rmtree(".cache", ignore_errors=True)
    st.cache_data.clear()
    st.rerun()

# For presence checks by *basename* (robust to subfolders inside Drive)
_DRIVE_BASENAMES = set(os.path.basename(p) for p in ALL_DRIVE_FILES)
def has_drive(basename: str) -> bool:
    return basename in _DRIVE_BASENAMES

def file_if_exists(path: str) -> str | None:
    return path if os.path.exists(path) else None

# ---------- HELPERS ----------
@st.cache_data(show_spinner=True)
def initial_view_from_boundary() -> tuple[float, float] | None:
    """Approximate centre from dataset boundary; None if not available."""
    path = os.path.join(GEO_DIR, "h3_frontier.geojson")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        coords = []
        for feat in data.get("features", []):
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]; break
            if geom.get("type") == "MultiPolygon":
                coords = geom.get("coordinates", [[[]]])[0][0]; break
        if coords:
            lon = float(np.mean([c[0] for c in coords])); lat = float(np.mean([c[1] for c in coords]))
            return (lat, lon)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=True)
def load_df_from_drive(files_sig: int, filename: str, value_col: str) -> pd.DataFrame:
    """
    Load a Parquet from Drive and normalise columns to: h3_cell, hour, value, [daytype].

    files_sig is only used to invalidate cache when Drive listing changes.
    """
    _ = files_sig                       # participate in cache key
    df = load_parquet_from_folder(filename)  # suffix or exact path allowed
    # normalise common columns
    ren = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "h3_cell": ren[c] = "h3_cell"
        if lc == "hour":    ren[c] = "hour"
        if lc == "daytype": ren[c] = "daytype"
    df = df.rename(columns=ren)

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in {filename}. Columns: {list(df.columns)}")

    out = df[["h3_cell", "hour", value_col]].copy().rename(columns={value_col: "value"})
    # force types
    out["h3_cell"] = out["h3_cell"].astype(str)
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if "daytype" in df.columns:
        out["daytype"] = df["daytype"].astype(str)
    return out

@st.cache_data(show_spinner=True)
def load_users_daytype_from_daily_drive(files_sig: int) -> pd.DataFrame | None:
    """
    Build weekday/weekend means from a daily file in Drive.
    Expects columns: h3_cell, hour, n_users, date_utc
    """
    _ = files_sig
    if not has_drive("h3_day_hour_users.parquet"):
        return None
    df = load_parquet_from_folder("h3_day_hour_users.parquet")
    df = df.rename(columns={"h3_cell": "h3_cell", "hour": "hour", "n_users": "n_users"})
    # types
    df["h3_cell"] = df["h3_cell"].astype(str)
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    df["n_users"] = pd.to_numeric(df["n_users"], errors="coerce")
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")

    dow = df["date_utc"].dt.dayofweek
    df["daytype"] = np.where(dow >= 5, "weekend", "weekday")
    g = df.groupby(["h3_cell", "hour", "daytype"], as_index=False, observed=True)["n_users"].mean()
    g = g.rename(columns={"n_users": "value"})
    return g

# Auto-centre on H3 (supports h3 v3 & v4)
def view_from_h3_cells(cells: list[str], sample: int = 6000):
    """Return (lat, lon, zoom) computed from H3 cells."""
    try:
        import h3
    except Exception:
        return None
    if not cells:
        return None
    if len(cells) > sample:
        rng = np.random.default_rng(0)
        cells = rng.choice(cells, size=sample, replace=False).tolist()

    if hasattr(h3, "h3_to_geo"):          # v3
        to_latlng = h3.h3_to_geo
    elif hasattr(h3, "cell_to_latlng"):   # v4
        to_latlng = h3.cell_to_latlng
    else:
        return None

    latlons = np.array([to_latlng(c) for c in cells], dtype="float64")  # (lat, lon)
    if latlons.size == 0:
        return None

    lat, lon = float(latlons[:, 0].mean()), float(latlons[:, 1].mean())
    lat_min, lat_max = float(latlons[:, 0].min()), float(latlons[:, 0].max())
    lon_min, lon_max = float(latlons[:, 1].min()), float(latlons[:, 1].max())
    span = max(lat_max - lat_min, lon_max - lon_min)

    if span < 0.02:   zoom = 13
    elif span < 0.05: zoom = 12
    elif span < 0.10: zoom = 11
    elif span < 0.20: zoom = 10
    elif span < 0.40: zoom = 9
    elif span < 0.80: zoom = 8
    else:             zoom = 7
    return (lat, lon, zoom)

# Perceptually-uniform colouring + colour bar
def colour_with_cmap(df: pd.DataFrame, value_col="value",
                     include_zero: bool = False, cmap_name: str = "cividis",
                     scaling: str = "robust"):
    """
    Map 'value' -> RGBA using a perceptually-uniform colormap.
    - include_zero=False: zero cells are hidden (alpha=0) & excluded from scaling.
    - scaling: 'robust' (p5–p95) or 'minmax' (min–max).
    Returns: df with _colour, and (vmin, vmax) used for scaling.
    """
    if len(df) == 0:
        df["_colour"] = []
        return df, (np.nan, np.nan)

    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(v) & ((v > 0) | include_zero)
    vref = v[valid]

    if vref.size == 0:
        df["_colour"] = [[200, 200, 200, 0]] * len(df)
        return df, (np.nan, np.nan)

    if scaling == "robust":
        vmin, vmax = np.nanpercentile(vref, [5, 95])
    else:
        vmin, vmax = float(np.nanmin(vref)), float(np.nanmax(vref))
    if not np.isfinite(vmin): vmin = float(np.nanmin(vref))
    if not np.isfinite(vmax): vmax = float(np.nanmax(vref))
    if vmax <= vmin: vmax = vmin + 1e-9

    norm = np.zeros_like(v, dtype="float64")
    norm[valid] = np.clip((v[valid] - vmin) / (vmax - vmin), 0, 1)

    cmap = mpl_cmaps.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)  # (N,4)
    if not include_zero:
        rgba[(v <= 0) | ~np.isfinite(v), 3] = 0
    else:
        rgba[(v <= 0) | ~np.isfinite(v), 3] = 60

    df = df.copy()
    df["_colour"] = rgba.tolist()
    return df, (float(vmin), float(vmax))

def render_colour_bar(cmap_name: str, vmin: float, vmax: float,
                      width_px: int = 420, height_px: int = 24):
    """Draw a horizontal colour bar (no deprecated args)."""
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return
    colours = (mpl_cmaps.get_cmap(cmap_name)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    bar = np.tile(colours[None, :, :], (height_px, 1, 1))
    st.image(bar, caption=f"{cmap_name} — {vmin:.2f} → {vmax:.2f}", width=width_px, use_container_width=False)

def add_geojson_layer(layers: list, filename: str, stroke=True, fill=False, line_rgba=[0,0,0,200], width=2):
    path = file_if_exists(os.path.join(GEO_DIR, filename))
    if not path: return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layers.append(pdk.Layer("GeoJsonLayer", data=data, stroked=stroke, filled=fill,
                                get_line_color=line_rgba, line_width_min_pixels=width))
    except Exception as e:
        st.warning(f"Cannot load {filename}: {e}")

# ---------- DATA SOURCES ----------
metrics = []
if has_drive("h3_hour_users.parquet"):
    metrics.append(("Users (all days)", "h3_hour_users.parquet", "n_users", False))
if has_drive("h3_day_hour_users.parquet"):
    metrics.append(("Users (weekday/weekend)", "h3_day_hour_users.parquet", "n_users", True))
if has_drive("h3_hour_person_minutes.parquet"):
    metrics.append(("Person-minutes", "h3_hour_person_minutes.parquet", "person_minutes", False))
if has_drive("h3_hour_dwell_median.parquet"):
    metrics.append(("Median dwell (sec)", "h3_hour_dwell_median.parquet", "dwell_median_s", False))
if has_drive("kde_in_h3.parquet"):
    metrics.append(("KDE in cell (traffic proxy)", "kde_in_h3.parquet", "kde_in_cell", False))

if not metrics:
    st.error("No Parquet files found in Drive folder.")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Filters")
metric_label = st.sidebar.selectbox("Metric", [m[0] for m in metrics], index=0)
_metric_def = {
    "Users (all days)":
        "Number of distinct users **present** in the H3 cell during the selected hour, "
        "aggregated over the whole dataset period (presence determined upstream by a dwell-time threshold). "
        "A proxy for footfall/attendance.",

    "Users (weekday/weekend)":
        "Same as *Users (all days)*, but averaged **separately** over days by day type. "
        "Use the day-type toggle that appears (weekday / weekend).",

    "Person-minutes":
        "Total dwell time accumulated by **all users** in the cell during the hour, measured in **person-minutes** "
        "(e.g., 10 people × 3 minutes = 30 person-minutes). Captures both volume and how long people stay.",

    "Median dwell (sec)":
        "Median (50th percentile) **per-user dwell time** in seconds for that cell and hour. "
        "More robust than the mean; higher values imply longer typical stays.",

    "KDE in cell (traffic proxy)":
        "**Unitless intensity** from a kernel density estimate of raw points, projected onto the H3 grid. "
        "Useful as a proxy for movement/throughput; comparable **within the current selection**."
}
st.sidebar.caption("What this metric means")
st.sidebar.info(_metric_def.get(metric_label, ""))

metric_file, metric_value_col, metric_has_daytype = [
    (m[1], m[2], m[3]) for m in metrics if m[0] == metric_label
][0]

hour = st.sidebar.slider("Hour (UTC)", 0, 23, 0)
show_frontier = st.sidebar.checkbox("Boundary (GeoJSON)", True)
show_grid     = st.sidebar.checkbox("H3 grid (if available)", False)
show_hotspots = st.sidebar.checkbox("Highlight hotspots (≥ P95)", False)

# ---------- LOAD + FILTER ----------
if metric_file == "h3_day_hour_users.parquet":
    df = load_users_daytype_from_daily_drive(FILES_SIG)
else:
    df = load_df_from_drive(FILES_SIG, metric_file, metric_value_col)

# optional robustness: clip negatives for non-negative metrics
if metric_value_col in ("n_users", "person_minutes", "dwell_median_s", "kde_in_cell") \
   or metric_file == "h3_day_hour_users.parquet":
    df["value"] = pd.to_numeric(df["value"], errors="coerce").clip(lower=0)

# filter by hour and optional day type
df = df[df["hour"] == hour].copy()

if "daytype" in df.columns:
    day_opts = sorted(df["daytype"].dropna().unique().tolist())
    daytype = st.sidebar.radio("Day type", day_opts, index=0, horizontal=True)
    df = df[df["daytype"] == daytype].copy()

# colouring
df, (vmin, vmax) = colour_with_cmap(
    df, value_col="value",
    include_zero=INCLUDE_ZERO,
    cmap_name=CMAP_NAME,
    scaling=SCALING_MODE,
)
if not INCLUDE_ZERO:
    df = df[df["value"] > 0]

if df.empty:
    st.warning("No activity for the current selection.")
    st.stop()

# ---------- LAYERS ----------
layers = [
    pdk.Layer(
        "H3HexagonLayer",
        data=df,
        get_hexagon="h3_cell",
        get_fill_color="_colour",
        get_line_color=[90, 90, 90, 120],
        line_width_min_pixels=0.5,
        stroked=True,
        pickable=True,
        extruded=False,
    )
]

# Outline for hotspots (top ~5% by value)
if show_hotspots:
    p95_thresh = float(np.nanpercentile(df["value"], 95))
    hot = df[df["value"] >= p95_thresh]
    if len(hot):
        layers.append(pdk.Layer(
            "H3HexagonLayer",
            data=hot,
            get_hexagon="h3_cell",
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[255, 255, 255, 220],
            line_width_min_pixels=1.5,
            stroked=True,
            extruded=False,
            pickable=False
        ))

if show_frontier:
    add_geojson_layer(layers, "h3_frontier.geojson", stroke=True, fill=False, line_rgba=[0,0,0,220], width=2)
if show_grid:
    add_geojson_layer(layers, "h3_polygons.geojson", stroke=True, fill=False, line_rgba=[120,120,120,100], width=1)

# ---------- MAP (auto-centre on current data) ----------
auto_view = view_from_h3_cells(df["h3_cell"].astype(str).unique().tolist())
if auto_view:
    lat, lon, zoom = auto_view
else:
    centre = initial_view_from_boundary()
    if centre:
        lat, lon = centre; zoom = 10
    else:
        lat, lon = DEFAULT_VIEW; zoom = 10

# Units for tooltips and KPIs
units_map = {
    "n_users": "users",
    "person_minutes": "person-minutes",
    "dwell_median_s": "seconds",
    "kde_in_cell": "index"
}
unit = "users (daily mean)" if metric_file == "h3_day_hour_users.parquet" \
       else units_map.get(metric_value_col, "value")

extra_day = "<br/><b>daytype:</b> {daytype}" if "daytype" in df.columns else ""
tooltip = {
    "html": (
        f"<b>h3:</b> {{h3_cell}}"
        f"<br/><b>value ({unit}):</b> {{value}}"
        f"<br/><b>hour (UTC):</b> {{hour}}"
        + extra_day
    ),
    "style": {"backgroundColor": "#111", "color": "white"},
}
deck = pdk.Deck(
    map_style=MAP_STYLE,
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
    layers=layers,
    tooltip=tooltip
)
try:
    st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
except TypeError:
    st.pydeck_chart(deck, use_container_width=True)

# ---------- COLOUR BAR / KPIs ----------
st.caption("Perceptually-uniform palette and robust scaling (p5–p95) for sound interpretation (colour-blind friendly).")

# Summary stats
median_v = float(np.nanmedian(df["value"]))
p95_v    = float(np.nanpercentile(df["value"], 95))
skew     = (p95_v / median_v) if median_v > 0 else np.nan

def skew_label(x: float) -> str:
    if not np.isfinite(x): return "—"
    if x < 1.5: return "low concentration"
    if x < 3:   return "moderate concentration"
    if x < 6:   return "high concentration"
    return "very high concentration"

cc1, cc2 = st.columns([3, 2])
with cc1:
    render_colour_bar(CMAP_NAME, vmin, vmax)
with cc2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cells shown", f"{len(df):,}")
    c2.metric(f"Median ({unit})", f"{median_v:,.0f}")
    c3.metric(f"P95 ({unit})", f"{p95_v:,.0f}")
    c4.metric("Skew (P95/Median)", f"{skew:,.2f}" if np.isfinite(skew) else "—")

st.caption(
    f"Concentration level: **{skew_label(skew)}** (Skew = P95/Median). "
    "Values ≳ 3 usually indicate strong concentration of activity."
)

# ---------- TOP CELLS ----------
st.subheader("Top 15 cells (current filter)")
cols = ["h3_cell","value","hour"] + (["daytype"] if "daytype" in df.columns else [])
st.dataframe(df.sort_values("value", ascending=False).head(15)[cols], use_container_width=True)

# ---------- COLUMN EXPLANATIONS ----------
value_expl = {
    "n_users": (
        "Number of distinct users **present** in the cell during the hour. "
        "Presence is determined upstream by a dwell-time threshold (e.g., ≥5 minutes)."
    ),
    "person_minutes": (
        "Total dwell time accumulated by all users in the cell during the hour, measured in **person-minutes**. "
        "Example: 10 people × 3 minutes each ⇒ 30 person-minutes."
    ),
    "dwell_median_s": (
        "Per-cell **median** dwell time per user during the hour, in **seconds** "
        "(less sensitive to outliers than the mean)."
    ),
    "kde_in_cell": (
        "Kernel Density Estimate (KDE) intensity projected onto the H3 cell — a proxy for flow/traffic through that area. "
        "Higher values indicate stronger concentration of trajectories."
    ),
}
if metric_file == "h3_day_hour_users.parquet":
    value_text = ("Mean **n_users** across days for the chosen day type "
                  "(*weekday*/*weekend*) and hour (UTC). Presence uses the upstream threshold.")
else:
    value_text = value_expl.get(metric_value_col, f"Value from column **{metric_value_col}** used for colouring and ranking.")

st.markdown(
    f"""
**Columns explained**

- **h3_cell** — H3 cell ID (string) at the dataset’s resolution; unique identifier for the hexagon.  
- **value** — {value_text}  
- **hour** — hour bin in **UTC** to which the value refers.  
{"- **daytype** — present only when derived from daily users; either **weekday** or **weekend**." if "daytype" in df.columns else ""}
"""
)

# ---------- EXPORT ----------
st.download_button(
    "Download selection (CSV)",
    data=df[["h3_cell","hour","value"] + (["daytype"] if "daytype" in df.columns else [])].to_csv(index=False).encode("utf-8"),
    file_name=f"explore_{metric_label.replace(' ','_')}_h{hour:02d}.csv",
    mime="text/csv"
)

# ---------- METHODOLOGY ----------
with st.expander("Methodology & interpretation"):
    st.markdown(
        """
**H3 grid.** The study area is partitioned into fixed-resolution hexagonal cells; values are aggregated per hour and cell.

**Metrics:**
- **n_users** — number of users present in the cell during the hour (presence threshold applied upstream).
- **person_minutes** — total dwell time of all users in minutes.
- **dwell_median_s** — median per-user dwell time in seconds.
- **kde_in_cell** — kernel density estimate projected onto the H3 grid.

**Notes:**
- Hours are in **UTC**. If you need local time, convert upstream before aggregation.
- Avoid comparing absolute colours across different selections — the scale is recomputed each time. Prefer comparing **shapes** and **ranks** within a view.
        """
    )
