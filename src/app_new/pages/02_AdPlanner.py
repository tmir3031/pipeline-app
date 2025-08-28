from __future__ import annotations
import os, json, re, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from matplotlib import colormaps as mpl_cmaps
import matplotlib.pyplot as plt

from data_loader import list_drive_files, load_parquet_from_folder

# =========================
# CONFIGURATION CONSTANTS
# =========================
GEO_DIR      = "data/geo"
MAP_STYLE    = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
MAP_HEIGHT   = 720
DEFAULT_VIEW = (44.437, 26.097)     # fallback (Bucharest)
DEFAULT_K    = 12
CONTEXT_N    = 300                  # number of grey context points
HOUR_CMAP    = "cividis"            # perceptually-uniform, colour-blind friendly

# Colours for map layers
C_ORANGE = [255, 120, 20, 220]      # selected panels in Single-hour mode
C_GREY   = [120, 120, 120, 80]      # context (unselected candidates)
C_BOUND  = [0, 0, 0, 150]           # dataset boundary outline


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Ad Planner", layout="wide")
st.title("Ad Planner — rank & pick panel locations")

st.markdown(
    """
This page selects billboard/panel locations under a **transparent rule**:

**Rule:** sort candidates by the chosen score and take the **top-K**.  
Use **Single hour** for a specific hour, or **All hours (aggregate)** to plan an
always-on list by averaging scores across selected hours.

- **Map:** orange = selected panels; optional grey = other candidates for context.  
- **All hours:** each selected panel is coloured by its **peak hour** (when its score is highest).
"""
)

# ==========================================================
# DRIVE FINGERPRINT — cache invalidation when files change
# ==========================================================
ALL_DRIVE_FILES = list_drive_files()  # data_loader already caches Drive listing
FILES_SIG = hash(tuple(sorted(f.lower() for f in ALL_DRIVE_FILES)))

# Sidebar maintenance helpers
mc1, mc2 = st.sidebar.columns(2)
if mc1.button("🔄 Clear cache"):
    st.cache_data.clear()
    st.rerun()
if mc2.button("🧹 Redownload (wipe .cache)"):
    shutil.rmtree(".cache", ignore_errors=True)
    st.cache_data.clear()
    st.rerun()


# =========================
# HELPERS
# =========================
def try_import_h3():
    try:
        import h3  # type: ignore
        return h3
    except Exception:
        return None

def initial_view_points(df: pd.DataFrame) -> tuple[float, float, int]:
    """Heuristic to choose a reasonable initial view from (lat, lon) columns."""
    if len(df) == 0 or not {"lat", "lon"}.issubset(df.columns):
        return (*DEFAULT_VIEW, 10)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy()
    lat_c = float(np.nanmean(lat)); lon_c = float(np.nanmean(lon))
    span = max(np.nanmax(lat) - np.nanmin(lat), np.nanmax(lon) - np.nanmin(lon))
    if span < 0.02:   zoom = 13
    elif span < 0.05: zoom = 12
    elif span < 0.10: zoom = 11
    elif span < 0.20: zoom = 10
    elif span < 0.40: zoom = 9
    elif span < 0.80: zoom = 8
    else:             zoom = 7
    return (lat_c, lon_c, zoom)

def add_boundary(layers: list, filename="h3_frontier.geojson", rgba=C_BOUND, width=1):
    """Optional study-area boundary (GeoJSON)."""
    path = os.path.join(GEO_DIR, filename)
    if not os.path.exists(path): return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layers.append(pdk.Layer("GeoJsonLayer", data=data, stroked=True, filled=False,
                                get_line_color=rgba, line_width_min_pixels=width))
    except Exception as e:
        st.warning(f"Cannot load {filename}: {e}")

def norm01(s: pd.Series) -> pd.Series:
    """Robust 0–1 scaling (p5–p95) to map score into point sizes."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.size == 0:
        return pd.Series([], dtype=float)
    lo, hi = np.nanpercentile(s, [5, 95])
    if not np.isfinite(lo): lo = float(np.nanmin(s))
    if not np.isfinite(hi): hi = float(np.nanmax(s) + 1e-9)
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return np.clip((s - lo) / (hi - lo), 0, 1)

def hour_to_rgba(hour: int, alpha: int = 220) -> list[int]:
    """Encode hour-of-day as a colour using a perceptually-uniform colormap."""
    t = (int(hour) % 24) / 23.0
    r, g, b, _ = mpl_cmaps.get_cmap(HOUR_CMAP)(t)
    return [int(255*r), int(255*g), int(255*b), alpha]

def hour_colour_legend():
    """Small legend with ticks at 0, 6, 12, 18, 23 for All-hours colouring."""
    cmap = mpl_cmaps.get_cmap(HOUR_CMAP)
    fig, ax = plt.subplots(figsize=(5.2, 0.6), dpi=150)
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([0, 256*6/23, 256*12/23, 256*18/23, 255])
    ax.set_xticklabels(["0", "6", "12", "18", "23"], fontsize=9)
    ax.set_xlabel("Peak hour (UTC)", fontsize=9)
    ax.tick_params(axis='x', length=0)
    st.pyplot(fig, clear_figure=True)

def to_geojson_points(dfp: pd.DataFrame) -> str:
    """GeoJSON exporter (robust to NaNs and lists), using lon/lat Point geometry."""
    cols = [c for c in dfp.columns if c not in ("lat", "lon") and not c.startswith("_")]
    feats = []
    for _, r in dfp.iterrows():
        props = {}
        for k in cols:
            v = r[k]
            if pd.api.types.is_scalar(v):
                if pd.isna(v): props[k] = None
                elif isinstance(v, (np.floating, np.integer)): props[k] = float(v)
                else: props[k] = v
            elif isinstance(v, (list, tuple, np.ndarray)):
                out = []
                for x in list(v):
                    if pd.api.types.is_scalar(x):
                        if pd.isna(x): out.append(None)
                        elif isinstance(x, (np.floating, np.integer)): out.append(float(x))
                        else: out.append(x)
                    else:
                        out.append(str(x))
                props[k] = out
            else:
                props[k] = str(v)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": props
        })
    return json.dumps({"type": "FeatureCollection", "features": feats})


# =========================
# DATA LOADING (from Drive)
# =========================
@st.cache_data(show_spinner=True)
def _list_candidate_paths(files_sig: int) -> list[str]:
    """
    Return Drive paths that look like candidates per hour.
    We accept either 'candidates_5.parquet' or 'ads/candidates_5.parquet' etc.
    """
    files = list_drive_files()
    rx = re.compile(r"(?:^|/)(?:ads/)?candidates_\d+\.parquet$", re.IGNORECASE)
    paths = [p for p in files if rx.search(p)]
    if not paths:
        # fallback if naming is slightly different, but contains 'candidates'
        paths = [p for p in files if p.lower().endswith(".parquet")
                 and "candidates" in os.path.basename(p).lower()]
    return sorted(paths)

def _extract_hour_from_name(path: str) -> int | None:
    """Extract hour from filename if column 'hour' is missing."""
    m = re.search(r"candidates_(\d+)\.parquet$", os.path.basename(path))
    return int(m.group(1)) if m else None

@st.cache_data(show_spinner=True)
def load_candidates(files_sig: int) -> pd.DataFrame:
    """
    Read and concatenate all candidate Parquets from Drive.
    Normalise column names, ensure required columns, and fill hour if missing.
    """
    paths = _list_candidate_paths(files_sig)
    if not paths:
        st.error("No candidate files found in Drive (e.g., 'ads/candidates_0.parquet').")
        st.stop()

    dfs = []
    for p in paths:
        df = load_parquet_from_folder(p)  # data_loader matches by suffix or exact path
        df = df.rename(columns={c: c.lower() for c in df.columns})
        # Validate required columns
        need = {"h3_cell", "score"}
        if not need.issubset(df.columns):
            st.error(f"'{p}' is missing required columns {need}. Found: {list(df.columns)}")
            st.stop()
        # Ensure hour
        if "hour" not in df.columns or df["hour"].isna().all():
            h = _extract_hour_from_name(p)
            if h is not None:
                df["hour"] = h
            else:
                st.error(f"Cannot infer 'hour' for '{p}'. Add an 'hour' column or fix the filename pattern.")
                st.stop()
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    # Ensure lon/lat exist — compute from H3 if necessary
    if not {"lon", "lat"}.issubset(out.columns):
        out = ensure_lonlat_from_h3(out)
    return out

@st.cache_data(show_spinner=True)
def ensure_lonlat_from_h3(df: pd.DataFrame) -> pd.DataFrame:
    """If lon/lat are missing, compute centroids from H3 IDs when h3 lib is available."""
    if {"lon", "lat"}.issubset(df.columns):
        return df
    h3 = try_import_h3()
    if h3 is None:
        st.warning("Columns 'lon'/'lat' are missing and h3 is not installed; map will fallback to default view.")
        df["lon"] = np.nan; df["lat"] = np.nan
        return df

    def to_latlon(c: str):
        if hasattr(h3, "cell_to_latlng"):        # v4
            lat, lon = h3.cell_to_latlng(c)
        else:                                    # v3
            lat, lon = h3.h3_to_geo(c)
        return float(lon), float(lat)

    lonlat = df["h3_cell"].astype(str).map(lambda c: to_latlon(c) if isinstance(c, str) else (np.nan, np.nan))
    lon, lat = zip(*lonlat)
    df = df.copy()
    df["lon"] = lon
    df["lat"] = lat
    return df


# =========================
# SCORE OPTIONS
# =========================
def available_score_options(df: pd.DataFrame):
    """Derive the list of scoring options present in the data."""
    opts, colmap, desc = [], {}, {}
    if "score" in df.columns:
        opts.append("Overall"); colmap["Overall"] = "score"
        desc["Overall"] = "Composite score from the pipeline (larger = better)."
    if "score_wd" in df.columns:
        opts.append("Weekday"); colmap["Weekday"] = "score_wd"
        desc["Weekday"] = "Score computed from weekday traffic only."
    if "score_we" in df.columns:
        opts.append("Weekend"); colmap["Weekend"] = "score_we"
        desc["Weekend"] = "Score computed from weekend traffic only."
    if "score_dw" in df.columns:
        opts.append("Dwell-focused"); colmap["Dwell-focused"] = "score_dw"
        desc["Dwell-focused"] = "Prioritises locations with longer median dwell time."
    if "score_kde" in df.columns:
        opts.append("KDE proxy"); colmap["KDE proxy"] = "score_kde"
        desc["KDE proxy"] = "Kernel-density proxy of movement through the H3 cell."
    if not opts:
        st.error("No recognised score columns found (expected one of: score, score_wd, score_we, score_dw, score_kde).")
        st.stop()
    return opts, colmap, desc


# =========================
# LOAD DATA
# =========================
df_all = load_candidates(FILES_SIG)
# Make sure 'hour' is clean integer for widgets
df_all["hour"] = pd.to_numeric(df_all["hour"], errors="coerce").astype("Int64")
hours_available = sorted([int(h) for h in df_all["hour"].dropna().unique().tolist()])


# =========================
# SIDEBAR (planning controls)
# =========================
st.sidebar.header("Planning scope")
scope = st.sidebar.radio("Scope", ["Single hour", "All hours (aggregate)"], index=0, key="scope_mode")
st.sidebar.caption("**Scope** controls whether we plan for one hour or average scores across multiple hours.")

if scope == "Single hour":
    st.session_state.pop("hours_all", None)
    hour = st.sidebar.slider("Hour (UTC)", 0, 23,
                             int(hours_available[0]) if hours_available else 8,
                             key="hour_single")
    st.sidebar.caption("Hours are in **UTC**. Choose the hour you want to plan for.")
    hours_sel = [hour]
else:
    st.session_state.pop("hour_single", None)
    hours_sel = st.sidebar.multiselect("Hours to include (UTC)", list(range(24)),
                                       default=list(range(24)), key="hours_all")
    if not hours_sel: hours_sel = list(range(24))
    st.sidebar.caption("In **All hours**, the ranking uses the **mean** score across the hours you include.")

score_opts, score_colmap, score_desc = available_score_options(df_all)
score_choice = st.sidebar.selectbox("Scoring criterion", score_opts, index=0)
st.sidebar.caption(
    "**Scoring options**\n\n" +
    "\n".join([f"- **{k}** — {score_desc[k]}" for k in score_opts])
)

k = st.sidebar.slider("Number of panels (budget)", 3, 100, DEFAULT_K)
show_context  = st.sidebar.checkbox("Show unselected candidates (grey)", True)
show_boundary = st.sidebar.checkbox("Show dataset boundary", True)


# =========================
# RANK & SELECT
# =========================
df_scope = df_all[df_all["hour"].isin(hours_sel)].copy()
scol = score_colmap[score_choice]
if scol not in df_scope.columns:
    st.error(f"Score column '{scol}' is missing in the data.")
    st.stop()
df_scope["score_used"] = pd.to_numeric(df_scope[scol], errors="coerce").astype(float)

if scope == "Single hour":
    df = df_scope.copy()
else:
    # aggregation: mean across selected hours (top-K rule applied to the mean)
    df = df_scope.groupby(["h3_cell", "lat", "lon"], as_index=False)["score_used"].mean()

# Rank & select
df = df.sort_values("score_used", ascending=False)
selected = df.head(k).copy()
selected["rank"] = np.arange(1, len(selected) + 1)

# --- Tooltip details & colours ---
if scope == "All hours (aggregate)":
    keys = ["h3_cell", "lat", "lon"]
    peak = df_scope.merge(selected[keys], on=keys, how="inner")
    idx = peak.groupby(keys, sort=False)["score_used"].idxmax()
    peak_rows = peak.loc[idx, keys + ["hour", "score_used"]].rename(
        columns={"hour": "hour_peak", "score_used": "score_peak"}
    )
    selected = selected.merge(peak_rows, on=keys, how="left")
    selected["hour_label"] = selected["hour_peak"].astype("Int64")
    selected["_color"] = selected["hour_peak"].apply(
        lambda h: hour_to_rgba(int(h)) if pd.notna(h) else [120,120,120,220]
    )
    selected["score_disp"] = selected["score_used"].round(2)
    selected["score_peak_disp"] = selected["score_peak"].round(2)
else:
    selected["hour_label"] = int(hours_sel[0])
    selected["_color"] = [C_ORANGE] * len(selected)
    selected["score_disp"] = selected["score_used"].round(2)

# Context (next best after the selected ones)
if show_context:
    base_sorted = df.reset_index(drop=True)
    context = base_sorted.iloc[k:].head(CONTEXT_N).copy()
else:
    context = pd.DataFrame(columns=df.columns)


# =========================
# MAP
# =========================
layers = []
if show_boundary:
    add_boundary(layers)

if len(context):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=context.assign(size_px=6),
        get_position=["lon", "lat"],
        get_radius="size_px", radius_units="pixels",
        get_fill_color=C_GREY,
        pickable=False
    ))

# Selected points (size by mean score; colour by hour if aggregate)
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=selected.assign(size_px=(8 + 12 * norm01(selected["score_used"]))),
    get_position=["lon", "lat"],
    get_radius="size_px", radius_units="pixels",
    get_fill_color="_color",
    pickable=True
))

lat, lon, zoom = initial_view_points(df_scope)

if scope == "All hours (aggregate)":
    tip_html = (
        "<b>Rank</b>: {rank}"
        "<br/><b>Mean score</b>: {score_disp}"
        "<br/><b>Peak hour (UTC)</b>: {hour_label}"
        "<br/><b>Peak score</b>: {score_peak_disp}"
        "<br/><b>H3 cell</b>: {h3_cell}"
    )
else:
    tip_html = (
        "<b>Rank</b>: {rank}"
        "<br/><b>Score</b>: {score_disp}"
        "<br/><b>Hour (UTC)</b>: {hour_label}"
        "<br/><b>H3 cell</b>: {h3_cell}"
    )

deck = pdk.Deck(
    map_style=MAP_STYLE,
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
    layers=layers,
    tooltip={"html": tip_html, "style": {"backgroundColor": "#111", "color": "white"}}
)
try:
    st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
except TypeError:  # Streamlit API compatibility
    st.pydeck_chart(deck, use_container_width=True)

# Under-map caption
if scope == "All hours (aggregate)":
    st.caption("Colours encode the **peak hour (UTC)** for each selected location; size encodes the mean score used for ranking.")
else:
    st.caption("Orange points are the selected locations for the chosen hour; size encodes the score.")


# =========================
# KPIs & TABLE
# =========================
c1, c2 = st.columns(2)
c1.metric("Selected", f"{len(selected)}/{k}")
c2.metric("Total score", f"{selected['score_used'].sum():,.2f}")

st.subheader("Chosen panels")
base_cols = ["rank", "score_used", "h3_cell", "lat", "lon", "hour_label"]
if scope == "All hours (aggregate)" and "score_peak" in selected.columns:
    base_cols.append("score_peak")
cols_to_show = [c for c in base_cols if c in selected.columns]
st.dataframe(selected[cols_to_show], use_container_width=True)

# Column legend
st.markdown(
    """
**Columns explained**

- **rank** — position in the top-K after sorting by the selected score.  
- **score_used** — the value used for ranking (for *All hours* this is the **mean** across the included hours).  
- **hour_label** — in *Single hour*: the selected hour (UTC); in *All hours*: the **peak hour** (UTC) when this location scored highest.  
- **score_peak** — (*All hours only*) the best score for this location among the included hours (helps read daily variability).  
- **h3_cell** — H3 index of the cell where the panel sits (resolution as in the pipeline).  
- **lat / lon** — panel anchor in WGS84 (degrees).
"""
)

# =========================
# EXPORTS
# =========================
tag = "allhours" if scope == "All hours (aggregate)" else f"h{hours_sel[0]:02d}"
export_df = selected.drop(columns=[c for c in selected.columns if c.startswith("_")], errors="ignore")

st.download_button(
    "Download selection (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"adplan_{tag}_{k}panels.csv", mime="text/csv"
)
st.download_button(
    "Download selection (GeoJSON)",
    data=to_geojson_points(export_df),
    file_name=f"adplan_{tag}_{k}panels.geojson", mime="application/geo+json"
)

# Colour legend (aggregate mode)
if scope == "All hours (aggregate)":
    hour_colour_legend()

# =========================
# HELP / METHODOLOGY
# =========================
with st.expander("Methodology & reading guide"):
    st.markdown(
        """
**Scoring criterion.**  
Choose a pre-computed score from the pipeline:
*Overall* (composite), *Weekday* (weekdays only), *Weekend* (weekends only),
*Dwell-focused* (higher where median dwell is longer), or *KDE proxy* (density of movement through the hex).
All scores are on a comparable scale where **larger is better**.

**Selection rule.**  
Sort by `score_used` and pick the **top-K**.

**Tooltips.**  
- *Single hour*: shows the score for the chosen hour and the hour (UTC).  
- *All hours*: shows the **mean score** used for ranking, plus the **peak hour** (UTC) and its **peak score** for that location.

**Interpretation tips.**  
- A location with high *mean score* and a **wide separation** between *peak* and *mean* suggests strong time-of-day effects.  
- Hours are in **UTC**; convert upstream if you require local time.
        """
    )
