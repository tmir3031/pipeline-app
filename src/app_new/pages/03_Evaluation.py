from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from data_loader import list_drive_files, load_parquet_from_folder

# ====== CONFIG  ======
GEO_DIR      = Path("data/geo")
MAP_STYLE    = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
MAP_HEIGHT   = 650
DEFAULT_VIEW = (39.91, 116.40, 9)  # Beijing fallback (lat, lon, zoom)

C_ORANGE = [230, 120, 40, 210]   # predictions (top-k candidates)
C_BLUE   = [90, 130, 180, 120]   # relevance (R_h) from hold-out
C_GREEN  = [30, 160, 70, 230]    # intersection (hit = pred ∩ R_h)
C_BOUND  = [0, 0, 0, 220]        # dataset boundary (geojson)
# ================================================

st.set_page_config(page_title="Evaluation", layout="wide")
st.title("Evaluation — do the ranked candidates hit the true hotspots?")

# ---------- helpers ----------
def file_if_exists(p: Path) -> Path | None:
    return p if p.exists() else None

def try_import_h3():
    try:
        import h3  # type: ignore
        return h3
    except Exception:
        return None

@st.cache_data
def centroid_lonlat_for_h3(cells: list[str]) -> pd.DataFrame:
    """Return (h3_cell, lon, lat) for plotting. Uses h3 if available; otherwise NaN."""
    h3 = try_import_h3()
    rows = []
    if h3 is not None:
        # v3/v4 compatibility
        def to_latlon(c: str):
            if hasattr(h3, "cell_to_latlng"):        # v4
                lat, lon = h3.cell_to_latlng(c)
            else:                                    # v3
                lat, lon = h3.h3_to_geo(c)
            return float(lon), float(lat)
        for c in cells:
            try:
                lon, lat = to_latlon(c)
                rows.append((c, lon, lat))
            except Exception:
                rows.append((c, np.nan, np.nan))
    else:
        rows = [(c, np.nan, np.nan) for c in cells]
    return pd.DataFrame(rows, columns=["h3_cell", "lon", "lat"])

@st.cache_data
def read_candidates_per_hour(pattern: str = "candidates_{hour}.parquet") -> dict[int, pd.DataFrame]:
    """
    Citește toate fișierele candidates_*.parquet din folderul Google Drive
    (setat în st.secrets['drive']['folder_url']) și întoarce hour -> DataFrame.
    """
    out: dict[int, pd.DataFrame] = {}
    # listare o singură dată (e deja cached în data_loader)
    drive_files = set(list_drive_files())

    def _exists_on_drive(rel_or_name: str) -> bool:
        # acceptă și 'ads/candidates_5.parquet' și doar basename-ul
        return any(p.endswith(rel_or_name) or os.path.basename(p) == rel_or_name for p in drive_files)

    for h in range(24):
        rel_path = pattern.replace("{hour}", str(h))           # ex: ads/candidates_5.parquet
        base_name = os.path.basename(rel_path)                 # ex: candidates_5.parquet
        if _exists_on_drive(rel_path) or _exists_on_drive(base_name):
            df = load_parquet_from_folder(rel_path)            # data_loader face matching by suffix/basename
            if "h3_cell" in df.columns and "score" in df.columns:
                out[h] = df.sort_values("score", ascending=False).reset_index(drop=True).copy()
    return out


@st.cache_data
def build_relevance_from_points(
    points_file: str, presence_threshold_s: int, test_days_frac: float, top_p: float, seed: int
) -> tuple[dict[int, set[str]], dict[int, dict[pd.Timestamp, list[str]]]]:
    """
    Held-out relevance per hour (R_h) din points_with_h3.parquet aflat în folderul Drive.
    """
    try:
        df = load_parquet_from_folder(points_file, columns=["user_id", "h3_cell", "t"])
    except Exception:
        return {h: set() for h in range(24)}, {h: {} for h in range(24)}

    if not {"user_id", "h3_cell", "t"}.issubset(df.columns):
        return {h: set() for h in range(24)}, {h: {} for h in range(24)}

    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df["date"] = df["t"].dt.floor("D")
    df["hour"] = df["t"].dt.hour.astype("int8")

    all_days = np.array(sorted(df["date"].dropna().unique()))
    if len(all_days) == 0:
        return {h: set() for h in range(24)}, {h: {} for h in range(24)}

    rng = np.random.default_rng(seed)
    rng.shuffle(all_days)
    split = int(round(len(all_days) * (1.0 - test_days_frac)))
    test_days = set(all_days[split:])

    df = df.sort_values(["user_id", "date", "hour", "t"])
    df["t_next"] = df.groupby(["user_id", "date", "hour"])["t"].shift(-1)
    df["cell_next"] = df.groupby(["user_id", "date", "hour"])["h3_cell"].shift(-1)
    fwd = df["t_next"].notna() & df["t_next"].gt(df["t"]) & df["cell_next"].eq(df["h3_cell"])
    sub = df.loc[fwd, ["user_id", "h3_cell", "date", "hour", "t", "t_next"]].copy()
    sub["dwell_s"] = (sub["t_next"] - sub["t"]).dt.total_seconds().clip(lower=0).astype("float32")

    sub = sub[sub["date"].isin(test_days)]
    if sub.empty:
        return {h: set() for h in range(24)}, {h: {} for h in range(24)}

    per_user = (
        sub.groupby(["h3_cell", "hour", "date", "user_id"], observed=True)["dwell_s"]
        .sum().reset_index()
    )
    per_user["present"] = per_user["dwell_s"] >= float(presence_threshold_s)
    n_users_d = (
        per_user.groupby(["h3_cell", "hour", "date"], observed=True)["present"]
        .sum().reset_index(name="n_users")
    )

    # daily top-k pentru stabilitate
    daily_top: dict[int, dict[pd.Timestamp, list[str]]] = {}
    for h, g in n_users_d.groupby("hour"):
        by_day: dict[pd.Timestamp, list[str]] = {}
        for d, gd in g.groupby("date"):
            by_day[d] = gd.sort_values("n_users", ascending=False)["h3_cell"].astype(str).tolist()
        daily_top[int(h)] = by_day

    # medie pe zile → top_p% ca relevanță
    R: dict[int, set[str]] = {}
    agg = n_users_d.groupby(["h3_cell", "hour"], observed=True)["n_users"].mean().reset_index()
    for h, g in agg.groupby("hour"):
        g = g.sort_values("n_users", ascending=False).reset_index(drop=True)
        k = max(1, int(np.ceil(len(g) * top_p)))
        R[int(h)] = set(g["h3_cell"].iloc[:k].astype(str))
    return R, daily_top

def precision_recall_f1_at_k(pred: list[str], rel: set[str], k: int) -> tuple[float, float, float]:
    if k <= 0 or not pred:
        return 0.0, 0.0, 0.0
    k = min(k, len(pred))
    topk = pred[:k]
    hits = sum(1 for c in topk if c in rel)
    p = hits / float(k)
    r = (hits / float(len(rel))) if rel else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return float(p), float(r), float(f1)

def average_precision(pred: list[str], rel: set[str]) -> float:
    if not rel:
        return 0.0
    score, hits = 0.0, 0
    for i, c in enumerate(pred, start=1):
        if c in rel:
            hits += 1
            score += hits / float(i)
    return float(score / len(rel))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))

# ---------- SIDEBAR (only what’s useful for tests) ----------
st.sidebar.header("Configuration")
heldout_mode = st.sidebar.radio(
    "Relevance construction",
    ("Held-out from raw points", "From aggregate users (fallback)"),
    index=0,
    help="Prefer hold-out from raw points; the fallback uses the top-p% of aggregated n_users."
)

presence_threshold_s = st.sidebar.number_input(
    "Presence threshold (sec)", min_value=60, max_value=3600, step=30, value=300,
    help="Seconds threshold to mark a user's presence in a cell/hour (must match Step 04)."
)

test_days_frac = st.sidebar.slider(
    "Hold-out fraction of days", min_value=0.10, max_value=0.50, step=0.05, value=0.30,
    help="Proportion of days used as the test split in the hold-out."
)

top_p = st.sidebar.slider(
    "Relevance size per hour (top p%)", min_value=3, max_value=15, step=1, value=12,
    help="Size of the relevance set R_h: top p% cells per hour (by n_users in the hold-out)."
) / 100.0

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, step=1, value=13)

k_eval = st.sidebar.slider("k for P@k / R@k / F1@k", min_value=10, max_value=100, step=5, value=25)

k_stab = st.sidebar.slider("k for stability (Jaccard)", min_value=10, max_value=100, step=5, value=25)

# Hour for map inspection
st.sidebar.subheader("Visual inspection (per-hour)")
hour_to_plot = st.sidebar.selectbox("Pick an hour to inspect", list(range(24)), index=0)
k_map = st.sidebar.slider("Top-k to display on the map", min_value=10, max_value=100, step=5, value=25)

show_boundary = st.sidebar.checkbox("Show dataset boundary", True)

# ---------- BUILD relevance + read candidates ----------
cands = read_candidates_per_hour("candidates_{hour}.parquet")

if heldout_mode.startswith("Held"):
    R, daily_top = build_relevance_from_points(
        "points_with_h3.parquet",
        presence_threshold_s, test_days_frac, top_p, seed
    )
else:
    R = {}
    try:
        dfu = load_parquet_from_folder("h3_hour_users.parquet", columns=["h3_cell", "hour", "n_users"])
        for h, g in dfu.groupby("hour"):
            gm = g.groupby("h3_cell", observed=True)["n_users"].mean().reset_index()
            gm = gm.sort_values("n_users", ascending=False)
            k = max(1, int(np.ceil(len(gm) * top_p)))
            R[int(h)] = set(gm["h3_cell"].iloc[:k].astype(str))
    except Exception:
        R = {h: set() for h in range(24)}
    daily_top = {h: {} for h in range(24)}

# ---------- Compute per-hour metrics ----------
rows, APs = [], []
for h in range(24):
    pred = cands.get(h, pd.DataFrame())
    pred_list = pred["h3_cell"].astype(str).tolist() if not pred.empty else []
    rel_set = R.get(h, set())

    P, Rk, F1 = precision_recall_f1_at_k(pred_list, rel_set, k_eval)
    AP = average_precision(pred_list, rel_set)
    APs.append(AP)

    # Jaccard stability (day-to-day, top-k)
    jacc = np.nan
    dd = daily_top.get(h, {})
    if dd:
        dates = sorted(dd.keys())
        if len(dates) >= 2:
            vals = []
            for d1, d2 in zip(dates[:-1], dates[1:]):
                s1 = set(dd[d1][:k_stab])
                s2 = set(dd[d2][:k_stab])
                vals.append(jaccard(s1, s2))
            if vals:
                jacc = float(np.mean(vals))

    rows.append({
        "hour": h,
        "P@k": P,
        "R@k": Rk,
        "F1@k": F1,
        "AP": AP,
        "Jaccard_stability": (float(jacc) if not np.isnan(jacc) else None),
        "k_eval": k_eval,
        "|R_h|": len(rel_set),
        "candidates": len(pred_list),
        "max_recall_if_k": (min(k_eval, len(rel_set)) / len(rel_set) if len(rel_set) else 0.0),
    })

per_hour = pd.DataFrame(rows).sort_values("hour")
MAP = float(np.mean(APs)) if APs else 0.0

# ---------- Header KPIs ----------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("MAP", f"{MAP:.3f}")
kpi2.metric("P@k (mean)", f"{per_hour['P@k'].mean():.3f}")
kpi3.metric("R@k (mean)", f"{per_hour['R@k'].mean():.3f}")
kpi4.metric("F1@k (mean)", f"{per_hour['F1@k'].mean():.3f}")

# --- Run configuration snapshot (for reproducibility) ---
cfg_box = {
    "heldout_mode": ("from_points" if heldout_mode.startswith("Held") else "from_users_file"),
    "presence_threshold_s": int(presence_threshold_s),
    "test_days_frac": float(test_days_frac),
    "heldout_top_p": float(top_p),          # 0..1  (e.g. 0.12 = 12%)
    "random_seed": int(seed),
    "k_eval": int(k_eval),
    "stability_k": int(k_stab),
    "hour_inspected": int(hour_to_plot),
    "map_top_k": int(k_map),
    "show_boundary": bool(show_boundary),
}

with st.expander("Run configuration (snapshot) — reproducibility"):
    st.json(cfg_box)
    st.download_button(
        "Download config snapshot (JSON)",
        data=json.dumps(cfg_box, indent=2).encode("utf-8"),
        file_name="evaluation_run_config.json",
        mime="application/json"
    )
    st.code(
        "eval:\n"
        f"  heldout_mode: {cfg_box['heldout_mode']}\n"
        f"  presence_threshold_s: {cfg_box['presence_threshold_s']}\n"
        f"  test_days_frac: {cfg_box['test_days_frac']}\n"
        f"  heldout_top_p: {cfg_box['heldout_top_p']}\n"
        f"  random_seed: {cfg_box['random_seed']}\n"
        f"  k: {cfg_box['k_eval']}\n"
        f"  stability_k: {cfg_box['stability_k']}\n",
        language="yaml",
    )

st.caption("Temporal stability (held-out only): mean Jaccard of top-k across consecutive test days. Higher is better.")

# ---------- Table + profiles ----------
st.subheader("Per-hour metrics")
st.dataframe(
    per_hour[["hour","P@k","R@k","F1@k","AP","Jaccard_stability","k_eval","|R_h|","candidates","max_recall_if_k"]],
    use_container_width=True
)

st.subheader("Profiles by hour")
prof = per_hour.melt(id_vars="hour", value_vars=["P@k","R@k","F1@k","AP"], var_name="metric", value_name="value")
import altair as alt
chart = alt.Chart(prof).mark_line(point=False).encode(
    x=alt.X("hour:O", title="hour (UTC)"),
    y=alt.Y("value:Q"),
    color="metric:N"
).properties(height=220, width="container")
st.altair_chart(chart, use_container_width=True)

# ---------- Map (visual QA) ----------
st.subheader(f"Map — hour {hour_to_plot:02d} (UTC)")

pred_df = cands.get(hour_to_plot, pd.DataFrame()).copy()
if not pred_df.empty:
    pred_df = pred_df.head(k_map).copy()
else:
    pred_df = pd.DataFrame(columns=["h3_cell","score","lon","lat"])

# relevance set (as centroids)
rel_cells = list(R.get(hour_to_plot, set()))
rel_geo = centroid_lonlat_for_h3(rel_cells)

# ensure lon/lat in predictions (prefer those from step 07)
if "lon" not in pred_df.columns or "lat" not in pred_df.columns:
    # fallback: compute centroids
    cen = centroid_lonlat_for_h3(pred_df["h3_cell"].astype(str).tolist())
    pred_df = pred_df.merge(cen, on="h3_cell", how="left")

# intersection (true hits)
rel_set_plot = set(rel_cells)
pred_df["_in_rel"] = pred_df["h3_cell"].astype(str).isin(rel_set_plot)
hits = pred_df[pred_df["_in_rel"]].copy()

layers = []

# boundary (optional)
if show_boundary:
    p_frontier = file_if_exists(GEO_DIR / "h3_frontier.geojson")
    if p_frontier:
        with open(p_frontier, "r", encoding="utf-8") as f:
            data = json.load(f)
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=data, stroked=True, filled=False,
            get_line_color=C_BOUND, line_width_min_pixels=2
        ))

# relevance (blue)
if len(rel_geo):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=rel_geo.dropna(),
        get_position='[lon, lat]',
        get_radius=90,
        get_fill_color=C_BLUE,
        pickable=False,
    ))

# predictions (orange)
if len(pred_df):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=pred_df.dropna(subset=["lon","lat"]),
        get_position='[lon, lat]',
        get_radius=120,
        get_fill_color=C_ORANGE,
        pickable=True,
        tooltip=True,
    ))

# hits (green) — drawn above predictions
if len(hits):
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=hits.dropna(subset=["lon","lat"]),
        get_position='[lon, lat]',
        get_radius=130,
        get_fill_color=C_GREEN,
        pickable=True,
        tooltip=True,
    ))

# view: centre on predictions if available; otherwise fallback
if len(pred_df) and pred_df["lon"].notna().any():
    lat = float(pred_df["lat"].mean())
    lon = float(pred_df["lon"].mean())
    zoom = 11
else:
    lat, lon, zoom = DEFAULT_VIEW

tooltip = {
    "html": "<b>h3:</b> {h3_cell}<br/><b>score:</b> {score}",
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

st.caption("Colours: **orange** = top-k predictions (candidates), **blue** = held-out relevance cells, "
           "**green** = intersections (hit = prediction ∩ relevance).")

# ---------- Downloads ----------
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download per-hour CSV",
        data=per_hour.to_csv(index=False).encode("utf-8"),
        file_name="08_eval_per_hour.csv",
        mime="text/csv"
    )
with c2:
    summary = pd.DataFrame([{
        "hours_evaluated": int(per_hour["hour"].nunique()),
        "P@k_mean": float(per_hour["P@k"].mean()),
        "R@k_mean": float(per_hour["R@k"].mean()),
        "F1@k_mean": float(per_hour["F1@k"].mean()),
        "MAP": MAP,
        "mean_Jaccard_stability": float(per_hour["Jaccard_stability"].dropna().mean()) if "Jaccard_stability" in per_hour else np.nan,
        "heldout_mode": ("from_points" if heldout_mode.startswith("Held") else "from_users_file"),
        "heldout_top_p": top_p,
        "test_days_frac": test_days_frac,
        "presence_threshold_s": presence_threshold_s,
        "k_eval": k_eval,
        "stability_k": k_stab,
    }])
    st.download_button(
        "Download summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="08_eval_summary.csv",
        mime="text/csv"
    )

# ---------- Footnotes ----------
with st.expander("How to read these metrics (short)"):
    st.markdown("""
- **P@k** — of the first *k* recommendations, what fraction are genuine hotspots?  
- **R@k** — of all hotspots in the hold-out (Rₕ), what fraction are captured in the top-k?  
- **F1@k** — harmonic mean of P@k and R@k (balance between precision and coverage).  
- **AP / MAP** — *Average Precision* per hour and the mean across hours (ranking within the list matters).  
- **Jaccard stability** — how similar the top-k sets are across consecutive test days (temporal consistency).  
- **max_recall_if_k** — theoretical upper bound on recall when *k < |Rₕ|*.
    """)
