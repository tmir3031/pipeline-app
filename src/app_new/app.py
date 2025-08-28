from __future__ import annotations
from pathlib import Path
import re
import datetime as dt
import os
import pandas as pd
import streamlit as st

try:
    from utils import load_config
    CFG = load_config()
except Exception:
    CFG = {}

# --- Page config ----------------------------------------------------------------
st.set_page_config(
    page_title="Home — Mobility on H3",
    page_icon="🧭",
    layout="wide",
)

# --- Small helpers ---------------------------------------------------------------
def _extract_folder_id(url_or_id: str) -> str:
    """
    Extract a Google Drive folder ID from a full URL; if the input is already an ID,
    return it unchanged. Works for links like:
    https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing
    """
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
    return m.group(1) if m else url_or_id

def _drive_cache_dir() -> Path | None:
    """
    Resolve the local cache directory used by data_loader/gdown:
    .cache/drive/<FOLDER_ID>
    Returns None if secrets are not configured.
    """
    try:
        folder_url = st.secrets["drive"]["folder_url"]
        folder_id = _extract_folder_id(folder_url)
        return Path(".cache") / "drive" / folder_id
    except Exception:
        return None

def _fmt_size(bytes_path: Path) -> str:
    """Human-readable file size or '—' if missing."""
    try:
        b = bytes_path.stat().st_size
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024:
                return f"{b:,.0f} {u}"
            b /= 1024
    except Exception:
        pass
    return "—"

def _fmt_mtime(p: Path) -> str:
    """YYYY-MM-DD HH:MM or '—' if missing."""
    try:
        t = dt.datetime.fromtimestamp(p.stat().st_mtime)
        return t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"

def _drive_has(drive_rel_or_name: str, all_drive_paths: list[str]) -> bool:
    """
    True if any file in Drive matches by suffix or basename.
    Accepts 'ads/candidates_5.parquet' or just 'candidates_5.parquet'.
    """
    base = os.path.basename(drive_rel_or_name)
    return any(p.endswith(drive_rel_or_name) or os.path.basename(p) == base for p in all_drive_paths)

def _local_path_in_cache(drive_rel_path: str) -> Path | None:
    """
    Convert a Drive-relative path (as returned by list_drive_files) into a local path
    under the gdown cache. Returns None if cache base is unknown.
    """
    base = _drive_cache_dir()
    return (base / drive_rel_path) if base else None

# --- HERO -----------------------------------------------------------------------
left, right = st.columns([0.78, 0.22])
with left:
    st.markdown(
        """
# 🧭 Mobility Analytics — H3, KDE & Ad Planner

**Objective.** Demonstrate a robust mobility analysis pipeline: spatial discretization with **H3**, hourly/day-type aggregation, comparison with **KDE**, and an applied outcome — an **Ad Planner** that ranks candidate cells for OOH placements.
        """
    )
st.markdown("---")

# --- Force re-download data (clear Drive cache + Streamlit cache) ---------------
with st.sidebar:
    st.subheader("Data control")
    if st.button("🔁 Force redownload data"):
        cache_dir = _drive_cache_dir()
        if cache_dir and cache_dir.exists():
            # Remove the entire local mirror of the Drive folder
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        # Clear Streamlit's @st.cache_data across the app
        st.cache_data.clear()
        st.success("Cache cleared. Redownloading will occur on next data access.")
        st.rerun()

# --- Readiness snapshot (from Google Drive) -------------------------------------
st.subheader("Dataset readiness (from Google Drive)")

drive_ok = True
drive_paths: list[str] = []
try:
    # This import will trigger a one-time mirrored download on first call.
    from data_loader import list_drive_files
    drive_paths = list_drive_files()  # relative paths within the cached Drive folder
except Exception as e:
    drive_ok = False
    st.warning(
        "Could not list files from Google Drive. Check that **Secrets** contain "
        "`[drive] folder_url` and the folder is shared as *Anyone with the link — Viewer*."
    )

# Expected files (minimum viable set for all three pages)
expected = {
    # Explore
    "h3_hour_users.parquet": "Explore — users per hour",
    "h3_day_hour_users.parquet": "Explore — weekday/weekend (optional)",
    "h3_hour_person_minutes.parquet": "Explore — person-minutes (optional)",
    "h3_hour_dwell_median.parquet": "Explore — dwell median (optional)",
    "kde_in_h3.parquet": "Explore — KDE proxy (optional)",
    # Evaluation (raw points, optional if using fallback)
    "points_with_h3.parquet": "Evaluation — held-out from raw points (optional)",
}
# Candidates are named per hour; we count how many exist
candidate_names = [f"ads/candidates_{h}.parquet" for h in range(24)]

rows = []
if drive_ok:
    # Report for core files
    for fname, desc in expected.items():
        exists = _drive_has(fname, drive_paths)
        lp = None
        # If we can resolve a specific path, show size/mtime from the local cache
        # (find the first matching path)
        for p in drive_paths:
            if p.endswith(fname) or os.path.basename(p) == os.path.basename(fname):
                lp = _local_path_in_cache(p)
                break
        rows.append({
            "Item": fname,
            "Role": desc,
            "Exists": "Yes" if exists else "No",
            "Size": _fmt_size(lp) if (lp and lp.exists()) else "—",
            "Modified": _fmt_mtime(lp) if (lp and lp.exists()) else "—",
        })

    # Candidates summary row
    cand_count = sum(1 for c in candidate_names if _drive_has(c, drive_paths) or _drive_has(os.path.basename(c), drive_paths))
    rows.append({
        "Item": "ads/candidates_{hour}.parquet",
        "Role": "Ad Planner & Evaluation — per-hour candidates",
        "Exists": f"{cand_count}/24",
        "Size": "—",
        "Modified": "—",
    })

# Optional local geo boundary (kept in repo, not on Drive)
geo_frontier = Path("data/geo/h3_frontier.geojson")
rows.append({
    "Item": "data/geo/h3_frontier.geojson",
    "Role": "Optional map boundary overlay",
    "Exists": "Yes" if geo_frontier.exists() else "No",
    "Size": _fmt_size(geo_frontier) if geo_frontier.exists() else "—",
    "Modified": _fmt_mtime(geo_frontier) if geo_frontier.exists() else "—",
})

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# KPIs
c1, c2, c3 = st.columns(3)
cand_count = next((r for r in rows if r["Item"] == "ads/candidates_{hour}.parquet"), None)
c1.metric("Candidate hours available", cand_count["Exists"] if cand_count else "—")
c2.metric("Drive reachable", "Yes" if drive_ok else "No")
c3.metric("Boundary file present", "Yes" if geo_frontier.exists() else "No")

# --- Scope & contributions -------------------------------------------------------
left, right = st.columns([0.55, 0.45])
with left:
    st.subheader("Scope")
    st.markdown(
        """
1. **H3 discretization** of urban space with hourly / day-type (weekday / weekend) aggregates.  
2. **Methodological comparison** between discrete density (H3) and continuous density (**KDE**).  
3. **Applied result**: identify the best **candidate H3 cells** for ad inventory planning.
        """
    )
    st.subheader("Contributions")
    st.markdown(
        r"""
- Reproducible pipeline for **spatio-temporal aggregation** on an H3 grid.  
- Interactive **exploration UI** (hour, day-type, optional KDE overlay).  
- **Ad Planner** with a simple, interpretable score:  
        """
    )

# --- Methodology (short) ---------------------------------------------------------
with st.expander("Methodology — short explanation"):
    st.markdown(
        r"""
**Discretization (H3).** Each point is mapped to an `h3_cell` at the configured resolution.  
**Temporal aggregation.** Group by `hour` and (if present) `daytype ∈ {weekday, weekend}`.

**Main indicators**
- `n_users` — number of present users in the cell for the selected slice;
- `dwell_median_s` — median dwell time (seconds).

**KDE.** Estimates continuous density over the same area; shown as an overlay for visual comparison.
        """
    )

# --- Navigation & how to use -----------------------------------------------------
st.markdown("---")
st.subheader("How to use this app")
st.markdown(
    """
1) **Explore** — H3 choropleth by metric; control *Hour* and *Day type* in the sidebar.  
2) **Ad Planner** — rank candidates by the chosen score; select **Top-K** and export CSV/GeoJSON.  
3) **Evaluation** — compute P@k / R@k / F1@k / AP (MAP across hours), plus stability; inspect hits on the QA map.
    """
)

# If your filenames differ, adjust these paths accordingly.
st.page_link("pages/01_Explore.py",    label="Open Explore")
st.page_link("pages/02_AdPlanner.py",  label="Open Ad Planner")
st.page_link("pages/03_Evaluation.py", label="Open Evaluation")

# --- Limitations & ethics --------------------------------------------------------
with st.expander("Limitations & ethical notes"):
    st.markdown(
        """
- Potential **coverage bias** in mobility data (space/time).  
- H3 implies a **resolution–stability** trade-off.  
- KDE is sensitive to **bandwidth** (over/under-smoothing).  
- Ad Planner results are **indicative**, not prescriptive.
        """
    )

st.markdown("---")
st.caption("Tip: use the **Pages** menu on the left to open Explore / Ad Planner / Evaluation.")
