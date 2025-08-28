from __future__ import annotations
from pathlib import Path
import datetime as dt
import pandas as pd
import streamlit as st

# Config loader is optional
try:
    from utils import load_config
    CFG = load_config()
except Exception:
    CFG = {}

st.set_page_config(
    page_title="Home — H3 · KDE · Ad Planner",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------
left, right = st.columns([0.78, 0.22])
with left:
    st.markdown(
        """
        # 🧭 Mobility Analytics — H3, KDE & Ad Planner
        **Objective.** Demonstrate a robust mobility analysis pipeline: spatial discretisation with **H3**, hourly/day-type aggregation, comparison with **KDE**, and an applied outcome — an **Ad Planner** that ranks candidate cells for OOH placements.
        """
    )
st.markdown("---")

# ---------------------------------------------------------------------
# SCOPE & CONTRIBUTIONS
# ---------------------------------------------------------------------
c1, c2 = st.columns([0.55, 0.45])
with c1:
    st.subheader("Scope")
    st.markdown(
        """
        1. **H3 discretisation** of the urban space with hourly / day-type (weekday / weekend) aggregates.  
        2. **Methodological comparison** between discrete density (H3) and continuous density (**KDE**).  
        3. **Applied result**: identify the best **candidate H3 cells** for ad inventory planning.
        """
    )
    st.subheader("Contributions")
    st.markdown(
        """
        - Reproducible pipeline for **spatio-temporal aggregation** on an H3 grid.  
        - Interactive **exploration UI** (hour, day-type, optional KDE overlay).  
        - **Ad Planner** with a simple, interpretable score:  
          \\[
            \\textbf{score} = n\\_{users} \\times \\log(1 + \\text{dwell\\_median\\_s})
          \\]
        """
    )

#---------------------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------------------
with st.expander("Methodology — short explanation"):
    st.markdown(
        r"""
**Discretisation (H3).** Each point is mapped to an `h3_cell` at the configured resolution.  
**Temporal aggregation.** Group by `hour` and (if present) `daytype ∈ {weekday, weekend}`.

**Main indicators**
- `n_users` — number of (present) users in the cell for the selected slice;
- `dwell_median_s` — median dwell time (seconds).

**KDE.** Estimates continuous density over the same area; shown as an overlay for visual comparison.

**Ad Planner score**
\[
\text{score} = n_{users} \cdot \log\!\big(1 + \text{dwell\_median\_s}\big)
\]
The logarithm tempers extreme dwell values whilst keeping the ranking guided by both traffic and retention.
        """
    )

# ---------------------------------------------------------------------
# DATASET READINESS SNAPSHOT
# ---------------------------------------------------------------------
PARQUET_DIR = Path("data/parquet")
GEO_DIR     = Path("data/geo")
points = PARQUET_DIR / "points_with_h3.parquet"
users  = PARQUET_DIR / "h3_hour_users.parquet"
cand_dir = PARQUET_DIR / "ads"
frontier = GEO_DIR / "h3_frontier.geojson"
hours_have = [h for h in range(24) if (cand_dir / f"candidates_{h}.parquet").exists()]

def size_str(p: Path) -> str:
    try:
        b = p.stat().st_size
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if b < 1024: return f"{b:,.0f} {u}"
            b /= 1024
    except Exception:
        pass
    return "—"

def mtime_str(p: Path) -> str:
    try:
        t = dt.datetime.fromtimestamp(p.stat().st_mtime)
        return t.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"

st.subheader("Dataset readiness (quick check)")
status_rows = [
    {"Item": "points_with_h3.parquet",       "Exists": points.exists(), "Size": size_str(points), "Modified": mtime_str(points)},
    {"Item": "h3_hour_users.parquet",        "Exists": users.exists(),  "Size": size_str(users),  "Modified": mtime_str(users)},
    {"Item": "ads/candidates_{hour}.parquet","Exists": len(hours_have) > 0, "Size": f"{len(hours_have)}/24 hours", "Modified": "—"},
    {"Item": "h3_frontier.geojson",          "Exists": frontier.exists(),"Size": size_str(frontier), "Modified": mtime_str(frontier)},
]
st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Candidate hours available", f"{len(hours_have)} / 24")
c2.metric("Raw points present", "Yes" if points.exists() else "No")
c3.metric("Users file present", "Yes" if users.exists() else "No")

# ---------------------------------------------------------------------
# HOW TO USE / NAVIGATION
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("How to use this app")
st.markdown(
    """
    1) **Explore** — H3 map by `n_users`; control *Hour* and *Day type* in the sidebar.  
    2) **AdPlanner** — see the top-N candidate cells, **Top-10 table**, and **CSV export**.  
    3) **Evaluation** — check P@k / R@k / F1@k / MAP and inspect hits on the map.
    """
)

# Direct links (Streamlit ≥ 1.22). If not available, navigate via the left sidebar.
st.page_link("pages/01_Explore.py",   label="Open Explore")
st.page_link("pages/02_AdPlanner.py", label="Open AdPlanner")
st.page_link("pages/03_Evaluation.py", label="Open Evaluation")

# ---------------------------------------------------------------------
# LIMITATIONS & ETHICS
# ---------------------------------------------------------------------
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
st.caption("Tip: use the **Pages** menu on the left to open Explore / AdPlanner / Evaluation.")
