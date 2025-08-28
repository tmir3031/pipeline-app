# Mobility on H3 — Streamlit App

> Visual analytics & planning tools over H3 hex grids (Explore, Ad Planner, Evaluation), with data loaded on‑the‑fly from a public Google Drive folder.

> **Note.** This repository is **frontend‑only**. Parquet datasets are **not** tracked in Git; they are fetched at runtime from Google Drive (public link) and cached locally on first use.

---

## Table of contents
- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Data requirements & Drive sharing](#data-requirements--drive-sharing)
- [Streamlit Secrets (Drive URL & cache)](#streamlit-secrets-drive-url--cache)
- [Local development](#local-development)
- [Deploy on Streamlit Cloud](#deploy-on-streamlit-cloud)
- [How Drive loading & caching works](#how-drive-loading--caching-works)
---

## Overview

The project provides three complementary views over mobility metrics aggregated on an **H3 hexagonal grid**:

1. **Explore** — choropleth maps per hour using perceptually‑uniform colour maps and robust scaling (p5–p95). Supports metrics like `n_users`, `person_minutes`, `dwell_median_s`, and KDE proxies.
2. **Ad Planner** — rank‑and‑pick billboard/panel locations using precomputed **scores**; select **Top‑K** and export CSV/GeoJSON. In multi‑hour mode, colours encode the **peak hour**.
3. **Evaluation** — model‑agnostic evaluation with a **held‑out relevance set** (built from raw points or aggregated users): P@k, R@k, F1@k, AP/MAP, and temporal stability (Jaccard). Includes a QA map by hour.

All pages load **Parquet** files from a Google Drive **folder** configured via Streamlit Secrets, downloaded once with `gdown`, then served from a local cache.

---

## Repository layout

```
.
├─ Home.py
├─ pages/
│  ├─ 01_Explore.py
│  ├─ 02_Ad_Planner.py
│  └─ 03_Evaluation.py
├─ data_loader.py
├─ data/
│  └─ geo/                # optional: h3_frontier.geojson, h3_polygons.geojson
├─ .streamlit/
│  └─ secrets.toml        # local only; on Cloud use UI Secrets
├─ requirements.txt
└─ README.md
```

---

## Data requirements & Drive sharing

### Sharing
- The Drive folder must be shared as **“Anyone with the link — Viewer.”**
- URL format: `https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing`

### Expected files (minimum)
- **Explore**
  - `h3_hour_users.parquet`
  - `h3_day_hour_users.parquet` (optional, enables weekday/weekend split)
  - `h3_hour_person_minutes.parquet` (optional)
  - `h3_hour_dwell_median.parquet` (optional)
  - `kde_in_h3.parquet` (optional)
- **Ad Planner**
  - `candidates_<hour>.parquet` (e.g., `candidates_0.parquet` … `candidates_23.parquet`)    Columns: `h3_cell`, `lat`, `lon`, `hour`, `score` (+ optionally `score_wd`, `score_we`, `score_dw`, `score_kde`)
- **Evaluation**
  - `candidates_<hour>.parquet` (or under `ads/`) — columns: `h3_cell`, `score`, (`lat`,`lon` optional)
  - `points_with_h3.parquet` for held‑out from raw points (`user_id`, `h3_cell`, `t` in UTC)    **or** fallback: `h3_hour_users.parquet`

---

## Local development

```
# 1) Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Dependencies
pip install -r requirements.txt

# 3) (Optional) create .streamlit/secrets.toml as above
# 4) Run
streamlit run Home.py
```

---

## Deploy on Streamlit Cloud

1. Push the repository to **GitHub**.
2. In Streamlit Cloud: **New app** → pick repo/branch → set **main file** to `app.py`.
3. Add **Secrets** as shown above.
5. **Deploy**.

**Updating:** Push new commits to GitHub; the app redeploys automatically. If you modify **Secrets**, use the app menu → **Reboot app** afterwards.

---

## How Drive loading & caching works

- `data_loader.py` reads `secrets["drive"]["folder_url"]`, extracts the folder ID, and uses `gdown.download_folder(...)` to mirror the entire Drive folder into a local cache:
  - `.cache/drive/<folder_id>/`
- On subsequent runs, Parquet files are read **from the local cache**.
- Streamlit uses `@st.cache_data(ttl=...)` to memoize dataframe loading and the list of files.

**Refreshing data**
- If you update files on Drive but keep the same names, use the **Streamlit menu → Clear cache**.
- If you add/rename files, clear cache as above; the app re‑lists and re‑loads them on next run.
- After editing **Secrets**, **Reboot app** from the Streamlit menu.
