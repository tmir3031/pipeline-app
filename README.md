# Geolife Trajectory Pipeline & Ad Planner

Modular pipeline for ingesting, cleaning and analyzing GPS data from **Geolife**, with a UI (Streamlit) for exploration, POI planning (Ad Planner) and evaluation.
The project aims to be **reproducible**, **configurable** and easily extensible.

---

## Project structure

```
project-big-data/
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── geo/
│   │   ├── h3_frontier.geojson
│   │   ├── h3_polygons.geojson
│   │   └── ads_candidates.geojson
│   ├── parquet/
│   │   ├── ads/
│   │   └── ...
│   └── raw/                          
├── output/
│   ├── logs/
│   └── reports/
├── src/
│   ├── app_new/
│   │   ├── app.py                    # app Streamlit
│   │   └── pages/
│   │       ├── 01_Explore.py
│   │       ├── 02_AdPlanner.py
│   │       └── 03_Evaluation.py
│   └── tools/
│       ├── 01_ingest.py
│       ├── 02_reduce.py
│       ├── 03_h3_assign.py
│       ├── 04_aggregate_temporal.py
│       ├── 05_export_geojson.py
│       ├── 06_kde.py
│       ├── 07_ads_candidates.py
│       └── 08_eval.py
└── (venv*/, node_modules/ – pot fi ignorate)
```

> Note: files and directories in `data/` may be generated progressively by pipeline steps; not all of them will exist at the start.

---

## Installation

1) Python 3.9+ recommended (ideally 3.10/3.11).
2) Create and activate a virtual environment:
```
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```
3) Install the dependencies:
```
pip install -r requirements.txt
```

---

## Configuration (config.yaml)

All parameters are centralized in `config.yaml`.

> Adjust parameters according to volume, granularity and analysis goals.
 
---

## Pipeline run (step by step)

All scripts accept `--config config.yaml`.

1) **Ingest** (PLT -> Parquet)
```
python src/tools/01_ingest.py --config config.yaml
```
- Reads `.plt` files from `data/raw/` and produces `data/parquet/points.parquet`.

2) **Reduce / cleaning**
```
python src/tools/02_reduce.py --config config.yaml
```
- Segments on gaps, computes speed/acceleration, applies filters (Hampel), buckets to 1 point / 5s.
- Output: `data/parquet/points_reduced.parquet` + reports în `output/reports/`.

3) **H3 assignment**
```
python src/tools/03_h3_assign.py --config config.yaml
```
- Maps points to H3 cells (`resolution` from config). 
- Output: `data/parquet/points_with_h3.parquet`.

4) **Temporal aggregation**
```
python src/tools/04_aggregate_temporal.py --config config.yaml
```
- Aggregates at the configured frequency (e.g., hourly).
- Results are written to `data/parquet/ads/` (e.g., `h3_hour_users.parquet`, `city_hour_users.parquet` etc.).

5) **GeoJSON export / GIS utilities**
```
python src/tools/05_export_geojson.py --config config.yaml
```
- Generates H3 polygons, boundary, or other layers in `data/geo/` (e.g., `h3_polygons.geojson`, `h3_frontier.geojson`).

6) **Spatial density (KDE)**
```
python src/tools/06_kde.py --config config.yaml
```
- Computes density surfaces and derived outputs (e.g., `kde_in_h3.parquet`).

7) **Generate candidates for Ad Planner**
```
python src/tools/07_ads_candidates.py --config config.yaml
```
- Proposes candidate locations based on traffic/criteria from `ads.*` (scrie și GeoJSON în `data/geo/ads_candidates.geojson`).

8) **Evaluation / scoring**
```
python src/tools/08_eval.py --config config.yaml
```
- Evaluation methodologies, scores, and reports in `output/reports/`.

---

## UI Application (Streamlit)

The app reads the results generated in the steps above.

```
# from the repository root
streamlit run src/app_new/app.py
```

Pages:
- **01 – Explore**: data loading/filters, maps, statistics.
- **02 – AdPlanner**: candidate selection, constraints (e.g., top-k per cell, min. users).
- **03 – Evaluation**: compare scenarios, export reports.

---

## Artifacts & outputs

- `data/parquet/points*.parquet` – point-level.  
- `data/parquet/ads/*.parquet` – aggregates (e.g., hourly, per H3 cell, user types).  
- `data/geo/*.geojson` – map layers (boundary, H3 polygons, candidates).  
- `output/logs/` – execution logs.  
- `output/reports/` – statistics, filtering audit, evaluations.

---

## Methodological notes

- **Timezone**: the source is generally Beijing local time (≈ UTC+8). Standard conversion: `source_tz -> UTC`.  
- **Spatial discretization**: H3 (`resolution` from config) for robust aggregated analyses.  
- **Reproducibility**: All steps depend only on `data/raw` and `config.yaml`; runs reproduce the same results with the same configuration.

---

## Troubleshooting

- **Geopandas/Shapely on Windows** – install compatible binary wheels (use Python 3.10+).  
- **Memory** – for large files, enable partitioned writes (if implemented) and run steps incrementally.  
- **Empty maps in Streamlit** – verify the presence of files in `data/geo/` and aggregates in `data/parquet/ads/`.

---

## References

- Microsoft Research Asia – *Geolife GPS Trajectory Dataset User Guide*  
- Standard GPS trajectory preprocessing literature (segmentation, speed/acceleration thresholds, Hampel filtering)  
- H3 — documentation
