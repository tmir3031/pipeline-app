import os, re
import streamlit as st
import pandas as pd
import gdown
from glob import glob

CACHE_TTL = int(st.secrets.get("app", {}).get("cache_ttl", 3600))

def _extract_folder_id(url_or_id: str) -> str:
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
    return m.group(1) if m else url_or_id

@st.cache_data(ttl=CACHE_TTL, show_spinner=True)
def _download_drive_folder(url_or_id: str):
    folder_id = _extract_folder_id(url_or_id)
    local_dir = os.path.join(".cache", "drive", folder_id)
    os.makedirs(local_dir, exist_ok=True)

    # dacă e gol, descarcă
    is_empty = not any(True for _ in os.scandir(local_dir))
    if is_empty:
        gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{folder_id}",
            output=local_dir,
            quiet=True,
            use_cookies=False,
        )

    files = [p for p in glob(os.path.join(local_dir, "**"), recursive=True) if os.path.isfile(p)]
    return local_dir, sorted(files)

def list_drive_files() -> list[str]:
    local_dir, files = _download_drive_folder(st.secrets["drive"]["folder_url"])
    return [os.path.relpath(p, local_dir) for p in files]

@st.cache_data(ttl=CACHE_TTL, show_spinner=True)
def load_parquet_from_folder(name_or_path: str, columns: list[str] | None = None) -> pd.DataFrame:
    local_dir, files = _download_drive_folder(st.secrets["drive"]["folder_url"])
    candidates = [p for p in files if os.path.basename(p) == name_or_path or p.endswith(name_or_path)]
    if not candidates:
        raise FileNotFoundError(f"Nu am găsit '{name_or_path}' în folderul Drive.")
    return pd.read_parquet(candidates[0], columns=columns)

@st.cache_data(ttl=CACHE_TTL, show_spinner=True)
def load_parquet_dataset(columns: list[str] | None = None) -> pd.DataFrame:
    local_dir, _ = _download_drive_folder(st.secrets["drive"]["folder_url"])
    return pd.read_parquet(local_dir, columns=columns)
