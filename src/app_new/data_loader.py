import os, re
import streamlit as st
import pandas as pd
import gdown
from glob import glob

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# Cache Time-To-Live (seconds) for Streamlit's data cache. This bounds how long
# previously computed results (including the mirrored Drive folder listing) are
# retained before the function is re-executed. Defaults to 3600 s if not set.
CACHE_TTL = int(st.secrets.get("app", {}).get("cache_ttl", 3600))


# ---------------------------------------------------------------------
# Utility: extract a Google Drive folder ID from a full URL
# ---------------------------------------------------------------------
def _extract_folder_id(url_or_id: str) -> str:
    """
    Returns the Drive folder ID when given either:
    - a full folder URL (e.g., https://drive.google.com/drive/folders/<ID>?usp=sharing), or
    - a bare folder ID (in which case it is returned unchanged).

    The regex captures the token following '/folders/' using a conservative
    character class (alphanumeric, underscore, hyphen).
    """
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
    return m.group(1) if m else url_or_id


# ---------------------------------------------------------------------
# Mirror the entire Drive folder locally (once per cache window)
# ---------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=True)
def _download_drive_folder(url_or_id: str):
    """
    Mirrors a *public* Google Drive folder (shared as 'Anyone with the link — Viewer')
    into a deterministic local cache directory: .cache/drive/<FOLDER_ID>.

    Returns
    -------
    local_dir : str
        Absolute or relative path to the local mirror root.
    files : list[str]
        Sorted list of absolute file paths (recursive) within the mirror.

    Notes
    -----
    - This function is cached via Streamlit (@st.cache_data), so the download is
      performed only when the cache is cold or expired.
    - `use_cookies=False` instructs gdown to use unauthenticated access, which
      requires the folder to be public.
    """
    folder_id = _extract_folder_id(url_or_id)
    local_dir = os.path.join(".cache", "drive", folder_id)
    os.makedirs(local_dir, exist_ok=True)

    # dacă e gol, descarcă
    # If the mirror directory is empty, perform an initial download. Subsequent
    # calls will reuse the cached files until the cache TTL expires or the
    # directory is manually cleared.
    is_empty = not any(True for _ in os.scandir(local_dir))
    if is_empty:
        gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{folder_id}",
            output=local_dir,
            quiet=True,
            use_cookies=False,  # public folders only
        )

    # Recursively enumerate all regular files inside the mirror directory.
    files = [p for p in glob(os.path.join(local_dir, "**"), recursive=True) if os.path.isfile(p)]
    return local_dir, sorted(files)


# ---------------------------------------------------------------------
# Public API: list files (relative to the mirror root)
# ---------------------------------------------------------------------
def list_drive_files() -> list[str]:
    """
    Lists all files discovered in the mirrored Drive folder as *relative paths*
    (relative to the mirror root). This is convenient for presence checks and
    downstream matching by suffix/basename.
    """
    local_dir, files = _download_drive_folder(st.secrets["drive"]["folder_url"])
    return [os.path.relpath(p, local_dir) for p in files]


# ---------------------------------------------------------------------
# Public API: load a single Parquet file by basename or suffix path
# ---------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=True)
def load_parquet_from_folder(name_or_path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Loads exactly one Parquet file from the mirrored folder.

    Matching strategy
    -----------------
    - If `name_or_path` is a basename (e.g., 'file.parquet'), match by basename.
    - If it is a relative suffix (e.g., 'ads/candidates_5.parquet'), match any
      file whose path ends with that suffix.

    Parameters
    ----------
    name_or_path : str
        Basename or suffix path identifying the Parquet file to load.
    columns : list[str] | None
        Optional column projection for efficient reads.

    Raises
    ------
    FileNotFoundError
        If no file in the mirrored folder matches the requested name/suffix.

    Notes
    -----
    - The function is cached, so repeated reads of the same file/columns within
      the TTL window will be served from memory.
    """
    local_dir, files = _download_drive_folder(st.secrets["drive"]["folder_url"])
    candidates = [p for p in files if os.path.basename(p) == name_or_path or p.endswith(name_or_path)]
    if not candidates:
        raise FileNotFoundError(f"Not found '{name_or_path}' in folder Drive.")  # Original message retained
    return pd.read_parquet(candidates[0], columns=columns)


# ---------------------------------------------------------------------
# Public API: load the entire mirror as a Parquet *dataset*
# ---------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=True)
def load_parquet_dataset(columns: list[str] | None = None) -> pd.DataFrame:
    """
    Treats the mirrored folder as a Parquet *dataset* and loads it in one call.
    This is appropriate when the directory contains multiple Parquet parts with
    a compatible schema (e.g., partitioned by hour/date).

    Parameters
    ----------
    columns : list[str] | None
        Optional column projection to reduce I/O and memory footprint.

    Returns
    -------
    pd.DataFrame
        Concatenated view over all Parquet parts discovered under the mirror.
    """
    local_dir, _ = _download_drive_folder(st.secrets["drive"]["folder_url"])
    return pd.read_parquet(local_dir, columns=columns)
