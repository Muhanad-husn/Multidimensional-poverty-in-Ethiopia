"""Data loading — DHS 2019 recodes and GADM admin-1 boundaries.

Path constants are resolved relative to the project root (the folder containing
pyproject.toml). DHS microdata is placed manually in ``data/raw/`` by the user and
is never committed (license forbids redistribution); GADM boundaries are downloaded
through a cached HTTP session into ``data/external/``.

Status: implemented through Session 2 — recode loaders, GADM admin-1 loader, and
the DHS-region <-> GADM-admin1 crosswalk (see IMPLEMENTATION_PLAN.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat
import requests_cache

# This file lives at src/ethiopia_mpi/data.py
# parents[0] = src/ethiopia_mpi/  parents[1] = src/  parents[2] = project root
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / ".cache"

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# DHS Ethiopia 2019 recodes expected in data/raw/. Each downloaded ZIP unpacks
# into its own ET<rec>81DT/ folder containing ET<rec>81FL.DTA.
DHS_RECODES = {
    "HR": "ETHR81DT/ETHR81FL.DTA",  # Household Recode
    "PR": "ETPR81DT/ETPR81FL.DTA",  # Household Member Recode
    "IR": "ETIR81DT/ETIR81FL.DTA",  # Individual (women) Recode
    "BR": "ETBR81DT/ETBR81FL.DTA",  # Births Recode
    "KR": "ETKR81DT/ETKR81FL.DTA",  # Children's Recode (child anthropometry)
}

# Shared HTTP cache (gitignored) — used for the GADM boundary download.
session = requests_cache.CachedSession(
    cache_name=str(CACHE_DIR / "http_cache"),
    backend="sqlite",
    expire_after=60 * 60 * 24 * 30,
    allowable_methods=("GET", "HEAD"),
)


def download_file(url: str, dest: str | Path, force: bool = False) -> Path:
    """Download a file to ``dest`` (relative paths resolve against ``EXTERNAL_DIR``)."""
    dest = Path(dest)
    if not dest.is_absolute():
        dest = EXTERNAL_DIR / dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        return dest

    resp = session.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest


def read_processed(name: str) -> pd.DataFrame:
    """Read a parquet from ``data/processed/<name>.parquet``."""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def write_processed(df: pd.DataFrame, name: str) -> Path:
    """Write a DataFrame to ``data/processed/<name>.parquet``."""
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


# --- DHS recode loading -------------------------------------------------------

def load_dhs_recode(
    recode: str,
    raw_dir: str | Path | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a DHS Ethiopia 2019 recode from a ``data/raw``-style directory.

    Parameters
    ----------
    recode
        One of ``DHS_RECODES`` keys: ``HR`` (Household), ``PR`` (Household
        Member), ``IR`` (Individual/women), ``BR`` (Births), ``KR`` (Children's).
    raw_dir
        Directory holding the ``ET<rec>81DT/`` subfolders. Defaults to the
        project's ``data/raw/``. Tests point this at ``tests/fixtures/raw/``.
    columns
        Optional subset of columns to read. The real recodes are very wide
        (HR ~1,400 cols, IR ~4,400) — passing only the needed columns keeps
        memory and load time down.

    Returns
    -------
    DataFrame with numeric value codes (categoricals are *not* converted to
    their labels — MPI construction operates on the codes).

    Notes
    -----
    HR/PR carry the ``hv*`` household-recode columns; IR/BR/KR carry the
    ``v*`` woman/birth columns. Survey design: weight ``*v005``/1e6, strata
    ``*v022``/``*v023``, cluster/PSU ``*v021``.
    """
    recode = recode.upper()
    if recode not in DHS_RECODES:
        raise ValueError(
            f"Unknown recode {recode!r}; expected one of {sorted(DHS_RECODES)}."
        )

    base = Path(raw_dir) if raw_dir is not None else RAW_DIR
    path = base / DHS_RECODES[recode]
    if not path.exists():
        raise FileNotFoundError(
            f"DHS recode {recode} not found at {path}. Raw DHS microdata is not "
            f"committed (license forbids redistribution) — see data/raw/README.md "
            f"for the access workflow."
        )

    df, _meta = pyreadstat.read_dta(str(path), usecols=columns)
    return df


# --- GADM admin-1 boundaries --------------------------------------------------

# GADM 4.1 country GeoPackage for Ethiopia (all admin levels, one layer each).
GADM_ETH_GPKG_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_ETH.gpkg"


def load_gadm_admin1():
    """Load Ethiopia GADM 4.1 admin-1 (region) polygons as a GeoDataFrame.

    Downloads (and caches) the GADM 4.1 Ethiopia GeoPackage into
    ``data/external/`` on first use, then reads the admin-1 layer.

    Returns
    -------
    geopandas.GeoDataFrame with admin-1 polygons; ``NAME_1`` holds the region
    name used by :func:`region_crosswalk`.
    """
    import geopandas as gpd

    path = download_file(GADM_ETH_GPKG_URL, "gadm41_ETH.gpkg")
    gdf = gpd.read_file(path, layer="ADM_ADM_1")
    return gdf


# --- DHS region <-> GADM admin-1 crosswalk ------------------------------------

# DHS Ethiopia 2019 region codes (hv024/v024). Confirmed against the value
# labels of the real HR recode. ``gadm_name`` matches GADM 4.1 admin-1 NAME_1
# spellings — verified against the downloaded GADM layer in Session 4. The only
# scaffold correction was SNNPR: GADM's NAME_1 is "Southern Nations,
# Nationalities" (truncated, no trailing "and Peoples").
_REGION_CROSSWALK = [
    (1, "tigray", "Tigray"),
    (2, "afar", "Afar"),
    (3, "amhara", "Amhara"),
    (4, "oromia", "Oromia"),
    (5, "somali", "Somali"),
    (6, "benishangul-gumuz", "Benshangul-Gumaz"),
    (7, "snnpr", "Southern Nations, Nationalities"),
    (8, "gambela", "Gambela Peoples"),
    (9, "harari", "Harari People"),
    (10, "addis ababa", "Addis Abeba"),
    (11, "dire dawa", "Dire Dawa"),
]


def region_crosswalk() -> pd.DataFrame:
    """DHS region code <-> GADM admin-1 name crosswalk.

    Returns
    -------
    DataFrame with columns ``dhs_code`` (int, matches ``hv024``/``v024``),
    ``dhs_region`` (DHS value label), and ``gadm_name`` (GADM 4.1 ``NAME_1``).

    The ``gadm_name`` column is a scaffold — spellings are verified against the
    actual GADM layer in Session 4.
    """
    return pd.DataFrame(
        _REGION_CROSSWALK, columns=["dhs_code", "dhs_region", "gadm_name"]
    )
