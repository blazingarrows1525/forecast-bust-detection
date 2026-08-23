"""Central configuration for the Forecast Bust Detection system (SIH26079).

Every magic number in this project lives here so that a reviewer can audit the
whole experimental setup from one file.  See LOGIC.md for the reasoning behind
each choice; the short "why" is repeated inline.
"""
from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
ARTIFACTS = DATA / "artifacts"

IMD_RAW = RAW / "imd"
SHAPES_RAW = RAW / "shapes"
WB2_RAW = RAW / "wb2"

for _p in (RAW, INTERIM, PROCESSED, ARTIFACTS, IMD_RAW, SHAPES_RAW, WB2_RAW):
    _p.mkdir(parents=True, exist_ok=True)

SUBDIVISION_CONFIG = CONFIG_DIR / "imd_subdivisions.json"
SUBDIVISION_GPKG = INTERIM / "imd_subdivisions.gpkg"

# --------------------------------------------------------------------------
# Domain: India box and season.  LOGIC.md sec 3.
# --------------------------------------------------------------------------
INDIA_BBOX = dict(lat_min=6.0, lat_max=38.0, lon_min=66.0, lon_max=100.0)

# Regime features need the circulation that *drives* Indian rainfall, which
# lives well outside the India box: the Somali jet crosses 45-55 E, monsoon
# depressions form over the Bay of Bengal, and western disturbances arrive from
# Iran/Afghanistan.  Cropping to India would decapitate every one of those
# signals, so large-scale fields use this wider domain.
MONSOON_DOMAIN = dict(lat_min=-10.0, lat_max=45.0, lon_min=40.0, lon_max=110.0)

# Pressure levels retained from the 3-D ERA5 fields.
LEVELS = (200, 500, 850)

# JJAS.  The monsoon is where the decisions and the busts are (LOGIC.md sec 3).
SEASON_MONTHS = (6, 7, 8, 9)

# --------------------------------------------------------------------------
# Temporal split.  MUST be a whole held-out year, never a random split
# (LOGIC.md sec 8.3).  HRES archive covers 2016-2022.
# --------------------------------------------------------------------------
TRAIN_YEARS = (2016, 2017, 2018, 2019, 2020)
VAL_YEARS = (2021,)          # used for threshold / calibration selection
TEST_YEARS = (2022,)         # touched only at final evaluation
ALL_YEARS = TRAIN_YEARS + VAL_YEARS + TEST_YEARS

# --------------------------------------------------------------------------
# Lead times.  Day 1-10 produced; Day 3-7 is the decision-relevant band that we
# optimise and report on (LOGIC.md sec 4.2).
# --------------------------------------------------------------------------
LEAD_DAYS = tuple(range(1, 11))
DECISION_BAND = (3, 4, 5, 6, 7)

# --------------------------------------------------------------------------
# Bust definition.  LOGIC.md sec 4.3.  These are the numbers that define the
# entire project, so they are named, not inlined.
# --------------------------------------------------------------------------
# IMD daily rainfall intensity categories (mm/day), lower bounds.
# Source: IMD rainfall classification used in operational bulletins.
RAIN_CATEGORIES = [
    ("no_rain", 0.0, 2.5),
    ("light", 2.5, 15.6),
    ("moderate", 15.6, 64.5),
    ("heavy", 64.5, 115.5),
    ("very_heavy", 115.5, 204.5),
    ("extremely_heavy", 204.5, float("inf")),
]
# Index at/above which a category counts as "high impact" (heavy and above).
HEAVY_CATEGORY_INDEX = 3

# A bust requires a category jump of at least this many classes ...
BUST_MIN_CATEGORY_JUMP = 1
# ... with at least one side in the heavy+ band, OR an absolute area-mean error
# above the per-(subdivision, month) percentile below.
BUST_ERROR_PERCENTILE = 95.0
# Floor on absolute error (mm/day): below this a "bust" is not decision-relevant
# even if the percentile is tiny for a dry subdivision.
BUST_MIN_ABS_ERROR_MM = 10.0

# --------------------------------------------------------------------------
# Decision cost.  A missed bust is far worse than a false low-confidence flag
# (LOGIC.md sec 8.4).  A false flag costs ~10 forecaster-minutes.
# --------------------------------------------------------------------------
COST_MISSED_BUST = 10.0
COST_FALSE_ALARM = 1.0

# --------------------------------------------------------------------------
# Regime taxonomy.  Soft probabilities, never a hard label (LOGIC.md sec 5.1).
# --------------------------------------------------------------------------
REGIMES = (
    "active_monsoon",
    "break_monsoon",
    "monsoon_depression",
    "western_disturbance",
    "orographic",
    "coastal",
)

# Monsoon Core Zone used for the active/break index (Rajeevan et al. 2010).
MONSOON_CORE_ZONE = dict(lat_min=18.0, lat_max=28.0, lon_min=65.0, lon_max=88.0)

# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
WB2_BUCKET = "weatherbench2"
# Deterministic HRES forecasts at 0.703 deg (512x256).  WB2 zarr chunks span the
# whole globe, so resolution is a pure bandwidth decision, not a subsetting one:
# 0.703 deg costs ~9 GB of transfer and gives India ~48x45 cells, enough that a
# subdivision area-mean measures forecast error rather than grid mismatch.
# The 1.5 deg store is kept as a fast fallback.  See DECISIONS.md D-002/D-003.
HRES_STORE = "datasets/hres/2016-2022-0012-512x256_equiangular_conservative.zarr"
HRES_STORE_COARSE = (
    "datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr"
)
ENS_STORE = (
    "datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr"
)
# CAREFUL: the WB2 ERA5 stores named "1959-2022" actually END on 2021-12-31.
# Verified empirically, not from the filename.  Using them would have left the
# 2022 held-out test year with no atmospheric features at all -- a failure that
# would have surfaced only at final evaluation.  Only the "1959-2023_01_10"
# stores cover JJAS 2022.  See DECISIONS.md D-009.
ERA5_STORE_2D = (
    "datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
ERA5_STORE_3D = ERA5_STORE_2D  # same store carries the pressure-level fields

IMD_RF25_POST_URL = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"
IMD_RF25_REFERER = (
    "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"
)

# --------------------------------------------------------------------------
# Data quality gates.  LOGIC.md sec 10 / sec 13.
# --------------------------------------------------------------------------
# Physical range checks (min, max) applied before a field is allowed into
# feature engineering.  A field failing these is rejected, not clipped.
PHYSICAL_RANGES = {
    "rainfall_mm": (0.0, 2000.0),
    "t2m_k": (180.0, 340.0),
    "msl_pa": (85000.0, 110000.0),
    "z500_m2s2": (45000.0, 60000.0),
    "q850_kgkg": (0.0, 0.05),
    "wind_ms": (-150.0, 150.0),
    "tcwv_kgm2": (0.0, 120.0),
}
# Maximum tolerated fraction of NaN inside a subdivision mask before the
# (subdivision, date) truth value is marked unusable.
MAX_NAN_FRACTION = 0.40
# Input age (hours) beyond which the API greys out and shows a STALE banner.
STALE_INPUT_HOURS = 36

# Minimum number of IMD land grid cells for a subdivision to be modelled.
MIN_GRID_CELLS_PER_SUBDIVISION = 3

RANDOM_SEED = 20260920


def load_subdivision_config() -> dict:
    """Return the parsed IMD subdivision definition."""
    with open(SUBDIVISION_CONFIG, "r", encoding="utf-8") as fh:
        return json.load(fh)


def subdivision_names() -> dict[str, str]:
    cfg = load_subdivision_config()["subdivisions"]
    return {k: v["name"] for k, v in cfg.items()}
