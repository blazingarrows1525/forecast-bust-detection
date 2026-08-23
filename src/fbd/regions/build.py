"""Build the 36 IMD meteorological subdivisions from Census-2011 districts.

The spatial unit is the single most consequential modelling choice after the
bust definition itself (LOGIC.md sec 4.1): grid-cell bust statistics are noisy
and operationally meaningless, because IMD issues bulletins and alerts at
subdivision/district level.

This module is deliberately strict.  If a single district fails to map, or maps
twice, it raises.  A silently mis-assigned district would corrupt every label
downstream and would be almost impossible to spot later.
"""
from __future__ import annotations

import sys
from collections import Counter

import geopandas as gpd
import pandas as pd

from fbd import config


DISTRICT_SHP = config.SHAPES_RAW / "2011_Dist.shp"


def _expand_members(members: list[str], districts: pd.DataFrame) -> list[int]:
    """Resolve 'State/District' and 'State/*' member strings to row indices."""
    idx: list[int] = []
    for m in members:
        state, _, dist = m.partition("/")
        if dist == "*":
            hit = districts.index[districts["ST_NM"] == state].tolist()
            if not hit:
                raise KeyError(f"state not found in shapefile: {state!r}")
        else:
            hit = districts.index[
                (districts["ST_NM"] == state) & (districts["DISTRICT"] == dist)
            ].tolist()
            if not hit:
                raise KeyError(f"district not found in shapefile: {m!r}")
        idx.extend(hit)
    return idx


def assign_subdivisions(districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach subdivision id/name columns to the district frame, validating
    that the mapping is a exact partition of all districts."""
    cfg = config.load_subdivision_config()["subdivisions"]

    districts = districts.copy()
    districts["subdivision_id"] = pd.NA
    districts["subdivision_name"] = pd.NA

    assigned = Counter()
    for sid, spec in cfg.items():
        rows = _expand_members(spec["members"], districts)
        for r in rows:
            assigned[r] += 1
        districts.loc[rows, "subdivision_id"] = sid
        districts.loc[rows, "subdivision_name"] = spec["name"]

    # --- hard validation: exact partition -------------------------------
    dupes = [r for r, n in assigned.items() if n > 1]
    if dupes:
        detail = districts.loc[dupes, ["ST_NM", "DISTRICT"]]
        raise ValueError(
            f"{len(dupes)} district(s) assigned to more than one subdivision:\n"
            f"{detail.to_string()}"
        )

    unmapped = districts.index.difference(list(assigned))
    if len(unmapped):
        detail = districts.loc[unmapped, ["ST_NM", "DISTRICT"]]
        raise ValueError(
            f"{len(unmapped)} district(s) not assigned to any subdivision:\n"
            f"{detail.to_string()}"
        )
    return districts


def build(write: bool = True) -> gpd.GeoDataFrame:
    if not DISTRICT_SHP.exists():
        raise FileNotFoundError(
            f"{DISTRICT_SHP} missing -- run scripts/fetch_boundaries.py first"
        )
    districts = gpd.read_file(DISTRICT_SHP)
    districts = assign_subdivisions(districts)

    subs = (
        districts.dissolve(by="subdivision_id", aggfunc={"subdivision_name": "first"})
        .reset_index()
        .rename(columns={"subdivision_name": "name"})
    )
    # Repair any self-intersections introduced by dissolving neighbours.
    subs["geometry"] = subs.geometry.buffer(0)
    subs["n_districts"] = (
        districts.groupby("subdivision_id").size().reindex(subs.subdivision_id).values
    )
    # Equal-area CRS for an honest area figure (India Albers-ish: EPSG:7755).
    subs["area_km2"] = subs.to_crs(7755).area / 1e6
    subs = subs.sort_values("subdivision_id").reset_index(drop=True)

    if write:
        config.SUBDIVISION_GPKG.parent.mkdir(parents=True, exist_ok=True)
        subs.to_file(config.SUBDIVISION_GPKG, layer="subdivisions", driver="GPKG")
    return subs


def main() -> int:
    subs = build()
    print(f"built {len(subs)} IMD subdivisions -> {config.SUBDIVISION_GPKG}")
    with pd.option_context("display.width", 140, "display.max_rows", 60):
        print(
            subs[["subdivision_id", "name", "n_districts", "area_km2"]]
            .assign(area_km2=lambda d: d.area_km2.round(0).astype(int))
            .to_string(index=False)
        )
    print(f"\ntotal districts covered: {int(subs.n_districts.sum())}")
    print(f"total area: {subs.area_km2.sum():,.0f} km2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
