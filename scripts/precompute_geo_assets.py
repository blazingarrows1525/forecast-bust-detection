"""Precompute the two geo assets the API serves, so the image can drop geopandas.

Why this exists (cloud optimisation, D-020)
-------------------------------------------
The serving image installed ``geopandas``, ``shapely`` and ``pyproj`` purely to
satisfy two read-only endpoints:

  * ``/api/regions``  -- read the GeoPackage, simplify, emit GeoJSON
  * ``_centroids()``  -- read the GeoPackage, take a representative point

Those pull in GDAL, GEOS and PROJ: hundreds of megabytes of native binaries in
every image layer, on every pull, for output that is **identical on every
request** because the geometry never changes at runtime.

Precomputing both to static JSON at build time removes the entire geo stack
from the serving dependency set. The API falls back to the GeoPackage if the
precomputed files are absent, so a development checkout with geopandas
installed still works unchanged.

Run:  PYTHONPATH=src python scripts/precompute_geo_assets.py
Out:  data/interim/regions.geojson
      data/interim/centroids.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402

# Matches the simplification the API applied at request time. Full-resolution
# district unions are ~10 MB of coordinates, which is a slow payload for a
# browser and pointless at map zoom levels.
SIMPLIFY_TOLERANCE = 0.02

GEOJSON_OUT = config.INTERIM / "regions.geojson"
CENTROIDS_OUT = config.INTERIM / "centroids.json"


def main() -> int:
    import geopandas as gpd

    src = config.SUBDIVISION_GPKG
    if not src.exists():
        print(f"[FAIL] {src} missing -- run `python -m fbd.regions.build` first")
        return 1

    g = gpd.read_file(src, layer="subdivisions")
    print(f"read {len(g)} subdivisions from {src.name}")

    # 1. Simplified GeoJSON for the map layer.
    simplified = g.copy()
    simplified["geometry"] = simplified.geometry.simplify(
        SIMPLIFY_TOLERANCE, preserve_topology=True
    )
    GEOJSON_OUT.write_text(simplified.to_json(), encoding="utf-8")

    # 2. Representative interior points for placing the 3-D risk columns.
    #    `representative_point` rather than `centroid`: a centroid can fall
    #    outside a concave polygon, which would float Konkan & Goa's column out
    #    over the Arabian Sea.
    pts = g.geometry.representative_point()
    centroids = {
        sid: [round(float(p.x), 6), round(float(p.y), 6)]
        for sid, p in zip(g.subdivision_id, pts)
    }
    CENTROIDS_OUT.write_text(json.dumps(centroids, indent=1), encoding="utf-8")

    gj_mb = GEOJSON_OUT.stat().st_size / 1e6
    ct_kb = CENTROIDS_OUT.stat().st_size / 1e3
    src_mb = src.stat().st_size / 1e6
    print(f"  {GEOJSON_OUT.name:22s} {gj_mb:6.2f} MB")
    print(f"  {CENTROIDS_OUT.name:22s} {ct_kb:6.1f} KB  ({len(centroids)} regions)")
    print(f"  (replaces {src.name}, {src_mb:.2f} MB, plus the whole geo stack)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
