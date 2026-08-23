"""Build the observed-rainfall truth table: subdivision x day area-means (JJAS).

Output: data/interim/truth_subdivision_daily.parquet
Also writes a coverage report so that subdivisions without usable IMD gauge
coverage are excluded explicitly and visibly, rather than by accident.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.ingest import imd  # noqa: E402

OUT = config.INTERIM / "truth_subdivision_daily.parquet"
COVERAGE = config.INTERIM / "subdivision_coverage.csv"


def main() -> int:
    t0 = time.time()

    cov = imd.grid_cell_counts()
    cov["modelled"] = cov.n_land_cells >= config.MIN_GRID_CELLS_PER_SUBDIVISION
    cov.to_csv(COVERAGE, index=False)
    dropped = cov.loc[~cov.modelled, "subdivision_id"].tolist()
    print(f"subdivisions with usable IMD coverage: {int(cov.modelled.sum())}/{len(cov)}")
    if dropped:
        print(
            "  EXCLUDED (no IMD gauge grid cells -- the 0.25 deg product is "
            f"mainland-only): {dropped}"
        )

    truth = imd.subdivision_daily()
    truth = truth[truth.subdivision_id.isin(cov.loc[cov.modelled, "subdivision_id"])]
    truth["year"] = truth.valid_date.dt.year
    truth["month"] = truth.valid_date.dt.month
    truth.to_parquet(OUT, index=False)

    print(f"\nwrote {OUT}  rows={len(truth):,}  in {time.time()-t0:.0f}s")
    print(f"  date range: {truth.valid_date.min().date()} .. {truth.valid_date.max().date()}")
    print(f"  subdivisions: {truth.subdivision_id.nunique()}")
    print(f"  missing obs_rain_mm: {truth.obs_rain_mm.isna().mean():.4%}")
    print("\nmean JJAS rainfall (mm/day) by subdivision, wettest first:")
    agg = (
        truth.groupby("subdivision_id")
        .obs_rain_mm.agg(["mean", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    print(agg.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
