# HANDOFF — Forecast Bust Detection (SIH26079)

**Read this first in a new Claude Code / Antigravity session.**  
It replaces the need to re-read earlier conversation history.

- **Canonical Specification & Engineering Contract:** [`LOGIC.md`](file:///c:/Users/ASUS/Desktop/sih/LOGIC.md) (LOCKED — do not silently deviate)
- **Decision & Deviation Log:** [`DECISIONS.md`](file:///c:/Users/ASUS/Desktop/sih/DECISIONS.md) (D-001 through D-013 with evidence)
- **Public Writeup & Results:** [`README.md`](file:///c:/Users/ASUS/Desktop/sih/README.md)
- **Central Configuration:** [`src/fbd/config.py`](file:///c:/Users/ASUS/Desktop/sih/src/fbd/config.py)

Session state as of hand-off: **~15:45 IST, 23 Aug 2026**.

---

## 1. Where the Project Stands

The system is a fully functional, end-to-end operational meta-model that predicts **when an existing medium-range rainfall forecast over India is about to fail (bust)**, which subdivision, which lead day, and why.

### Headline Results on Held-Out Test Year (2022)
**Day 3–7 Decision Band, 20,060 subdivision-days, bust base rate 3.31%.**  
Every predictor is evaluated under the *same* Isotonic Calibration fitted on the 2021 validation year:

| Predictor | AUROC | Brier | BSS | ECE | Decision Cost / 1000 | Value |
|---|---|---|---|---|---|---|
| Climatology (region, month, lead) | 0.519 | 0.0321 | −0.005 | 0.013 | 330.5 | 0.000 |
| Forecast Rain Amount Only | 0.750 | 0.0311 | 0.028 | 0.014 | 273.0 | 0.174 |
| Logistic Regression (spread + lead) | 0.754 | 0.0310 | 0.031 | 0.010 | 293.4 | 0.112 |
| **Lagged Ensemble Spread (Proxy Baseline)** | **0.758** | 0.0310 | 0.031 | 0.011 | 293.5 | 0.112 |
| **XGBoost + Isotonic Calibration (Our Model)** | **0.840** | **0.0291** | **0.088** | **0.011** | **237.1** | **0.282** |

- **AUROC Margin:** **+0.082** over the lagged ensemble spread *proxy* -- but see below.
- **AUROC Margin vs the REAL 50-member IFS ENS: +0.025** (0.832 vs 0.807 on the
  6,732 decision-band rows where true ENS exists). This is now the number to
  defend in front of a judge; the +0.082 answers a weaker question. See **D-014**.
- **Economic Value:** **19.2% reduction** in asymmetric disaster decision costs ($C_{\text{miss}} = 10, C_{\text{fa}} = 1$).
- **Test Suite:** **21/21 tests pass** across `tests/test_core.py` and `tests/test_model_api.py`.
- **Offline Serving:** FastAPI backend serving precomputed SQLite bulletins (`data/artifacts/bulletins.sqlite`, 63 MB, 79,900 rows) with interactive Leaflet dashboard (vendored JS/CSS, zero CDN).

---

## 2. Recently Completed Background Tasks

1. **`scripts/fetch_ens.py` (2022, every 3rd init) -- DONE.**
   14,760 rows -> `data/raw/wb2/ens/ens_spread_2022_every3.parquet` in 14.8 min.
   Scored via `scripts/evaluate_ens_baseline.py --decision-band-only`; result
   recorded as **D-014** and reflected in the README headline table.
   Finding: true ENS spread scores **0.807 AUROC** vs the lagged proxy's 0.731,
   so the proxy understated the baseline and overstated our margin.
2. **Training-year ENS (2019 + 2020, every 6th init).** Fetched to unlock the
   *calibrated* half of the ENS comparison (Brier / BSS / decision cost), which
   `evaluate_ens_baseline.py` refuses to compute without training-year data
   (fitting isotonic on 2022 would be test-year leakage). Re-run the evaluation
   after the parquet files land to populate the blank cells in the README row.

## 3. The Full Pipeline in Execution Order

All raw datasets (HRES, ERA5, IMD) are locally cached in `data/raw/`. Once cached, the entire pipeline is reproducible in ~3 minutes from the repository root:

```bash
# 1. Spatial aggregation definition: 36 IMD subdivisions from 641 districts
python -m fbd.regions.build

# 2. Ingest IMD gridded observations (2016-2022) & build subdivision truth
python scripts/fetch_imd.py
python scripts/build_truth.py

# 3. Ingest HRES forecasts and ERA5 initial state
python scripts/fetch_hres.py
python scripts/fetch_era5.py

# 4. Feature engineering & dataset construction -> data/processed/dataset.parquet (279,650 rows x 93 cols)
python scripts/build_dataset.py

# 5. Train baselines, XGBoost meta-model & calibration -> data/artifacts/bust_model.joblib
python scripts/train_model.py

# 6. Generate SQLite bulletins with bagged uncertainty intervals & TreeSHAP reasons
python scripts/generate_bulletins.py

# 7. Verification, ablation and stress testing
python scripts/ablation.py
python scripts/stress_test.py
python -m pytest tests/ -v

# 8. Terminal demo & API server
python scripts/replay_demo.py
python -m uvicorn fbd.api.app:app --app-dir src --port 8912
```

---

## 4. Repository & Architecture Map

```
LOGIC.md                     Canonical engineering contract & mathematical specification (LOCKED)
DECISIONS.md                 Every deviation/refinement from LOGIC.md with evidence (D-001..D-013)
README.md                    Public overview, headline tables, and case study narrative
HANDOFF.md                   This session handoff document
config/imd_subdivisions.json Definition of 36 IMD subdivisions mapped to 641 Census-2011 districts

src/fbd/
  config.py                  Central constants: Bounding boxes, split years, bust thresholds, costs
  regions/
    build.py                 Constructs 36 subdivisions; validates exact partition of 641 districts
    masks.py                 Exact polygon-cell overlap weights using equal-area CRS (EPSG:7755)
  ingest/
    wb2.py                   WeatherBench 2 Zarr access; India subsetting; lead timedelta mapping
    imd.py                   IMD 0.25° NetCDF -> subdivision area-means (mainland gate excluded islands)
    hres.py                  Cached HRES -> subdivision forecasts & pairs with observed truth
  labels/
    bust.py                  The mathematical bust definition: Magnitude + Category Flip + Significant Rain
                             P95 thresholds fitted strictly on training years (2016-2020)
  features/
    forecast.py              Lagged ensemble spread (Hoffman & Kalnay 1983 proxy), jumpiness, anomalies
                             Strict causality: Lead L uses only leads >= L (verified by poison test)
    era5.py                  Somali Jet, Monsoon Trough, Bay of Bengal vorticity, TCWV, shear, MCZ flux
                             Causality: Joined on init_date, never valid_date
  regime/
    classify.py              6 synoptic regime soft probabilities via physical ERA5 scores + softmax
                             PLACE_WEIGHT=2.5 lifts static orographic/coastal terms appropriately
  model/
    train.py                 XGBClassifier (depth 4, lr 0.05, class-weighted) + Isotonic Calibration
    baselines.py             Climatology, Forecast-rain-only, Logistic, Spread baselines (same calibration)
  ood/
    detector.py              Mahalanobis distance OOD detector (99.5th percentile); earned refusal validation
  explain/
    reasons.py               Native TreeSHAP (pred_contribs=True) -> Direction-aware plain English reasons
                             FAMILY deduplication limits 1 factor per concept family; precomputed knots
  evaluate/
    metrics.py               AUROC, Brier, BSS, ECE (equal-count bins), Asymmetric Decision Cost, Value
  api/
    schema.py                Pydantic output schemas (BustPrediction, Bulletin, ReviewQueueItem)
    app.py                   FastAPI service (clock replay mode & live mode with staleness detection)

web/
  index.html                 Self-contained interactive Leaflet dashboard
  vendor/leaflet.{js,css}    Vendored assets (100% offline, zero CDN dependencies)

tests/
  test_core.py               12 unit tests: causality poison tests, exact partitions, bust label invariants
  test_model_api.py          9 unit tests: cost/value metrics, ECE, OOD rejection, reasons, schemas

data/
  raw/imd/                   IMD 0.25° NetCDF files (2016-2022, 178 MB)
  raw/shapes/                Census-2011 district shapefiles
  raw/wb2/hres/              Cached HRES forecasts (7 NetCDF files, 72 MB)
  raw/wb2/era5/              Cached ERA5 2D/3D fields (~450 MB)
  raw/wb2/ens/               IFS ENS 50-member subsample
  interim/                   imd_subdivisions.gpkg, observed truth, spatial weights
  processed/dataset.parquet  279,650 rows x 93 columns (The complete modeling dataframe)
  artifacts/                 bust_model.joblib, bulletins.sqlite (63 MB, 79,900 rows), evaluation metrics
```

---

## 5. Summary of Traps & Critical Gotchas (Do Not Re-Fall Into Them)

1. **WeatherBench 2 Zarr Chunks Span the Globe (D-002):** Subsetting lat/lon saves RAM and disk, *not* download bandwidth.
2. **ERA5 Store Naming Trap (D-009):** Stores named `1959-2022` end on `2021-12-31`. We use `1959-2023_01_10-6h-240x121...` which includes 2022.
3. **IMD Gridded Rainfall is Mainland-Only (D-006):** Andaman & Nicobar and Lakshadweep have 0 gauge cells. 34 subdivisions are modeled, with islands explicitly documented.
4. **Bust Definition Requires Both Percentile & Category (D-004 / LOGIC.md §4.3):** Percentile alone is circular; category alone misfires on $15.5$ vs $15.7\text{ mm}$ boundary straddles.
5. **Fair Calibration Invariant (D-010):** All baselines and the meta-model share the exact same Isotonic calibration on the 2021 validation set.
6. **Day 10 Lagged Spread Boundary (D-011):** HRES stops at 240h, so lagged spread has 1 member at Day 10. The headline evaluation is the **Day 3–7 decision band** where the baseline has full 3-member support.
7. **Native TreeSHAP vs Package Incompatibility (D-013):** XGBoost 3.x serializes `base_score` as `[5E-1]`, which crashes `shap 0.49`. We use `xgb.Booster.predict(..., pred_contribs=True)` directly.
8. **Season Concatenation Differences (D-011):** Grouping by year before computing `.diff()` prevents differencing June 1st against the previous September 30th.

---

## 6. Definition of Done & Remaining Next Steps

- [x] Canonical engineering specification written: [`LOGIC.md`](file:///c:/Users/ASUS/Desktop/sih/LOGIC.md).
- [x] Full training, validation, and test dataset processed (`data/processed/dataset.parquet`).
- [x] Model trained & calibrated (`data/artifacts/bust_model.joblib`).
- [x] SQLite bulletin store generated (`data/artifacts/bulletins.sqlite`).
- [x] Full test suite implemented & passing (21/21 tests in `tests/`).
- [x] Interactive Leaflet dashboard & FastAPI service operating offline.
- [x] Terminal replay demo for Assam & Meghalaya June 2022 floods.
- [x] **Run IFS ENS Baseline Comparison:** done. Documented as **D-014**; README
      headline table now carries the true-ENS row and the corrected +0.025 margin.
- [x] **Complete the CALIBRATED ENS comparison:** done at 15:38 IST. Training-year
      subsamples landed (7,560 rows each for 2019/2020, every 6th init). Calibrated
      table now populated in README row and in D-014 addendum. Result: on the
      6,732 decision-band rows the model beats a real, calibrated 50-member IFS
      ENS by +0.040 AUROC (0.832 vs 0.792), -16.8% decision cost (238.7 vs 287.0),
      and 2x economic value (0.289 vs 0.145). The raw-vs-raw AUROC margin
      (+0.025, 0.807 vs 0.832) is still the honest AUROC headline.
- [x] **Docker Smoke Test:** passed. `fbd:0.1.0` builds (957 MB, ~20 s export),
      container reports `(healthy)` via the built-in HEALTHCHECK, `/api/health`
      returns 200 with 79,900 rows loaded, and the dashboard renders 36
      subdivision polygons with **zero external network requests** (Leaflet 1.9.4
      served from `web/vendor/`), confirming the air-gap guarantee in LOGIC.md 13.
      Note: **`/api/bulletin/latest` is not a route** -- the path parameter is
      `{region_id}` and `init_date` is a required query param, so that URL returns
      422. Use `/api/bulletin?init_date=YYYY-MM-DD&lead_day=N`.

---

## 7. First Commands to Run in a New Session

```bash
# 1. Check test suite health
PYTHONPATH=src pytest tests/ -v

# 2. Check IFS ENS download progress
ls -lh data/raw/wb2/ens/
tail -n 20 data/raw/wb2/ens_fetch.log

# 3. Start local API and Dashboard
python -m uvicorn fbd.api.app:app --app-dir src --port 8912
# Open http://localhost:8912 in browser
```
