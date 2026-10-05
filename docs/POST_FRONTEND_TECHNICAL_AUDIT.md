# Post-frontend technical audit (ML and data)

Phase F of the master prompt, written 2026-10-05 after the frontend slice
shipped on `frontend-redesign`. It answers the prompt's fourteen questions
from the code, with file references, not from memory. Where a claim could
not be verified here, it says so.

Companion files:
- `docs/FEATURE_AVAILABILITY_MATRIX.md`: what is legal at prediction time
- `docs/REALTIME_SOURCE_MATRIX.md`: candidate live sources
- `docs/ML_RESEARCH_BACKLOG.md`: what to try next, and in what order
- `docs/CLOUD_AND_CICD_PLAN.md`: deployment and CI

---

## 1. What data is fetched, and what is only an idea?

**Fetched and used:**

| source | script | what | years |
|---|---|---|---|
| IMD 0.25° gridded daily rainfall | `scripts/fetch_imd.py`, `src/fbd/ingest/imd.py` | observed truth, 03Z→03Z days | 2016–2022 JJAS |
| WeatherBench 2 IFS HRES, 0.703° (512×256), 00 UTC (D-003) | `scripts/fetch_hres.py`, `ingest/hres.py`, `ingest/wb2.py` | the forecast being judged: `total_precipitation_24hr`, leads 1–10 | 2016–2022 |
| WeatherBench 2 ERA5 `1959-2023_01_10`, 1.5° (240×121) | `scripts/fetch_era5.py`, `features/era5.py` | 00 UTC state at issue time: u, v, z, q, tcwv, msl at 850/500/200 hPa | 2016–2022 |
| WeatherBench 2 IFS ENS, 50 members, 0.703° | `scripts/fetch_ens.py`, `ingest/ens_fetch.py` | real ensemble spread; short ensembles refused (`EXPECTED_MEMBERS = 50`) | 2019–2022 (backtest + serving); 2018 for the S3a-C confirmation |
| Census-2011 districts | `regions/` | 641 districts → 34 modelled subdivisions (D-001, D-006) | static |

**Opt-in at serving time only:** NASA GIBS MODIS imagery as a basemap
(`web/index.html`, D-023). It is not a model input.

**Ideas, not fetched:** GFS/GEFS, ECMWF AIFS, GPM IMERG, MOSDAC/INSAT, live
IMD feeds. None has an adapter. `REALTIME_SOURCE_MATRIX.md` evaluates them.
There is no live ingestion path: `mode=live` reports `STALE` by design
(`api/app.py`).

## 2. Sources and versions behind each feature

All 52 served inputs are listed in `src/fbd/model/train.py:35-82`; the served
list is `bust_model.joblib["features"]`. By group:

- **Forecast (18).** WB2 IFS HRES at 0.703°, area-mean per subdivision via
  `weights_*.parquet` (the observed side, IMD, is 0.25°). Lagged-ensemble statistics come from earlier HRES runs
  verifying the same day (`features/forecast.py:11-13`). Climatology is fitted
  on training years only (`forecast.py:105`, `:130`).
- **ERA5 subdivision fields (7) and national indices (15).** The WB2 ERA5
  00 UTC analysis (`era5.py:131`). Indices are standardised against
  training-year climatology (`era5.py:145`).
- **Regime (8).** Derived from the ERA5 fields (`regime/classify.py`).
- **Static (4).** Terrain and coast indices from the region build.

Full provenance per feature: `FEATURE_AVAILABILITY_MATRIX.md`.

## 3. What timestamps were available at prediction time?

- **Issue time is 00 UTC on T_init.** The ERA5 state is sampled at that
  analysis time, never at verification time (LOGIC.md §7.1, poison-tested in
  `tests/test_core.py`).
- **Lead L verifies on V = T_init + (L−1) days.** The WB2 accumulation window
  is [init + (L−1)·24 h, init + L·24 h] (DECISIONS.md, the D-004 area).
- **The lagged spread for lead L uses only runs issued at or before T_init**
  (`forecast.py:11-13`).
- **Known offset (D-005).** IMD days run 03Z→03Z while WB2 accumulations run
  00Z→00Z. The 3-hour offset is fixed, accepted and applies equally to every
  baseline.
- **Open item: the ERA5 analysis window.** ERA5 is a reanalysis. Its 00 UTC
  analysis may assimilate observations from after 00 UTC within the 4D-Var
  window, and operationally it is published days later. A live system cannot
  use ERA5; it would use the operational analysis.
  - Status: **unknown until verified.** The exact window should be checked
    against ECMWF's ERA5 documentation.
  - The robust fix is in the backlog (B2): rebuild the ERA5 features from
    WB2's `hres_t0` (operational initial conditions) and measure the change.

## 4. How the label is built, and how the train-only threshold is enforced

`src/fbd/labels/bust.py`. The three conditions:

```
bust = |F − O| ≥ max(10 mm, P95_train(subdivision, month))
       AND category(F) ≠ category(O)
       AND max(F, O) ≥ 15.6 mm/day
```

- **Thresholds.** `error_thresholds()` (`bust.py:57-80`) filters to
  `valid_date.year ∈ train_years` before computing any percentile.
  `label()` defaults `train_years` to `config.TRAIN_YEARS` (2016–2020;
  `bust.py:100`).
- **Thin cells back off.** A subdivision-month with fewer than 200 training
  rows uses the subdivision-wide P95, then the global P95.
- **Backtest folds** pass their own training years, so each fold's labels
  carry no test-year information.

## 5. Which model artifacts are loaded, and where?

| stage | loads | file |
|---|---|---|
| training | `dataset.parquet` → writes `bust_model.joblib` (`features`, `booster`, `calibrator`, `prior`, `params`) | `scripts/train_model.py`, `model/train.py` |
| combiner fit | `bust_model.joblib` + ENS spread (2021) → `combiner.json` | `scripts/fit_combiner.py`, `model/combined.py` |
| batch inference | model + combiner + OOD detector + bagged models → `bulletins.sqlite` | `scripts/generate_bulletins.py` |
| serving | **only** `bulletins.sqlite`, `metrics` JSON artifacts, `regions.geojson`, `centroids.json`, `voxel_grid.json`; no model runs at request time | `api/app.py` |
| registered studies | frozen fold models `data/processed/backtest/model_{year}_{mode}.joblib`, candidate outputs `data/artifacts/candidates/*.json` | `scripts/backtest.py`, `scripts/promote.py` |

## 6. `model_probability`, `ens_spread`, `baseline_probability`, `bust_probability`

| field | what it is |
|---|---|
| `model_probability` | XGBoost + isotonic probability from the 52 inputs alone |
| `ens_spread` (`ens_spread_mm` in the API) | 50-member IFS ENS spread for that subdivision and lead, in mm/day |
| `baseline_probability` | the ENS-spread baseline: spread alone, mapped to a probability with the same isotonic calibration on the same validation year (D-010). The honest comparator, shown in the map's Baseline view. |
| `bust_probability` | **The served number.** A two-input logistic on `[logit(uncalibrated model prob), log1p(ENS spread)]`, fitted once on 2021 (`combiner.json`, `model/combined.py:3-6`, D-029/D-030). `NULL` whenever `status != OK`. |

The UI shows `model_probability` and `ens_spread` only as labelled
ingredients beside the served number, never instead of it.

## 7. Calibration data and method

- **Incumbent:** isotonic regression fitted on the validation year 2021
  (`config.VAL_YEARS`, D-008).
- **Baselines:** the same isotonic fit on the same year (D-010).
- **Served combination:** calibrated by its own logistic fit on 2021.
- **Known weakness:** overconfidence in the extreme tail. In 2022 the top
  equal-count bin (1,988 rows) predicts 0.207 and observes 0.172, so the
  served combination is overconfident by 0.036 (the model alone by 0.049;
  D-030). Calibration error is reported with an interval on the landing
  register.

## 8. The OOD rule, and how REFUSE is represented

- **Detector:** `src/fbd/ood/detector.py`, a Mahalanobis distance on
  standardised training features. Covariance shrinkage is `1e-3·I` and the
  inverse is taken with `pinv`. The threshold is the **99.5th percentile**
  of training distances (`quantile=0.995`, `detector.py:29-52`). A row is
  OOD when `distance > threshold`.
- **In the store:** `status = OUT_OF_DISTRIBUTION`, `bust_probability =
  NULL`, `ood_distance` kept; `meta.ood_threshold` is recorded.
- **In the API:** `review_tier = REFUSE`. `classify_tier` maps any missing
  probability to REFUSE, never AUTO_OK (`quality/escalation.py:52-66`).
- **In the UI:** a hatch or stripe, never a hue. The map has
  `OOD_FILL = "url(#oodHatch)"`, the columns have `refusalTexture()`, and
  the landing has ghost plates. All are test-pinned.
- **Measured:** refused days bust 23.4% vs 3.4% for scored days in 2022
  (`/api/metrics.refusal`).

## 9. Folds and promotion rules

- **Folds** (`src/fbd/evaluate/folds.py:23-28`): expanding window.
  - 2019: train 2016–17, validate 2018
  - 2020: train 2016–18, validate 2019
  - 2021: train 2016–19, validate 2020
  - 2022: train 2016–20, validate 2021
- **Primary years:** `PRIMARY_YEARS = (2019, 2020, 2021)`. 2022 is reported
  beside the primary, never in it.
- **Confirmation fold:** `CONFIRM_FOLDS = {2018: train 2016, validate 2017}`.
- **Modes:** `legacy` (the published pipeline) and `strict` (regime
  statistics fitted on training years only).
- **S3 promotion rule** (`evaluate/promotion.py`, registered in
  `docs/PREREGISTRATION_S3.md`):
  - slate: `mlp`, `temporal`, `spatial`
  - measure: mean within-year AUROC margin over the XGBoost fold models,
    2019–2022
  - level: Bonferroni α = 0.05/3, i.e. a 98.33% interval
  - verdict: `promoted` if lo > 0, `incumbent_better` if hi < 0, otherwise
    `indistinguishable`
- **Registration gate:** `evaluate/registration.py` and `provenance.py`
  refuse to score unless the registration file is committed unchanged and
  the pinned hashes match.

## 10. Studies already performed

| study | decision | result | served? |
|---|---|---|---|
| S1 | D-025 | model ahead of real ENS spread over 2022 | n/a |
| S1b | D-026 | not distinguishable from ENS spread 2019–2021 (ENS wins 2019) | n/a |
| S1c | D-029 | model + ENS outranks ENS alone on average 2019–2021 | **yes**: D-030, store v0.2.0 |
| S3a (MLP) | D-027 | **promoted under the S3 rule**: outranks XGBoost on mean 2019–2022, clearly better in some years, not 2022 | **no**: "promotion into the served product is decided after S3c" |
| S3a-C | D-028 | MLP's lead over ENS spread not confirmed in untouched 2018 | n/a |
| S3b (GRU, 14-day history) | D-031 (branch `s3b-temporal`, PR #3, CI green, not merged) | not distinguishable; history adds nothing | no |

**Correction.** The master prompt's summary says the MLP "did not displace
the incumbent". The precise record: the MLP *was* promoted under the
registered rule; serving it was deferred until after S3c.

## 11. Registered or specified, not yet scored

**S3c, the spatial CNNs.** The slate entry `spatial` is declared in
`PREREGISTRATION_S3.md`. The design spec is
`docs/superpowers/specs/2026-09-28-s3c-spatial-candidate-design.md`
(branch `s3c-spatial`, PR #6, stacked on #3, awaiting your review). It still
needs `PREREGISTRATION_S3C.md` with `spatial_params_sha256` through the
addendum mechanism, the implementation, and one registered score. Nothing
has been scored.

## 12. Endpoints and store fields that expose the output

See `docs/PROJECT_GUIDE.md` §6.1–6.2: 14 endpoints and the `bulletins`,
`meta` and `overrides` tables. `BustPrediction` (`api/schema.py`) is the
single object every page renders from.

## 13. Constraints that must stay true

- **Serving:** one process with no model at request time; precomputed
  SQLite; static pages; no build step.
- **Offline:** no external origins except opt-in dated imagery (CI scan +
  `tests/test_web_pages.py`); system fonts only.
- **The served number:** it is the combination; ingredients are labelled
  beside it.
- **Honesty:** refusal ≠ low risk; null stays null; claims are read from
  `/api/metrics`.
- **Analysis:** register before scoring; never touch the test year to
  choose anything.
- **Public mode:** read-only with `FBD_READ_ONLY=1` (override → 403);
  GenAI off by default.

## 14. Smallest high-value improvements still unimplemented

In priority order (details and acceptance criteria in `ML_RESEARCH_BACKLOG.md`):

1. **S3c to its registered score.** It completes the declared slate and
   unblocks the MLP serving decision.
2. **Operational-analysis substitution (B2).** Rebuild the ERA5 features
   from WB2 `hres_t0` and measure the AUROC change. This proves the model
   works on data a live system would actually have.
3. **Tail calibration.** Register a recalibration comparison (isotonic vs
   beta vs Venn-Abers) against the held-out tail overconfidence.
4. **Cost-ratio sensitivity in the UI.** REVIEW_THRESHOLD is derived from
   10:1, so show how the queue changes at 5:1 and 20:1. This is presentation
   only, with no new claim.
5. **Live-path skeleton without the Internet.** Add the provider protocol
   and the ingestion state machine (master prompt §11.3–11.5) with a
   fixture-backed fake provider and tests, so a live source can be added
   later without touching serving.
