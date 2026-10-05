# PROJECT GUIDE — learn the whole system, then extend it

One self-contained guide to Forecast Bust Detection: what it is, how every
layer fits together (data → model → store → API → 2-D/3-D UI), how to run and
test it, the discipline behind its results, and concrete recipes for adding
more. Written to be the single file you read to come up to speed and keep open
while you build.

**Canonical sources it summarises** (go to these for the exact, current
numbers — this guide is a map, those are the ground truth):
`LOGIC.md` (the locked engineering contract), `DECISIONS.md` (every decision
D-001…D-031 with evidence), `DATA.md` (data policy), `FRONTEND_LOGIC.md` (what
the UI may claim), `docs/DEPLOY.md` (hosting). Registered intervals move when a
store is regenerated; where this guide quotes one it is a snapshot and names
the decision record that owns it.

---

## 1. What the system is

It predicts **when an existing medium-range rainfall forecast over India is
about to fail** — which subdivision, which lead day, and why — instead of
predicting the weather itself. It is **decision support**: it never issues or
suppresses a public warning; a human forecaster is always the authority. The
product is a ranked *review queue* of region-days for a duty forecaster to
check twice, with a map and 3-D views over the same data.

### What a "bust" is
A large error that also flips the rainfall category across an
operationally-significant threshold — an error that would have changed the
alert decision. For subdivision *s*, lead *L*, forecast `F`, IMD-observed `O`
(area-mean):

```
bust = |F − O| ≥ max(10 mm, P95(s, month))     # magnitude
       AND category(F) ≠ category(O)            # the decision flips
       AND max(F, O) ≥ 15.6 mm/day              # significant rain
```

Thresholds (`P95`) are fitted on each fold's **training years only**, so the
label carries no test-year information.

---

## 2. The whole pipeline, end to end

```
 RAW (fetched, not in git)        DERIVED (in git)         SERVED
 ─────────────────────────        ────────────────         ──────
 IMD 0.25° rainfall  ─┐
 WB2 HRES forecasts   ├─ build_truth ─▶ truth parquet ─┐
 WB2 ERA5 analysis    ┤  build_dataset ─▶ dataset.parquet (279,650 × 93)
 WB2 IFS ENS (50)    ─┘        │                        │
                               ▼                        ▼
                        train_model ─▶ bust_model.joblib (XGBoost + isotonic)
                        fit_combiner ─▶ combiner.json  (model + ENS logistic)
                               │                        │
                               ▼                        ▼
                     generate_bulletins ─▶ bulletins.sqlite (79,900 rows)  ◀── the product serves THIS
                               ▲                        │
 regions.build ─▶ precompute_geo_assets ─▶ regions.geojson, centroids.json
 weights_*.parquet ─▶ precompute_voxel_grid ─▶ voxel_grid.json
                                                        │
                                                        ▼
                                   FastAPI (one process) ── GET /api/* + 4 static pages
```

**Nothing runs the model at request time.** The daily batch scores everything
offline; serving is a static-ish read path over SQLite. That is the constraint
the whole design bends around — it is what makes the fully-offline, air-gapped
demo possible (LOGIC.md §11/§14).

Execution order (all under `scripts/`, `PYTHONPATH=src`):
`fetch_imd/hres/era5/ens` → `build_truth` → `build_dataset` → `train_model` →
`fit_combiner` → `generate_bulletins` → `precompute_geo_assets` →
`precompute_voxel_grid`. Evidence/eval: `ablation`, `stress_test`,
`evaluate_ens_baseline`, `compute_confidence_intervals`, `monitor_drift`,
`plot_*`.

---

## 3. The data

Three tiers (full policy in `DATA.md`):

| tier | examples | where |
|---|---|---|
| **In git** (small derived) | `dataset.parquet` build inputs, `weights_*.parquet`, `*.json` artifacts, `regions.geojson`, `centroids.json`, `voxel_grid.json` | the repo |
| **Release asset** (~74 MB) | `bulletins.sqlite` | GitHub Release, checksum-pinned by `fetch_release_artifacts.py` |
| **Fetched, never redistributed** (~570 MB raw) | IMD rainfall, WB2 HRES/ERA5/IFS-ENS | fetched from primary sources by `fetch_*.py` |

Sources: IMD 0.25° gridded rainfall (2016–2022); WeatherBench 2 IFS HRES
forecasts, ERA5 analysis, and 50-member IFS ENS (anonymous GCS). Seasons are
JJAS (1 Jun – 30 Sep) only.

Spatial unit: **34 modelled IMD subdivisions** (36 defined; Andaman & Nicobar
and Lakshadweep drop out as the gauge product is mainland-only, D-006), built
as unions of 641 Census-2011 districts (D-001). Grid overlaps are computed
once in EPSG:7755 and cached (`weights_*.parquet`); the feature pipeline *and*
the 3-D voxel grid both reuse that exact overlap, so the rendered volume sits
on the model's own grid.

---

## 4. The modelling core

### Features (52 served inputs, from a 93-column dataset)
Grouped: the forecast itself (`fcst_rain_mm`, anomaly vs seasonal normal,
vs P90), the **lagged-ensemble proxy** (spread/mean/range/jumpiness —
structurally undefined at Day 10, D-011), climatology (per subdivision/month,
and the climatological bust rate), ERA5 **national indices** (Somali jet,
monsoon-trough MSLP, shear, TCWV, q850, BoB vorticity… as standardised `*_z`
and 1-/3-day tendencies `*_d1/_d3`), ERA5 **subdivision fields** (moisture
flux, winds, z500, MSLP…), the soft **regime vector** (active/break monsoon,
depression, western disturbance, orographic, coastal + entropy — kept for
explanation, adds no skill, D-012), and static geography (elevation, coastal
index…). Full list: `BustModel.features`.

Causality (LOGIC.md §7.1, poison-tested in `tests/test_core.py`): ERA5 state is
sampled at **issue time** `T_init`, never the verification day; the lagged
spread for lead *L* uses only runs issued at or before `T_init`.

### Model and the served number
- **Incumbent:** XGBoost gradient-boosted trees + **isotonic calibration** on
  the validation year (D-008), explained by native XGBoost **TreeSHAP** (D-013),
  guarded by a **Mahalanobis OOD detector** (refuses states unlike training).
- **Baselines**, all given the *same* isotonic calibration on the *same*
  validation year (D-010): climatology, forecast-amount-only, logistic
  (spread+lead), and **real 50-member IFS ENS spread** (the honest comparator).
- **The served `bust_probability` is the combination** (D-029/D-030): a
  two-input logistic regression on `[logit(uncalibrated model prob), log1p(ENS
  spread)]`, fitted once on 2021 (`combiner.json`). `model_probability` and
  `ens_spread` are stored and shown *beside* it, never instead of it.

### Uncertainty & tiers
Every number ships with a **bagged prediction interval** (`pi_low/high`,
`confidence_in_estimate`) and a **review tier** (`fbd/quality/escalation.py`):
`AUTO_OK` / `REVIEW` / `REFUSE`. The REVIEW threshold is not free — it is the
cost-optimal `1/(1+10)=0.0909` under the 10:1 miss:false-alarm ratio. **REFUSE
means unknown risk, not low risk** (refused days bust ~7× more often), which is
why there are three tiers and not two.

---

## 5. The registered-analysis discipline (how results are earned here)

Every comparative claim is **pre-registered before the data is scored**. A
registration markdown (`docs/PREREGISTRATION_*.md`) carries a fenced
````registration```` block of frozen hashes; the scoring script refuses to run
unless that file is committed unchanged and the pinned inputs still match
(`fbd/evaluate/registration.py`, `provenance.py`). This is what keeps the
headline numbers honest, and it is the pattern you follow to add a new result.

The study arc so far (each is a decision record):

| id | question | verdict |
|---|---|---|
| S1 (D-025) | model vs real ENS spread, 2022 season | model ahead, +0.0316 AUROC |
| S1b (D-026) | does it hold 2019–2021? | not distinguishable (ENS wins 2019) |
| S1c (D-029) | model **+** ENS vs ENS alone | combination ahead, +0.0244 |
| D-030 | serve the combination (store v0.2.0) | now the served number |
| S3a (D-027) | an MLP on the same 52 inputs vs XGBoost | MLP ahead on average, not in 2022 |
| S3a-C (D-028) | confirm MLP's ENS lead on untouched 2018 | not confirmed |
| S3b (D-031) | + a GRU over 14 days of history | not distinguishable; history adds nothing |
| **S3c** (in review) | + CNNs on the forecast window & synoptic map | **spec written, not yet scored** |

The S3 slate (`mlp`, `temporal`, `spatial`) was declared in full up front with
one Bonferroni-corrected promotion rule (`docs/PREREGISTRATION_S3.md`); each
candidate is scored by `scripts/promote.py` against the frozen XGBoost fold
models and read against the MLP (so a gain is attributed to *information*, not
to being a neural net). `promote.py` already supports addenda and the MLP
secondary, so S3c reuses the machinery.

---

## 6. The serving architecture

One FastAPI process (`src/fbd/api/app.py`) serves precomputed SQLite + four
static pages. Vanilla JS, **no build step**, two vendored libs (Leaflet,
Three.js r128). CORS open. Each endpoint opens a short-lived SQLite connection;
a missing store is a 503 the page renders as an empty state. Two **clock
modes**: `mode=replay` (the requested date *is* "now") and `mode=live`
(compares to the wall clock; the archive build correctly reports `STALE`).

### 6.1 The store (`bulletins.sqlite`, 79,900 rows)

`bulletins` — one row per `(region_id, init_date, lead_day)`:

| column | meaning |
|---|---|
| `region_id`, `region` | `ASSAM_MEGHALAYA`, "Assam & Meghalaya" |
| `init_date`, `lead_day` (1–10), `valid_date` | issue day, lead, forecast day |
| `status` | `OK` / `OUT_OF_DISTRIBUTION` / `CLIMATOLOGY_FALLBACK` / `UNAVAILABLE` |
| `bust_probability` | **served number** = model + ENS combination; `NULL` if not `OK` |
| `model_probability`, `ens_spread` | the two ingredients (v0.2.0) |
| `confidence_in_estimate`, `pi_low`, `pi_high` | bagged interval |
| `dominant_factors` (JSON) | reason strings (ENS line first, then TreeSHAP) |
| `regime_json` (JSON) | soft regime vector |
| `ood_distance` | Mahalanobis distance (drives the refusal) |
| `forecast_rain_mm`, `observed_rain_mm`, `actual_bust` | verification log |
| `baseline_probability` | calibrated ENS-spread baseline (model-vs-baseline toggle) |
| `model_version` | `0.2.0` |

Also `meta` (key/value: `model_version`, `combined`, `combiner`,
`drift_status`, `ood_threshold`, …) and `overrides` (immutable audit log,
never dropped on regeneration).

### 6.2 API reference

| method + path | params | returns | used by |
|---|---|---|---|
| `GET /api/health` | — | `HealthResponse` (status, version, drift, notes) | ops |
| `GET /api/replay/dates` | — | `{init_dates: string[]}` | every page |
| `GET /api/regions` | — | GeoJSON subdivision polygons | index, command |
| `GET /api/voxel-grid` | — | packed region-index grid + outline (§6.4) | volume |
| `GET /api/bulletin` | `init_date, lead_day?, mode?` | `Bulletin{predictions: BustPrediction[]}` | index |
| `GET /api/bulletin/{region_id}` | `init_date, mode?` | `BustPrediction[]` (all leads) | command |
| `GET /api/convergence` | `region_id, valid_date, mode?` | `BustPrediction[]`, longest lead first | landing |
| `GET /api/review-queue` | `init_date, top?=20, decision_band_only?=true` | `ReviewQueueItem[]` | index, command, volume |
| `GET /api/verification` | `init_date, lead_day` | `{records, n_busts_observed}` | demo |
| `GET /api/metrics` | — | held-out metrics + CIs + settlement + backtest + combination + served + refusal + test_year | landing |
| `GET /api/risk-cube` | `init_date, mode?` | space × lead field (§6.4) | command, volume |
| `POST /api/override` | `OverrideRequest` | `{ok, n_overrides, note}` (403 if `FBD_READ_ONLY=1`) | index |
| `GET /api/overrides` | `limit?=100` | `{overrides}` | index |
| `GET /metrics` | — | Prometheus text (telemetry, **not** the model table) | monitoring |

GenAI routes (`/api/assistant/*`) mount only when `FBD_GENAI_ENABLED=1`
(default off; a failed mount never blocks serving, D-015/D-019).

**`BustPrediction`** is the central object every cell/row/panel is built from
(`src/fbd/api/schema.py`). Never a bare number — it carries `status`,
`bust_probability` (served) + `model_probability` + `ens_spread_mm`,
`confidence_in_estimate` + `prediction_interval`, `dominant_factors`,
`regime`, `review_tier` + `tier_guidance`, `data_quality` + `input_age_hours`
+ `ood_distance`, `baseline_probability`, the verification fields, and
`model_version`. Example (Assam, 2022-06-14, Day 4): served `0.581`, model
`0.706`, ENS `11.7`, baseline `0.112`, tier `REVIEW`, `actual_bust 1`.

### 6.3 Frontend conventions (all pages)
- No build step; edit HTML + reload. Libs vendored in `web/vendor/`
  (`leaflet.{js,css}`, `three.min.js` = **r128**).
- **Air-gap:** pages add no external origin (no CDN/font/tiles). New dep →
  vendor it. Guarded by `tests/test_web_pages.py` and the CI offline job.
- **Colour for risk:** viridis-like ramp; refusals get a hatch (2-D) or stripe
  (3-D) — never a cool/green colour for a refusal.
- **Claims gate:** headline numbers come from `/api/metrics`, gated on
  `served.*`; never hardcode them (`FRONTEND_LOGIC.md` §8).
- Light/dark theme persisted in `localStorage`. No SIH/competition branding.

### 6.4 Geometry & the 3-D data
- **`centroids.json`** — `{region_id: [lon, lat]}`, a *representative interior
  point* per subdivision (not the centroid — concavity would float Konkan &
  Goa's column out to sea). Places 3-D columns and risk-cube lon/lat.
- **`voxel_grid.json`** (`/api/voxel-grid`) — region-index raster on the 0.25°
  model grid: `ny, nx, lat0, lon0, dlat, dlon, nodata=255, region_ids[≤36],
  ids_b64 (base64 uint8), outline[47 polylines]`. Largest-area wins each cell;
  **boundary cells never blended** (D-024). `outline` is a finer second
  geography drawn under the volume for orientation; its mismatch is bounded and
  labelled, not hidden.
- **`/api/risk-cube`** — the space × lead field in one lean payload:
  `{init_date, lead_days[10], decision_band[5], n_regions, data_quality,
  input_age_hours, banner, regions:[{region_id, region, lon, lat, p[10],
  baseline[10], status[10], actual[10], fcst_mm[10], obs_mm[10],
  valid_date[10]}]}`. Probabilities as parallel arrays; reasons fetched per
  region on click.

### 6.5 Page: `index.html` — the 2-D map (the product)
Leaflet choropleth + review queue + override UI. `boot()` → `/api/regions` +
`/api/replay/dates`; `load(init)` → `/api/bulletin` → colour each polygon by
`bust_probability` at the chosen lead. Functions to know: `colour(p,status)`,
`installHatch()` (OOD), `setLead`/`stepDate`/`toggleScrub` (play through dates),
`setView("model"|"baseline")`, imagery overlay (opt-in, dated — the only
external origin, D-023), the override bar (`POST /api/override`→`loadOverrides`),
`toggleTheme`, `bindKeys` (m/b/arrows).

### 6.6 Page: `command.html` — the 3-D column "command centre"
A risk column per subdivision (geography on x/z, lead day up y, colour = prob).
Three.js r128, `WebGLRenderer`, **hand-rolled orbit controls** and
`mergeGeometries` (r128 ships `OrbitControls`/`BufferGeometryUtils` only under
`examples/`). Scene = `mapGroup` (projected outlines + ground), `colGroup`
(columns), `bandGroup` (the Day 3–7 slab), + `Raycaster`. `proj(lon,lat)->[x,z]`;
polygons → `THREE.Shape`→`ShapeGeometry`. Builders: `buildMap`, `buildBand`,
`buildColumns` (refused segments get `refusalTexture()`), `colourFor(p,status)`.
`pick`→`select(rid,lead)`→`/api/bulletin/{rid}`; `fillQueue` ←`/api/review-queue`;
`loadCube`←`/api/risk-cube`. Loop: `animate`.

### 6.7 Page: `volume.html` — the raymarched volume
The data is genuinely 3-D (lon × lat × lead); columns discretise lead, this
**raymarches a 3-D texture** so risk accumulates along lead. **Requires
WebGL2** (`sampler3D`/GLSL3; fails loudly otherwise). Three.js r128,
`RawShaderMaterial` + `DataTexture3D`.
- **Field packing:** `uField` `sampler3D`, size `uSize=(nx, ny=lat, nz=lead)`,
  RGBA uint8 = **R prob, G status/flags, B region index, A edge bits** (/255).
  Voxel grid gives region-per-cell; risk-cube gives prob-per-(region,lead).
- **Display shader** raymarches front-to-back with a viridis transfer function.
  Tunable uniforms (your main control surface): `uDensity`, `uFloor`,
  `uThreshold`+`uShowIso`, `uShowRefused`, `uAmbient`, `uSlice`(0=all,1..nz=one
  lead)+`uSliceDim`, `uHover`+`uFocus`, `uTime`.
- **Pick shader** renders to an offscreen target encoding `R region, G lead,
  B prob, A hit` so a click reads voxel identity off the GPU.
- **Handedness:** north is **−z**; `layerTop(lead,nz)` is a layer's top-face y.
- **Fly-through** (`Space`/`Esc`): a script over the *same* view state; targets
  come from `/api/review-queue?top=1`, captions from risk-cube rows — never a
  scripted lie. Any direct input takes control back.

---

## 7. Repository map

```
src/fbd/
  api/        app.py (FastAPI, all routes), schema.py (Pydantic), genai_routes.py
  model/      train.py (BustModel=XGBoost+isotonic), baselines.py, mlp.py, temporal.py,
              combined.py (served combiner apply), params.py (frozen hyperparams + hashes)
  evaluate/   folds.py, promotion.py, registration.py, provenance.py, backtest_stats.py,
              uncertainty.py, ens.py, combine.py, metrics.py, settle.py
  features/   forecast.py, era5.py, standardise.py   | labels/bust.py
  regions/    build.py (subdivisions), masks.py (grid overlap weights)
  quality/    escalation.py (tiers), drift.py        | explain/reasons.py (SHAP→text)
  ood/        detector.py (Mahalanobis)              | obs/metrics.py (Prometheus)
  genai/      agent.py, client.py, settings.py, tools.py (default-OFF assistant)
web/          index.html, landing.html, command.html, volume.html, vendor/
scripts/      fetch_*, build_*, train_model, fit_combiner, generate_bulletins,
              precompute_geo_assets, precompute_voxel_grid, promote, audit_*, plot_*, monitor_drift
docs/         DEPLOY.md, FIGURES.md, FRONTEND_BUILT.md, PROJECT_GUIDE.md (this),
              PREREGISTRATION_*.md, superpowers/specs|plans/
tests/        26 files (see §9)
data/         raw/ (fetched, gitignored), interim/ & processed/ (derived), artifacts/
```

Top-level docs: `README.md`, `LOGIC.md`, `DECISIONS.md`, `HANDOFF.md`,
`DATA.md`, `FRONTEND_LOGIC.md`, `FRONTEND_HANDOFF.md`, `HANDOFF_VOLUME.md`,
`PROJECT_BLUEPRINT.md`, `TECHNICAL_STUDY_GUIDE.md`.

---

## 8. Run, test, deploy

```bash
# serve (offline defaults) — pages at http://localhost:8912/
python -m uvicorn fbd.api.app:app --app-dir src --port 8912

# get the store (or regenerate it)
python scripts/fetch_release_artifacts.py            # downloads + checksum-verifies
PYTHONPATH=src python scripts/generate_bulletins.py  # regenerate from committed artifacts + ENS

# full test suite
PYTHONPATH=src pytest tests/ -q

# container (read-only public mode)
docker run -p 8912:8912 -e FBD_READ_ONLY=1 ghcr.io/blazingarrows1525/forecast-bust-detection:latest
```

**Preview tip (Claude Code):** add `.claude/launch.json` with an entry
`{name:"fbd", runtimeExecutable:"python", runtimeArgs:["-m","uvicorn",
"fbd.api.app:app","--app-dir","src","--port","8912"], port:8912}` so
`preview_start {name:"fbd"}` works.

**Deploy:** the image auto-publishes to GHCR on every push to `master`. Free
public host = **Render** (web service, Existing Image, Free, `PORT=8912`,
`FBD_READ_ONLY=1`, health check `/api/health`) — full steps in
`docs/DEPLOY.md` §3.

---

## 9. Tests & CI (26 test files)

CI (`.github/workflows/ci.yml`) runs five jobs: **Unit tests**,
**Decision-log invariants** (published numbers match the artifacts),
**Offline serving path makes no network calls**, **Supply chain and static
analysis** (pip-audit, bandit, secret scan), **Docker image builds and serves**
(boot + smoke test, including read-only / non-root). Torch tests skip on CI.

Notable tests: `test_core.py` (causality poison tests), `test_fold_leakage.py`,
`test_published_numbers.py` (docs ↔ artifacts), `test_web_pages.py` (air-gap,
no-SIH, claims gate), `test_voxel_grid.py` (geometry agreement),
`test_promotion.py` (S3 rule), `test_combined*.py` (the served combination),
`test_temporal_*` (S3b).

---

## 10. Extending each layer — concrete recipes

**Add a field to every cell/row (end-to-end):** (1) `generate_bulletins.py` —
compute it, add to `SCHEMA` + the `INSERT` tuple, regenerate; (2) `schema.py` —
add it to `BustPrediction`/`ReviewQueueItem` as optional; (3) `app.py` — read
in `_to_prediction` via `_opt(r,"col")`, and in the queue/risk-cube builders;
(4) the page — consume it; (5) extend `tests/conftest.py` + `test_combined_api.py`.

**Add an API endpoint:** a `@app.get("/api/...")` **above** the `StaticFiles`
mount (the `/` mount swallows anything below it); return a Pydantic model;
add a test.

**Add a model/candidate (rigorously):** declare it in the slate, write a
`PREREGISTRATION_*.md` with frozen hashes, add an audit that proves any harness
change neutral, commit + push + green CI, then score **once** with
`promote.py`. Follow S3b/S3c as the template; `promote.py` already has the
addenda + MLP-secondary hooks.

**Add a 2-D map layer:** a new `setView` branch painting a different field from
the same `/api/bulletin` rows; a new key in `bindKeys`.

**Add a 3-D visual (command):** a builder that reads `cube.regions[i].<field>`
and adds meshes to a new `Group`; register it with the raycaster if clickable.

**Add a 3-D channel / transfer function (volume):** pack a value into a texture
channel (or add a second `sampler3D`), add a uniform + a shader branch, and a
UI control that sets the uniform. Edit the viridis polynomial for a new ramp.

**Time animation (either 3-D page):** loop over `/api/replay/dates`, re-pull
`/api/risk-cube` per date, tween the field texture / column heights (mirror
`index.html`'s `toggleScrub`).

**Mobile/touch (3-D):** orbit is pointer-event based (1-finger drag works); add
pinch-zoom by handling two-pointer `pointermove` in `bindControls`.

**Ideas worth building** (from the blueprint's remaining list): multi-model AI
disagreement features (GraphCast/Pangu/GenCast) as a candidate; a PDF bulletin
export; a cost-ratio slider that recomputes the tier threshold live;
map uncertainty (hatch OOD, draw the 90% interval); retrospective case studies.

---

## 11. Invariants you must not break

- **Offline / air-gap.** No page adds an external origin (except opt-in dated
  imagery, D-023). The serving image carries **no geo stack** (geopandas
  importable in the image is a CI failure, D-020).
- **The claims gate.** UI headline numbers come from `/api/metrics`, gated on
  `served.*`. Copy must match the evidence (`FRONTEND_LOGIC.md` §8).
- **Refused ≠ safe.** `OUT_OF_DISTRIBUTION`/`REFUSE` reads as *unknown,
  elevated* risk; three tiers exist for exactly this.
- **No blending at boundaries.** A boundary cell takes one subdivision's value.
- **The served number is the combination** (model + ENS, D-030); ingredients
  shown beside it, not instead.
- **Registered before scored.** New comparative results follow the
  pre-registration discipline; `promote.py`/the guards enforce it.
- **No SIH / competition branding** on any product surface.
- **Read-only public deploy:** `FBD_READ_ONLY=1` → `POST /api/override` = 403.

---

## 12. Gotchas

- **Three.js is r128** — `OrbitControls`/`BufferGeometryUtils` are
  examples-only and hand-reimplemented; revisit if you upgrade.
- **`volume.html` needs WebGL2** (`sampler3D`/GLSL3); it degrades to a clear
  error, not a blank canvas.
- **North is −z** in both 3-D pages — a sign flip mirrors India.
- **Centroids are representative points**, not geometric centroids.
- **The static mount is last and at `/`** — register new routes before it.
- **Per-request SQLite**; no global connection. Missing store → 503 → empty state.
- **`data_quality` is recomputed per request** (`_quality`), not read from the
  store column, so clock mode works.
- **On Windows, import torch before scikit-learn** (DLL load order, WinError
  1114) — the model entry points already do.

---

## 13. Glossary

**Bust** forecast failure that flips the alert decision (§1). **Decision band**
leads 3–7, where a forecast is trusted but sometimes shouldn't be. **OOD**
out-of-distribution (Mahalanobis) → the system refuses to score. **Tier**
AUTO_OK/REVIEW/REFUSE. **The combination** served number = model + ENS spread
(logistic). **ENS spread** disagreement across the 50-member IFS ensemble.
**Lagged proxy** a cheap spread baseline from successive HRES runs (undefined at
Day 10). **Registration** a committed, hash-pinned pre-analysis the scoring
script verifies before running. **Incumbent** the frozen XGBoost fold models
every S3 candidate is scored against. **JJAS** the Jun–Sep monsoon season.
**Replay vs live** clock modes: requested date is "now" vs compared to the wall
clock (archive ⇒ STALE).
```
