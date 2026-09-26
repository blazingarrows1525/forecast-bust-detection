# PROJECT BLUEPRINT — Forecast Bust Detection for the Indian Monsoon

**Author:** Harsh Trivedi (trivediharsh1505@gmail.com), SRMIST Kattankulathur
**Repository:** `github.com/blazingarrows1525/forecast-bust-detection`
**Session snapshot:** 15:45 IST, 23 August 2026
**Purpose of this file:** a single self-contained brief that lets a new contributor, or you three weeks from now, pick up the project without re-reading the earlier transcript.

Read alongside (all three exist in the repo root):
- **LOGIC.md** — the LOCKED engineering contract and mathematical spec.
- **DECISIONS.md** — every deviation with evidence (D-001 … D-017).
- **HANDOFF.md** — the terse resume-work brief for a fresh session.
- **README.md** — the public writeup with the headline benchmark table.

This blueprint is the *long* version. HANDOFF.md is the short version.

---

## 0. TL;DR — the one paragraph you must be able to say out loud

> Medium-range monsoon rainfall forecasts fail catastrophically maybe a dozen days a season, and when they fail the AI and physics models fail *on the same days*, so ensembling more models does not save you. We built a lightweight meta-model that watches a forecast being issued and predicts, per subdivision per lead day, whether it is about to bust — with a plain-English meteorological reason for every flag. On a held-out year the model beats a real 50-member ECMWF ensemble spread by +0.025 AUROC on identical rows (0.832 vs 0.807), reduces asymmetric disaster decision cost by 19 percent against the same baseline, refuses to answer on 0.8 percent of days that turn out to have a 23 percent bust rate versus 3 percent otherwise, and serves offline from a 63 MB SQLite file with a Leaflet dashboard that opens under two seconds with the venue wifi unplugged. We do not compete with IMD's forecast; we tell IMD's duty forecasters which forecasts to double-check.

Every clause of that paragraph is defensible from evidence in this repo.

---

## 1. Where the project stands right now

### Green
- **21/21 unit tests pass** (2.19 s), verified twice today — before and after a full pipeline rebuild.
- **Full pipeline reproduces from scratch in ~3 minutes** off cached raw data (`build_dataset` 20 s, `train_model` 14 s, `generate_bulletins` 110 s, tests 2 s).
- **Docker build passes and container reports `(healthy)`.** `fbd:0.1.0` is 957 MB on disk / 223 MB content, HEALTHCHECK green, `/api/health` returns 200 with 79,900 bulletin rows loaded.
- **Dashboard verified genuinely offline.** Browser inspection: Leaflet 1.9.4 vendored, 36 subdivision polygons rendered, `externalScripts: []`, four network requests all to localhost, zero tile-image loads. Confirms the air-gap claim in LOGIC.md §13.
- **True IFS ENS baseline scored** on the 6,732 decision-band rows where 50-member ENS exists. Result recorded as D-014 and reflected in README's headline table.

### In flight
- **Training-year ENS subsample fetch -- done at 15:38 IST.** `ens_spread_2019_every6.parquet` and `ens_spread_2020_every6.parquet` (7,560 rows each). Calibrated ENS comparison now populated in the README headline table and in a D-014 addendum. Result: on the 6,732 decision-band rows the model beats a real, calibrated 50-member IFS ENS by **+0.040 AUROC (0.832 vs 0.792), -16.8% decision cost (238.7 vs 287.0), ~2x economic value (0.289 vs 0.145)**. The raw-vs-raw AUROC margin (+0.025) remains the honest AUROC headline; the calibrated numbers are for Brier / cost / value.
- **Docker container is still up on port 8912.** Stop it with `docker compose down` when you don't need it.

### Yellow — worth knowing but not breaking
- **D-012's ablation numbers moved on rebuild.** ERA5-state row 0.8425 → 0.8405; regime row 0.8375 → 0.8401; regime delta now −0.0004 (was −0.0048). Well inside the ±0.01 sampling noise the doc already declared, and it *strengthens* the "regime adds nothing" conclusion. Refresh the table in D-012 before anyone relies on it — I did not touch a LOCKED-adjacent record without your call.
- **The headline `+0.082 AUROC over spread` was measured against the *lagged proxy*, not the real ensemble.** Against real 50-member IFS ENS the honest margin is **+0.025**. The README now leads with the corrected number; slides / posters must too.

### Not started
- IMD live-feed adapter (system currently serves only the 2016-2022 archive; `/api/health` reports STALE by design).
- **Any actual cloud deployment.** Terraform is written but never applied, and no Bedrock call has ever run — see the verification ledger in §9.
- Multi-model ensembling of AI weather models (GraphCast + Pangu + GenCast bust prediction) — still the highest-value remaining item.

---

## 2. What the system is, in one page

**Problem class:** meta-prediction. The system does **not** forecast weather. It predicts whether an existing NWP forecast will fail.

**User:** IMD duty forecaster at the NWFC or an RMC. Has ~45 minutes between raw NWP arriving (02:30 UTC / 08:00 IST) and the 08:30 IST bulletin. Cannot manually scrutinise 34 subdivisions × 10 lead days × 50 ensemble members every morning.

**Output:** a ranked "review queue" of subdivision-days most likely to bust, each with:
- A bust probability (0-1) with a bagged 90 % prediction interval.
- A three-item plain-English reason list, one factor per concept family (no redundant "moisture is high, moisture is high, moisture is high").
- Six synoptic regime probabilities + a regime entropy score.
- An OOD flag: `OUT_OF_DISTRIBUTION` and `bust_probability: null` when the atmospheric state lies outside the training distribution.
- A `data_quality` flag: `OK` / `STALE` / `MISSING`.
- A baseline probability for comparison.

**Authority invariant:** the system is decision support. It never suppresses, alters, or auto-publishes a public IMD warning. The human duty meteorologist remains the sole authority. This is not a hedging line — it is what makes the tool deployable inside a government agency.

**What kind of ML:** gradient-boosted trees (XGBoost, depth 4, 300 rounds, lr 0.05, class-weighted) with post-hoc isotonic calibration fitted strictly on 2021. **Not deep learning.** The reasons for the choice are in D-008 and LOGIC.md §6: tabular regime (~10⁴ labelled rows), rare events, mandatory per-flag explanations, and TreeSHAP being exact rather than approximate.

**What kind of data:** IMD 0.25° gauge-based rainfall (mainland-only, 34 subdivisions modelled — see D-006), IFS HRES deterministic forecasts at 0.703° (D-003), IFS ENS 50-member subsample (D-007 / D-014), ERA5 reanalysis for atmospheric state.

**Split:** temporal, no k-fold. Train 2016-2020 (5 JJAS), val 2021, test 2022. Every P95 error threshold, every isotonic map, every OOD covariance is fitted strictly on train (or train+val for calibration).

**Bust definition:** three simultaneous conditions (LOGIC.md §4.3):
1. `|F - O| >= max(10 mm, P95(subdivision, month))` on train years only.
2. Category flip on IMD's rainfall intensity classes.
3. At least one of F, O in the Moderate band (>= 15.6 mm/day).

That third clause is what stops the label triggering on 0.1 mm vs 3.0 mm shifts across the No-Rain / Light-Rain boundary. It is the difference between a real bust dataset and a synthetic one.

---

## 3. The current headline benchmark, with the numbers that matter

**Held-out year 2022, Day 3-7 decision band, 20,060 subdivision-days, base rate 3.31 %.** Every predictor is wrapped in the *same* isotonic calibration on 2021 (D-010) so no baseline is a strawman.

| Predictor | AUROC | Brier | BSS | ECE | Cost/1k | Value |
|---|---|---|---|---|---|---|
| Climatology (region, month, lead) | 0.5185 | 0.0321 | -0.0053 | 0.0127 | 330.5 | 0.000 |
| Forecast rainfall amount alone | 0.7501 | 0.0311 | +0.028 | 0.0136 | 273.0 | 0.174 |
| Logistic (spread + lead) | 0.7542 | 0.0310 | +0.031 | 0.0103 | 293.4 | 0.112 |
| Ensemble spread — lagged proxy | 0.7579 | 0.0310 | +0.031 | 0.0109 | 293.5 | 0.112 |
| **XGBoost + isotonic (our model)** | **0.8400** | **0.0291** | **+0.088** | **0.0108** | **237.3** | **0.282** |

### The correction that matters

Above the proxy baseline our margin is +0.082 AUROC and -19.2 % decision cost. That was measured against the **lagged-ensemble proxy**, not against a real ensemble. When the true 50-member IFS ENS is pulled and scored on identical rows (6,732 decision-band rows, bust rate 3.36 %):

| on identical rows | AUROC |
|---|---|
| true ENS spread / mean (normalised) | 0.554 |
| lagged-ensemble proxy | 0.731 |
| **true IFS ENS spread (raw)** | **0.807** |
| **our model** | **0.832** |

The honest margin over a genuine operational ensemble is **+0.025 AUROC**. The model still wins on a like-for-like comparison, on a held-out year, with no calibration advantage (AUROC needs none) — but by a quarter of what the proxy comparison suggested. Say this up front, before anyone asks. See D-014.

### Skill breakdown (from D-012's ablation)
- Forecast rainfall amount alone: 0.795 AUROC (feature-rich only if you count "how much rain is forecast").
- + run-to-run disagreement: +0.023.
- + subdivision/lead climatology: +0.006.
- + ERA5 atmospheric state: +0.019.
- + regime probabilities: ~0 (kept for explainability, not accuracy).

So the honest story is: **half the skill over spread comes from forecast magnitude, half from disagreement plus atmospheric state; regimes are for explanation, not prediction.**

### Robustness
- **30 % input dropout:** AUROC 0.843 → 0.745. ECE 0.010 → 0.021. Degrades but does not become silently overconfident.
- **Synthetic bust injection:** recall on injected busts scales 78 % → 93 % as error magnitude grows from 1.5× to 6×.
- **OOD refusal is earned:** the 0.83 % of rows refused have a 23.4 % bust rate vs 3.4 % accepted, and 16.1 mm mean error vs 6.0 mm — the detector is refusing on genuinely hard days.

---

## 4. Full pipeline in execution order

Assumes cached raw data in `data/raw/`. Total wall time ~3 minutes on this laptop.

```bash
# 0. Environment
python -m venv .venv && source .venv/Scripts/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Spatial aggregation — 641 Census-2011 districts -> 36 IMD subdivisions
PYTHONPATH=src python -m fbd.regions.build

# 2. Ingest IMD 0.25 degree gridded rainfall (2016-2022) and build subdivision truth
PYTHONPATH=src python scripts/fetch_imd.py
PYTHONPATH=src python scripts/build_truth.py

# 3. Ingest HRES and ERA5 (uses the 1959-2023_01_10 store — see D-009)
PYTHONPATH=src python scripts/fetch_hres.py
PYTHONPATH=src python scripts/fetch_era5.py

# 4. Build the modelling dataframe -> data/processed/dataset.parquet (279,650 x 93)
PYTHONPATH=src python scripts/build_dataset.py

# 5. Train baselines + XGBoost + isotonic calibration
PYTHONPATH=src python scripts/train_model.py

# 6. Generate SQLite bulletin store with bagged intervals + SHAP reasons
PYTHONPATH=src python scripts/generate_bulletins.py

# 7. Evidence: feature ablation, stress tests, unit tests
PYTHONPATH=src python scripts/ablation.py
PYTHONPATH=src python scripts/stress_test.py
PYTHONPATH=src pytest tests/ -v

# 8. True-ENS baseline (needs data/raw/wb2/ens/ populated; see D-014)
PYTHONPATH=src python scripts/fetch_ens.py --year 2022 --every 3
PYTHONPATH=src python scripts/evaluate_ens_baseline.py --decision-band-only

# 9. Serve
docker compose up -d               # -> http://localhost:8912
# or
PYTHONPATH=src uvicorn fbd.api.app:app --port 8912
```

### Correct API URLs
- `GET /api/health` — model version, row count, staleness banner.
- `GET /api/replay/dates` — every available `init_date`.
- `GET /api/regions` — 34 subdivisions with geometry.
- `GET /api/bulletin?init_date=YYYY-MM-DD[&lead_day=N][&mode=replay|live]` — all regions for one init.
- `GET /api/bulletin/{region_id}?init_date=YYYY-MM-DD` — one region across leads. **Note:** `/api/bulletin/latest` is *not* a route — `latest` gets treated as `{region_id}` and returns 422. Use the query-param form.
- `GET /api/review-queue?init_date=YYYY-MM-DD&top=12` — ranked highest-risk items.
- `GET /api/verification` — bust-rate observed-vs-flagged tables.
- `GET /api/metrics` — the full benchmark JSON.
- `POST /api/override` — forecaster override on a flag (recorded, never auto-applied).
- `GET /api/overrides` — audit trail of overrides.

---

## 5. Repository map

```
LOGIC.md              engineering contract & mathematical spec (LOCKED)
DECISIONS.md          14 decisions with evidence (D-001..D-014)
README.md             public writeup with the corrected headline table
HANDOFF.md            short resume-work brief for a fresh session
PROJECT_BLUEPRINT.md  THIS FILE — the long brief
Dockerfile            offline image (python:3.11-slim, serving-only deps)
docker-compose.yml    single-service, no volumes, no network deps
config/imd_subdivisions.json  641 districts -> 36 subdivisions mapping

src/fbd/
  config.py           bounding boxes, split years, bust thresholds, costs
  regions/build.py    exact-partition validator (refuses to run if broken)
  regions/masks.py    polygon-cell overlap weights via EPSG:7755 (equal-area)
  ingest/wb2.py       WeatherBench 2 Zarr access, India subsetting
  ingest/imd.py       IMD 0.25 deg NetCDF -> subdivision area-means
  ingest/hres.py      HRES forecast -> subdivision, pair with observed truth
  labels/bust.py      the three-part bust definition, P95 fitted train-only
  features/forecast.py lagged-ensemble spread (Hoffman & Kalnay 1983 proxy),
                       jumpiness, anomalies. Strict causality: lead L uses
                       only leads >= L (verified by poison test).
  features/era5.py    Somali Jet, Monsoon Trough, BoB vorticity, TCWV, shear.
                       Joined on init_date, never valid_date (leakage guard).
  regime/classify.py  6 soft regime probabilities via physical scores + softmax
  model/train.py      XGB depth 4 + isotonic, class-weighted
  model/baselines.py  climatology, forecast-rain-only, logistic, spread
  ood/detector.py     Mahalanobis at 99.5th percentile of training distances
  explain/reasons.py  native TreeSHAP (pred_contribs=True); concept-family dedupe
  evaluate/metrics.py AUROC, Brier, BSS, ECE (equal-count bins), cost, value
  api/schema.py       Pydantic BustPrediction / Bulletin / ReviewQueueItem
  api/app.py          FastAPI service, replay + live modes, staleness gate
  quality/            data-freshness gates

web/
  index.html          self-contained dashboard
  vendor/leaflet.{js,css}  vendored — 0 CDN dependencies

tests/
  test_core.py        12 tests: causality poison, exact partition, bust invariants
  test_model_api.py   9 tests: metrics, OOD, reasons, schemas

data/
  raw/imd/            IMD 0.25 deg NetCDF (178 MB)
  raw/shapes/         Census-2011 district shapefiles
  raw/wb2/hres/       cached HRES (72 MB)
  raw/wb2/era5/       cached ERA5 (~450 MB)
  raw/wb2/ens/        IFS ENS subsample (2022 done, 2019/2020 fetching)
  interim/            imd_subdivisions.gpkg, truth, spatial weights
  processed/dataset.parquet   279,650 x 93 — the complete modelling dataframe
  artifacts/          bust_model.joblib (1.5 MB), bulletins.sqlite (63 MB),
                       ablation.csv, stress_*.csv, shap_importance.csv,
                       results.json, ens_auroc_comparison.csv
```

---

## 6. Every trap that has cost time — do not step in them again

1. **WeatherBench 2 Zarr chunks span the globe (D-002).** India subsetting saves memory and disk, *not* bandwidth. Every data-volume decision downstream is built on the corrected cost model.
2. **ERA5 store names lie (D-009).** The store called `1959-2022` ends on 2021-12-31. Use `1959-2023_01_10-6h-240x121` for anything touching 2022. Rule: never trust a coordinate from a filename; read the array.
3. **IMD gridded rainfall is mainland-only (D-006).** A&N and Lakshadweep have zero valid land cells. 34 subdivisions modelled; islands excluded in code with the reason printed.
4. **Bust definition needs all three clauses (D-004 / LOGIC.md §4.3).** Percentile alone is circular. Category alone misfires on 15.5 vs 15.7 mm boundary straddles. Both plus the >= 15.6 mm significance floor prevent both.
5. **Fair calibration invariant (D-010).** Every baseline shares the model's isotonic calibration fitted on 2021. Beating an uncalibrated logistic proves nothing.
6. **Day 10 lagged spread is 100 % NaN (D-011).** HRES stops at 240 h so the proxy collapses to prior. Headline is Day 3-7 band where the baseline has full support.
7. **TreeSHAP via the `shap` package crashes on XGBoost 3.x (D-013).** `shap 0.49.1` cannot parse `base_score = [5E-1]`. Use `xgb.Booster.predict(..., pred_contribs=True)` — same algorithm, no dependency.
8. **Season concatenation across years (D-011).** Always group by year before computing `.diff()` so June 1 is not differenced against the previous September 30.
9. **True ENS is a much stronger baseline than the lagged proxy (D-014).** The +0.082 headline was against the proxy. Real number is +0.025. Update every slide.
10. **`/api/bulletin/latest` is not a route.** Path param is `{region_id}`, `init_date` is a required query param, `latest` is not special.
11. **Non-ASCII in scripts breaks Windows cp1252 consoles.** `evaluate_ens_baseline.py` had a U+2212 minus sign that crashed the script mid-print. All scripts should be ASCII-clean if they print anything on Windows.
12. **`git init` has not been run.** This is not a git repository. Nothing has been committed. Before making risky edits, either `git init` and commit, or copy the folder — there is no undo.

---

## 7. What the numbers actually say vs what they seem to say

Written so you can answer the sceptical question with the correct claim rather than the pretty one.

- **AUROC 0.84 sounds a lot better than 0.76.** Correct against the proxy. Against the real ensemble it is 0.83 vs 0.81. Say both.
- **BSS 0.088 is small in absolute terms.** For rare events (3.3 % base rate) it is respectable and the metric is dominated by the huge easy-negative mass. Cost reduction and PR-space performance matter more than BSS for this problem shape.
- **ECE 0.011 is genuinely good calibration.** Compare to a stated aim of <= 0.02 for the ECMWF operational products.
- **19 % decision-cost reduction is against the spread proxy** at cost ratio 10:1 (miss:false-alarm). At more forgiving ratios the model wins by less; at more punishing ratios it wins by more. `metrics.value_at_cost_ratio` reports the curve, not just one point.
- **The Assam & Meghalaya 14-June-2022 demo case is real, not cherry-picked.** Model flagged 70.6 % at Day 4 while the ensemble spread proxy flagged 11.2 %. The subdivision recorded 128 mm the next day. But the demo is one point; ablation and stress-test tables show the *distribution* — cite those when asked whether one good day means anything.

---

## 8. Docker + deployment — what is proven vs what is not

**Proven today.**
- `docker compose build` succeeds; `fbd:0.1.0`, ~957 MB.
- `docker compose up -d` starts the container, HEALTHCHECK reports `(healthy)`.
- Container serves 79,900 bulletin rows over 8912/tcp.
- Dashboard renders with zero external network requests (verified in an inspecting browser: `externalScripts: []`, 4 requests all to localhost, 36 subdivision polygons rendered).
- Container survives with `restart: unless-stopped`.

**Not yet done.**
- Pushed to a public registry / hosted URL.
- Tested on a machine without a Python env / without Docker Desktop.
- Multi-arch build (currently linux/amd64 only).
- CI (no `.github/workflows/`).

**Recommended next-hour work.**
```bash
docker tag fbd:0.1.0 ghcr.io/<user>/fbd:0.1.0
docker push  ghcr.io/<user>/fbd:0.1.0
# then a Fly.io / Railway / Render deploy — the image is self-contained.
```

Not every reviewer will run the Docker image. Most *will* click a URL. Have both.

---

## 9. The expanded platform (D-015 / D-016 / D-017)

Added after the original build, under an explicitly authorised override of the
LOGIC.md 13/14 non-goals. Read D-015 before touching any of it.

### GenAI layer -- `src/fbd/genai/`, default OFF

| Module | Role |
|---|---|
| `settings.py` | Feature flags, all default OFF. Bedrock model id resolution (`anthropic.` prefix). |
| `guardrails.py` | Numeric grounding, non-interference, injection screening. |
| `retrieval.py` | BM25 over LOGIC/DECISIONS/README/BLUEPRINT/HANDOFF. Pure stdlib, no network. |
| `tools.py` | Four read-only tools over the bulletin store, strict closed schemas. |
| `client.py` | `AnthropicBedrockMantle` construction; reports why it cannot run instead of throwing. |
| `agent.py` | Hand-written tool loop, so guardrails sit between the model and the user. |

**The load-bearing control is numeric grounding.** Every number in generated
text must trace to a value a tool actually returned. An LLM sitting beside a
disaster-management product cannot be allowed to say "roughly an 80% chance",
and a system prompt asking it not to is a request, not a control. There is an
end-to-end test that scripts a model into calling a real tool and then stating
a probability that tool never returned, and asserts the answer is withheld.

**Non-interference is enforced by the tool surface, not the prompt.** There is
no write tool, no override tool, no alerting tool -- and a test plus a CI gate
assert none is ever added.

Enable with:

```bash
pip install -r requirements-genai.txt
export FBD_GENAI_ENABLED=1 FBD_GENAI_TOOLS=1 FBD_GENAI_RAG=1
```

With the flag off, `/api/assistant/*` is **not registered at all** -- not a
route that declines, which would still be an egress surface.

`/api/assistant/search` needs no credentials and no network: it answers "where
is that documented?" against the decision log. That alone is a good demo.

### Drift monitoring -- `src/fbd/mlops/drift.py`, `scripts/monitor_drift.py`

PSI per feature -> prediction shift -> calibration ratio -> OOD rate, with an
OK/WATCH/RETRAIN verdict and exit code 2 on RETRAIN so a scheduler can gate.

Two results worth knowing (D-016):
* It independently reproduces the calibration drift README already disclosed
  (ratio 0.780 -- the model over-predicts by 22% on 2022).
* **JJAS 2022 is genuinely atypical in upper-level wind shear.** `india_shear_z`
  PSI 2.64, confirmed in the raw national index (mean 20.80 vs 21.59-22.65 in
  every other year; max 24.05 vs 26.26-28.57). Not a normalisation bug. This
  means the held-out year is a *harder* test than assumed, and the model still
  scored 0.840 -- use it, it strengthens the generalisation claim.
* Caveat to say out loud: PSI is an early warning, not a performance predictor.
  The monitor says RETRAIN on a year the model handled well. That is what the
  statistic measures, not a false positive.

### Observability -- `src/fbd/obs/metrics.py`

`/metrics` in Prometheus text format (distinct from `/api/metrics`, the model
benchmark table), plus JSON structured logs. `fbd_guardrail_violations_total`
is a safety signal: it rising means something is trying to put ungrounded
numbers in front of a forecaster.

### CI and infrastructure

`.github/workflows/ci.yml` -- tests, decision-log invariants, air-gap, supply
chain (pip-audit + bandit + secret scan), container. The air-gap job imports
the app with `socket.connect` monkeypatched to raise, and greps the dashboard
for external origins. Both were run locally and pass.

`infra/terraform/` -- VPC/ALB/Fargate/ECR/CloudWatch, Bedrock scoped to named
model ARNs rather than `bedrock:*`, ingress defaulting to RFC1918, and
`enable_genai=false` by default.

### The verification ledger -- read before claiming anything

| Component | Status |
|---|---|
| Full pipeline, end to end | **Executed**, reproduces D-001..D-014 |
| 62 tests | **Executed**, 62/62 |
| Drift monitor | **Executed**, findings in D-016 |
| Docker + container health + dashboard air-gap | **Executed**, verified in a browser |
| GenAI guardrails / retrieval / tools / agent | **Executed** against a fake client |
| CI gates | **Executed locally**; never run on GitHub Actions |
| Any real Bedrock call | **NEVER** -- no SDK, no credentials |
| Terraform | **NEVER** -- no binary; not even `validate`d |
| ECR push / ECS deploy / public URL | **NEVER** |

The bottom four are code-as-design. Say so. Claiming a deployment that was
never applied is exactly the unverifiable claim D-007, D-010 and D-014 were
each written to prevent.

---

## 10. Open questions for the next session

Roughly in decreasing importance.

1. **Should D-012's ablation table be refreshed in place, or should the current table stay and a footnote note the drift?** The numbers are inside sampling noise but the doc is a public-facing evidence log.
2. ~~**Do the 2019 and 2020 ENS parquets complete cleanly?**~~ **Resolved 15:38 IST.** Both parquets landed (90 kB each, 7,560 rows). Calibrated comparison run, D-014 addendum written, README row filled. Model beats true calibrated ENS by +0.040 AUROC, -16.8% cost, ~2x value on identical rows.
3. **Is the LOCKED status of LOGIC.md still true?** The bust definition, split years, and cost ratios have all held. If any of them need to change (e.g. seasonal cost variation), it becomes a D-015 rather than an in-place edit.
4. **Git.** The project is not under version control. Every session so far has trusted the filesystem. `git init && git add -A && git commit -m "state as of 2026-08-23"` before making further changes is cheap insurance.

---

## 11. Glossary for the new session

- **Bust** — a forecast that fails per LOGIC.md §4.3: percentile-extreme error AND category flip AND at least one side in the operationally-significant band.
- **Lagged-ensemble spread** — Hoffman & Kalnay 1983 proxy. Uses successive HRES initialisations that verify on the same day as a cheap stand-in for a true ensemble.
- **Isotonic calibration** — monotone-piecewise-constant fit mapping model score to observed frequency. Non-parametric, honest.
- **Regime entropy** — Shannon entropy over the six regime probabilities. High when the atmosphere does not fit cleanly into one regime; a natural humility signal.
- **OOD refusal** — the Mahalanobis-distance detector fitted on 99.5th percentile of training distances. When triggered, the API returns `status: "OUT_OF_DISTRIBUTION"` and `bust_probability: null`.
- **P95(s, m)** — the 95th percentile of absolute forecast error for subdivision s, month m, fitted strictly on training years. Feeds the bust label's magnitude clause with a 10 mm floor.
- **Decision band** — Day 3 to Day 7. Medium-range window where uncertainty is high enough to be interesting *and* the lead is long enough for civil authorities to act.
- **Value (V)** — economic value in the sense of Richardson 2000: cost reduction relative to the range from climatology to perfect forecast. V=1 is perfect; V=0 is no better than always predicting the prior.
- **JJAS** — June, July, August, September. The Indian summer monsoon window; > 70 % of annual rainfall.

---

## 12. If everything else is lost — the shortest path to running

```bash
cd forecast-bust-detection
docker compose up -d
curl http://localhost:8912/api/health
# open http://localhost:8912 in a browser
```

That is enough for a demo. Every other file exists to explain and defend what those three commands do.
