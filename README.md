# Forecast Bust Detection for Medium-Range Weather Forecasts

**SIH 2026 · Software · Disaster Management · PS SIH26079 · Ministry of Earth Sciences**

Predicts **when an existing medium-range rainfall forecast is about to fail** over
India — which subdivision, which lead day, and why — instead of predicting the
weather itself.

> We are not competing with IMD's forecast.
> We are telling IMD's forecaster which days to look at twice.

The system is **decision support**. It never issues or suppresses a public
warning; a human forecaster is always the authority.

---

## Headline result

Held-out year **2022** (never touched during training or calibration),
**Day 3–7 decision band**, 20,060 subdivision-days:

| Predictor | AUROC | Brier | BSS | Decision cost / 1000 | Value |
|---|---|---|---|---|---|
| Climatology (region, month, lead) | 0.519 | 0.0321 | −0.005 | 330.5 | 0.000 |
| Forecast rainfall amount alone | 0.750 | 0.0311 | 0.028 | 273.0 | 0.174 |
| Logistic regression (spread + lead) | 0.754 | 0.0310 | 0.031 | 293.4 | 0.112 |
| **Ensemble spread — the baseline to beat** | **0.758** | 0.0310 | 0.031 | 293.5 | 0.112 |
| **XGBoost + isotonic calibration** | **0.840** | **0.0291** | **0.088** | **237.1** | **0.282** |
| **True IFS ENS spread (50 members, calibrated)** † | **0.792** | **0.0310** | **+0.045** | **287.0** | **0.145** |

† **The true-ensemble row is scored on a subsample and is the number that
matters.** The full-archive ensemble costs ~105 GB (DECISIONS.md D-007), so the
rows above it use a *lagged-ensemble proxy*. A stratified subsample of real
50-member IFS ENS (41 init dates, JJAS 2022) was pulled to check that proxy.
On the **6,732 decision-band rows where both exist**, scored identically:

| on identical rows (n=6,732, bust rate 3.36%) | AUROC |
|---|---|
| lagged-ensemble proxy | 0.731 |
| **true IFS ENS spread** | **0.807** |
| **our model** | **0.832** |

So the margin over a *real* operational ensemble is **+0.025 AUROC**, not the
+0.082 measured against the proxy — and **+0.025 cannot be distinguished from
zero on this subsample.** A cluster bootstrap over the 40 init dates puts it at
**+0.0251 [−0.0083, +0.0580]**, an interval that contains zero. Against the
lagged proxy the margin is solid (+0.0821 [+0.0615, +0.1039]); against a real
50-member ensemble, on 40 days, it is not yet established. See
[uncertainty](#uncertainty-every-number-above-has-an-interval) below and D-022.

AUROC is rank-based and needs no
calibration, so the true-ensemble comparison involves no fitting whatsoever.
The calibrated cells above (Brier, BSS, cost, value) use isotonic maps fitted
on **2019 + 2020** ENS subsamples -- training-year data, no leakage. Note that
isotonic calibration slightly *lowers* true-ENS AUROC (0.807 raw -> 0.792
calibrated) because monotone step-fits introduce rank ties on the test year.
AUROC is rank-based so the honest AUROC comparison is the **raw** row above
(0.807 vs 0.832, +0.025 in the model's favour); the calibrated numbers are for
Brier, cost, and value. Both are reported so nothing gets cherry-picked. See
D-014.

Against the lagged proxy the margin is **+0.082 AUROC** with a 19% reduction in
asymmetric decision cost, and every predictor is given the *same* isotonic
calibration on the *same* validation year so the comparison is fair.

Everything above is reproducible from `scripts/` with public data only.
No GPU is used anywhere.

### Uncertainty: every number above has an interval

```bash
PYTHONPATH=src python scripts/compute_confidence_intervals.py
```

The test year is 39,950 rows but only **122 init dates**. Each date contributes
36 subdivisions × 10 leads that share one synoptic situation, so the rows are
nowhere near independent — a per-row bootstrap would treat 39,950 correlated
observations as 39,950 independent ones and produce intervals roughly **2×
too narrow**. These resample **init dates**, not rows.

| Predictor | AUROC [95%] | BSS [95%] |
|---|---|---|
| Climatology | 0.518 [0.496, 0.541] | −0.005 [−0.011, −0.002] |
| Forecast rainfall alone | 0.750 [0.729, 0.771] | 0.028 [0.000, 0.054] |
| Ensemble spread (proxy) | 0.758 [0.732, 0.784] | 0.031 [0.007, 0.054] |
| **XGBoost + isotonic** | **0.840 [0.821, 0.859]** | **0.088 [0.053, 0.123]** |

Margins are bootstrapped **paired** — both predictors scored on the same
resampled rows — because two overlapping marginal intervals do not tell you
whether a difference is distinguishable from zero.

| Comparison | ΔAUROC [95%] | Distinguishable from zero? |
|---|---|---|
| model − lagged-ensemble proxy (20,060 rows, 120 dates) | +0.0821 [+0.0615, +0.1039] | **yes** |
| **model − true IFS ENS, raw spread** (6,732 rows, 40 dates) | **+0.0251 [−0.0083, +0.0580]** | **no** |
| model − true IFS ENS, calibrated (6,732 rows, 40 dates) | +0.0403 [+0.0052, +0.0758] | yes, barely |

**The honest reading.** Beating the cheap proxy is established. Beating a real
50-member operational ensemble **is not** — on 40 init dates the margin is
+0.025 with an interval that contains zero. The calibrated comparison clears
zero, but only because isotonic step-fits introduce rank ties that *lower* ENS
AUROC (0.807 → 0.792); the raw spread is the stronger ENS variant and therefore
the fair test. The fix is more init dates, not a better argument. Full method
and the intervals for Brier and ECE: [`DECISIONS.md` D-022](DECISIONS.md).

### The two claims, drawn

| calibration | earned refusal |
|---|---|
| [![Reliability diagram](docs/figures/reliability.png)](docs/FIGURES.md) | [![Earned refusal](docs/figures/earned_refusal.png)](docs/FIGURES.md) |

Left: observed bust frequency against predicted probability on the held-out
2022 rows the product actually serves. ECE 0.0107, and the curve tracks the
diagonal through the operational range — but the **highest bin predicts 0.207
and observes 0.158**, overconfident by 0.049 on ~80 rows. That is labelled on
the figure rather than left for a low ECE to paper over, because the top bin is
where a forecaster is looking.

Right: the days the model *declines* to score bust at **23.4%** against **3.4%**
for the days it accepts — 6.9×. A refusal is not a low-risk result, which is why
there are three escalation tiers and not two.

[`docs/FIGURES.md`](docs/FIGURES.md) has the method, the honest reading, and a
reconciliation of the small gap against the table above (the served subset
excludes 173 refused rows; 19,887 + 173 = 20,060).

Regenerate with `PYTHONPATH=src python scripts/plot_evaluation_figures.py`.

---

## What it does

Five outputs, matching the five the problem statement names:

1. **Forecast confidence map** — subdivision-wise, Day 1 to Day 10.
2. **Bust probability** — calibrated, per subdivision and lead time.
3. **Error-prone area detection** — a ranked *review queue*, not just a map.
4. **Explainable output** — plain-language meteorological reasons for every flag.
5. **Prototype dashboard + API** — FastAPI + Leaflet, runs fully offline.

## What a bust is

A bust is **a large error that also flips the rainfall category into or out of
operationally significant rain** — i.e. an error that would have changed the
forecaster's alert decision. Formally, for subdivision *s*, lead *L*, with
forecast `F` and IMD-observed `O` area-mean rainfall:

```
bust = |F − O| ≥ max(10 mm, P95(s, month))          # magnitude
       AND category(F) ≠ category(O)                 # decision flips
       AND max(F, O) ≥ 15.6 mm/day                   # significant rain
```

`P95` is fitted on **training years only**, so the held-out bust rate is a
measurement rather than a construction. Base rate: **4.07%** overall, rising
monotonically from 2.6% at Day 1 to 5.8% at Day 10 — the physically expected
behaviour, and a first-order sanity check on the whole label.

Full reasoning, including why neither the percentile nor the category criterion
works alone, is in [`src/fbd/labels/bust.py`](src/fbd/labels/bust.py).

---

## Quick start

### Run it straight from a clone (no raw data download)

The derived modelling frame and the trained model are committed to the repo, and
the precomputed dashboard store ships as a release asset — so you can run the
tests and the offline dashboard without downloading a single byte of raw NWP
data. See [`DATA.md`](DATA.md) for exactly what lives where.

```bash
pip install -r requirements.txt
export PYTHONPATH=src                                  # Windows: set PYTHONPATH=src

python -m pytest tests/ -q                             # 68 tests, all pass from the clone
python scripts/fetch_release_artifacts.py              # downloads bulletins.sqlite (~60 MB), checksum-verified
python -m uvicorn fbd.api.app:app --app-dir src --port 8912
```

Open <http://localhost:8912>. Prefer not to download the release asset? Regenerate
it locally from the committed artifacts instead: `python scripts/generate_bulletins.py`.

### Reproduce everything from the primary sources

To rebuild the derived data from scratch, in order (each step caches, so re-runs
are cheap):

```bash
python -m fbd.regions.build          # 36 IMD subdivisions from 641 districts
python scripts/fetch_imd.py          # IMD 0.25° gauge rainfall, 2016-2022 (178 MB)
python scripts/build_truth.py        # subdivision daily observed rainfall
python scripts/fetch_hres.py         # HRES forecasts (~9 GB transfer, ~21 min)
python scripts/fetch_era5.py         # ERA5 analysis (~17 GB transfer, ~26 min)
python scripts/build_dataset.py      # labels + features -> dataset.parquet
python scripts/train_model.py        # baselines + model + held-out evaluation
python scripts/generate_bulletins.py # batch score -> bulletins.sqlite
```

Set `PYTHONPATH=src` (or `pip install -e .`) so `fbd` is importable.

### Run the service

```bash
python -m uvicorn fbd.api.app:app --app-dir src --port 8912
```

Open <http://localhost:8912>. The map, replay slider, reason panel,
model-vs-baseline toggle and review queue are all served from the local SQLite
file — **no network access is required once the data is cached.**

### Run the demo

```bash
python scripts/replay_demo.py
```

Narrated terminal replay of the June 2022 Assam–Meghalaya floods, chosen from
the held-out year. Works with the network unplugged.

### Verify it

```bash
python -m pytest tests/ -q
python scripts/ablation.py
python scripts/stress_test.py
```

---

## The demo case

**Assam & Meghalaya, valid 17 June 2022** — the June 2022 Assam–Meghalaya
floods. Observed area-mean rainfall **115.6 mm/day** across the whole
subdivision.

| Forecast issued | Lead | Forecast | Our model | Ensemble spread |
|---|---|---|---|---|
| 11 June | Day 7 | 86.7 mm | 45.6% | 33.3% |
| 12 June | Day 6 | 67.4 mm | 42.3% | 21.2% |
| 13 June | Day 5 | 74.7 mm | 47.5% | 17.6% |
| **14 June** | **Day 4** | **83.7 mm** | **70.6%** | **11.2%** |

The forecast busted. As the event approached, **ensemble spread fell** — the
models were converging, which reads as growing confidence — while our model went
the other way.

The Day-4 reason panel said:

> * the forecast is +61.9 mm/day above this subdivision's seasonal normal (top 1% for this subdivision)
> * this subdivision and lead time historically bust often (5.0% of days)
> * column moisture over India is below normal (−0.8 sd)

---

## How it works

```
IMD 0.25° gauge rainfall ─┐
WB2 IFS HRES forecasts   ─┼─► area-weighted aggregation to 36 IMD subdivisions
WB2 ERA5 analysis        ─┤   (exact polygon×cell overlap, not centroid masking)
WB2 IFS ENS (subsample)  ─┘
                              │
                              ▼
                    BUST LABEL (§ above) ── train years only for all thresholds
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  forecast features    ERA5 state features    regime classifier
  spread, jumpiness,   moisture flux, shear,  6 soft probabilities
  climatology          Somali jet, BoB vort.  + entropy
        └─────────────────────┼─────────────────────┘
                              ▼
              XGBoost (class-weighted) + isotonic calibration
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
        OOD detector    decision engine   TreeSHAP
        refuse to score  asymmetric cost   → plain-language reasons
                              │
                              ▼
              SQLite bulletins → FastAPI → Leaflet map
                              │
                              ▼
                    DUTY FORECASTER (the authority)
```

**Causality is enforced everywhere.** Every feature is computed from information
available at initialisation time:

* ERA5 features are joined on `init_date`, never `valid_date` — the analysis
  valid on the target day does not exist when the forecast is issued.
* The lagged-ensemble spread for lead *L* uses only leads *≥ L*, because leads
  below *L* verify on the same day but are issued *later*. There is a test that
  poisons the short leads and asserts the feature does not move.
* Every threshold, climatology and calibration is fitted on training or
  validation years only.

---

## Honest findings

These are reported because they are true, not because they help.

* **Roughly half the skill over spread comes from forecast rainfall magnitude.**
  A model given only forecast amount and lead day already scores 0.795. The
  remaining gain comes from run-to-run disagreement (+0.023) and atmospheric
  state (+0.021).
* **Regime features add no predictive skill** (−0.005 AUROC, within noise). They
  are kept because the problem statement mandates regime-based explanation and
  because the classifier is independently validated — on its top-10% monsoon
  depression days, Odisha observes **25.1 mm/day vs 5.3 mm/day** otherwise.
* **The real ensemble is a much stronger baseline than our proxy** (0.807 vs
  0.731 AUROC on identical rows). Our margin over a genuine 50-member IFS ENS is
  **+0.025 AUROC**, not the +0.082 the proxy comparison suggests. The model still
  wins, by a modest and real amount. See D-014.
* **The spread baseline is undefined at Day 10** because the HRES archive stops
  at 240 h, which inflates the all-lead comparison. That is why the headline
  number is the Day 3–7 band, where the baseline has full support.
* **Calibration transfers imperfectly across years.** The isotonic fit on 2021
  (4.58% bust rate) slightly over-predicts on 2022 (3.59%). ECE is 0.0108. This
  is exactly the calibration drift the monitoring layer is designed to catch.
* **A 3-hour window offset exists** between IMD's 0830–0830 IST rainfall day and
  the 00Z–00Z forecast accumulation. It cannot be removed at this resolution and
  applies identically to model and baselines.

Every deviation from the build spec, with evidence, is in
[`DECISIONS.md`](DECISIONS.md).

---

## Robustness

| Test | Result |
|---|---|
| **30% input dropout** | AUROC 0.843 → 0.745; still beats the full-input spread baseline. ECE degrades 0.010 → 0.021, so the system becomes less calibrated, not silently over-confident. |
| **Synthetic bust injection** | Flagged rate tracks the true injected bust rate at every level (e.g. 37.3% flagged vs 31.7% truly bust at 3× scaling). Recall on injected busts: 78% → 93%. |
| **Out-of-distribution refusal** | 0.83% of region-days refused. Refused rows have a **23.4% bust rate vs 3.4%** for accepted rows, and 16.1 mm vs 6.0 mm mean error — the refusal is earned, not decorative. |
| **Staleness** | `mode=live` correctly reports `STALE` against a 2016–2022 archive and greys the map. A stale high-confidence number is the worst thing this system could produce. |

---

## Data

All public, all anonymous access, none requiring registration. **What is
committed to this repo, what ships as a release asset, and what is reproduced
from the primary sources is documented in full in [`DATA.md`](DATA.md).** The
raw third-party data is deliberately not rehosted here.

| Role | Source | Notes |
|---|---|---|
| India rainfall truth | [IMD 0.25° gridded](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) | gauge-based, ~6,955 stations; outranks ERA5 for rainfall over land |
| Deterministic forecasts | WeatherBench 2 IFS HRES | `gs://weatherbench2/datasets/hres/…512x256…` |
| Ensemble (baseline) | WeatherBench 2 IFS ENS | subsampled — see DECISIONS.md D-007 |
| Atmospheric state | WeatherBench 2 ERA5 | `…1959-2023_01_10-6h-240x121…` — **not** the store named "1959-2022", which ends in 2021 |
| Subdivision geometry | [datameet/maps](https://github.com/datameet/maps) Census-2011 districts | 641 districts → 36 IMD subdivisions |

---

## Repository layout

```
config/imd_subdivisions.json   the 36-subdivision definition (the domain artifact)
src/fbd/
  regions/    subdivision construction + exact area-overlap weights
  ingest/     WB2 zarr, IMD netCDF, HRES aggregation
  labels/     the bust definition
  features/   forecast-derived and ERA5-derived features
  regime/     six-regime soft classifier
  model/      baselines + XGBoost + calibration
  ood/        Mahalanobis out-of-distribution detector
  explain/    TreeSHAP → plain-language reasons
  evaluate/   AUROC, Brier, reliability, decision cost, value
  api/        FastAPI service + output schema
web/          Leaflet dashboard (vendored, no CDN)
scripts/      the pipeline, in run order
tests/        correctness tests for what would fail silently
DECISIONS.md  every deviation from LOGIC.md, with evidence
```

## Deliberate non-goals

No mobile app, no chatbot, no login, no microservices, no CNN/transformer/GNN,
no Kafka, no Kubernetes, no real-time streaming, and **no attempt to forecast
the weather better**. See LOGIC.md §14.
