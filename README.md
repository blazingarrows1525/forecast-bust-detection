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
| **True IFS ENS spread (50 members, calibrated)** † | 0.799 | 0.0309 | +0.034 | 289.8 | 0.123 |

† **The true-ensemble row is the number that matters.** The full-archive
ensemble costs ~105 GB (DECISIONS.md D-007), so the rows above it use a
*lagged-ensemble proxy*. Real 50-member IFS ENS was fetched for **all 122 init
dates of JJAS 2022** (and of 2021, for calibration); 120 of them have Day 3–7
labels, so the ensemble now covers the **same 20,060 decision-band rows** as
the table above. Scored identically on those rows:

| same rows (n=20,060, bust rate 3.31%) | AUROC |
|---|---|
| **true IFS ENS spread** | **0.808** |
| **our model** | **0.840** |

**The model outranks a real 50-member operational ensemble over the full
held-out season of 2022:** a cluster bootstrap over the 120 init dates puts the margin
at **+0.0316 [+0.0141, +0.0485]**, an interval above zero. The test was
registered before this data was fetched and run once
([`docs/PREREGISTRATION_S1.md`](docs/PREREGISTRATION_S1.md)): 122 ENS init
dates fetched; 120 used (dates with Day 3–7 labels). The model scores every one
of these rows, including those the served product would refuse as
out-of-distribution, so this measures ranking skill rather than the served
subset. The ensemble still carries information the model lacks: a two-term
combination of the model and ENS spread, fitted on 2021, beats the model alone
by +0.0154 [+0.0073, +0.0240]. Every interval is drawn in
[`docs/figures/ens_settlement.png`](docs/figures/ens_settlement.png); the
record is D-025.

The margin is not spread evenly. Broken down by month (exploratory, no claim
drawn) it is clear in June and July and not distinguishable from zero in
August or September; by lead it is largest at Days 1–2 and 8–10. The earlier
40-date subsample gave +0.0251 [−0.0083, +0.0580], which could not be told from
zero; the 80 dates added since give +0.0351 [+0.0157, +0.0539]. Outside 2022
it is not established: on average over 2019–2021 the margin is +0.0036
[−0.0059, +0.0127], an interval that contains zero, and in 2019 the ensemble
outranks the model ([below](#does-it-hold-in-other-years), D-026).

AUROC is rank-based and needs no calibration, so the headline comparison
involves no fitting whatsoever. The calibrated cells in the table (Brier, BSS,
cost, value) use isotonic maps fitted on the training and validation years'
ENS rows (2019 and 2020 every sixth date, all of 2021) — no test-year data.
Isotonic calibration slightly *lowers* true-ENS AUROC (0.808 raw → 0.799
calibrated) because monotone step-fits introduce rank ties on the test year,
so the fair AUROC comparison is the **raw** one above. Fitted on 2021 alone,
the model's own calibration year, the calibrated ENS still trails: ΔAUROC
+0.0439 [+0.0258, +0.0613] and 51 fewer cost units per 1,000 rows
[36, 66]. See D-014 and D-025.

### Does it hold in other years?

**Not established.** A rolling-origin backtest rebuilt the dataset and
retrained the model once per test year, each fold fitting everything (bust
thresholds, climatologies, standardisation, calibration) on its own earlier
years only, and compared it with real 50-member IFS ENS spread on every JJAS
init date of 2019, 2020 and 2021. The test was registered before the 2019–2020
ensemble data was fetched
([`docs/PREREGISTRATION_S1B.md`](docs/PREREGISTRATION_S1B.md)) and run once;
an audit first showed the fold machinery reproduces the published dataset and
model exactly.

| test year (trained on) | model − ENS spread, ΔAUROC [95%] | |
|---|---|---|
| 2019 (2016–17) | −0.0196 [−0.0359, −0.0031] | **ENS spread outranks the model** |
| 2020 (2016–18) | +0.0388 [+0.0246, +0.0533] | model ahead |
| 2021 (2016–19) | −0.0085 [−0.0260, +0.0071] | not distinguishable |
| **mean 2019–2021 (the registered test)** | **+0.0036 [−0.0059, +0.0127]** | **not distinguishable** |
| 2022 (2016–20; seen in S1) | +0.0324 [+0.0147, +0.0490] | model ahead |

The 2022 edge does not replicate on average, and in one of the three other
seasons the ensemble's own spread ranks busts better than the model does.
Against the cheap lagged proxy the model does hold up outside 2022: +0.0364
[+0.0278, +0.0452] on average over 2019–2021. Exploratory, no claim drawn: the
model is weakest late in the season, with September favouring the ensemble in
2019 and 2021 and August 2019 by −0.080. The 2019 fold trained on only two
seasons, which biases against the model; that is stated, not used to discount
the result. Figure: [`docs/figures/backtest.png`](docs/figures/backtest.png);
record: D-026.

### The model and the ensemble together

Neither the model nor the MLP beats a real ensemble's spread repeatably on its
own. Together they do. A two-input logistic combination of the model's
uncalibrated probability and the ENS spread, fitted inside each fold on its
validation year only, was registered before it was ever scored on 2019–2021
([`docs/PREREGISTRATION_S1C.md`](docs/PREREGISTRATION_S1C.md)) and run once:

**Model + ENS spread outranks ENS spread alone on average over 2019–2021:**
**+0.0244 [+0.0178, +0.0308]**.

| test year | combination | ENS spread | model alone | combination − ENS [95%] |
|---|---|---|---|---|
| 2019 | 0.818 | 0.802 | 0.783 | +0.0155 [+0.0056, +0.0255] |
| 2020 | 0.852 | 0.803 | 0.842 | +0.0488 [+0.0388, +0.0589] |
| 2021 | 0.833 | 0.824 | 0.815 | +0.0089 [−0.0047, +0.0209] |
| 2022 (seen in S1) | 0.856 | 0.808 | 0.841 | +0.0477 [+0.0365, +0.0581] |

The combination beats both of its parts in every year, and the ensemble adds to
the model in every year too (combination − model +0.0208 [+0.0172, +0.0245]).
Both inputs carry positive weight in every fold. So the operational answer is
neither "use the spread" nor "use the model": it is to read them together.
2021 alone does not clear zero. The same combiner on the MLP does better still
(+0.0334 [+0.0282, +0.0385] against ENS; a secondary). The served product does
not combine them yet; that is the next change to make. Record: D-029.

### Other model families

Three candidate families were declared before any was built, with one rule for
promoting them: beat the XGBoost fold models above, on the same rows, averaged
over the four test years, at a Bonferroni-corrected 98.33% interval
([`docs/PREREGISTRATION_S3.md`](docs/PREREGISTRATION_S3.md)). The first is an
MLP on exactly the model's 52 inputs, a control that isolates the architecture.
Scored once:

**The MLP outranks the XGBoost model across 2019–2022:**
**+0.0088 [+0.0023, +0.0157]**.

| test year | MLP AUROC | XGBoost AUROC | MLP − XGBoost [95%] |
|---|---|---|---|
| 2019 | 0.815 | 0.783 | +0.0319 [+0.0187, +0.0461] |
| 2020 | 0.847 | 0.842 | +0.0050 [−0.0041, +0.0140] |
| 2021 | 0.839 | 0.815 | +0.0237 [+0.0143, +0.0340] |
| 2022 | 0.815 | 0.841 | **−0.0255 [−0.0354, −0.0157]** |

The win is uneven: the MLP leads clearly in 2019 and 2021, and **XGBoost is
clearly better in 2022**, the season the product was built around. Against real
ENS spread, as a secondary that was not corrected for multiple comparisons and
was designed after D-026, the MLP leads on average over 2019–2021 by +0.0238
[+0.0167, +0.0307] and in each of those years, which XGBoost did not. That lead
was then put to a registered test on 2018, the one season no ensemble
comparison had touched (fold trained on 2016, calibrated on 2017; its ENS data
fetched only after registering, [`docs/PREREGISTRATION_S3A_CONFIRM.md`](docs/PREREGISTRATION_S3A_CONFIRM.md)).
**It was not confirmed:** MLP − ENS spread in 2018 is −0.0080 [−0.0254, +0.0087]
(AUROC 0.787 against 0.795; D-028). The MLP's lead over the ensemble is not
established. On that single training year XGBoost collapsed to 0.681 while the
MLP held up, a finding about little data rather than about the ensemble. The
served product is still XGBoost; promotion into it is a separate decision after
the temporal and spatial candidates.
Figure: [`docs/figures/candidate_mlp.png`](docs/figures/candidate_mlp.png);
record: D-027.

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
| **model − true IFS ENS, raw spread, 2022** (20,060 rows, 120 dates; registered, 10,000 resamples) | **+0.0316 [+0.0141, +0.0485]** | **yes** |
| **model − true IFS ENS, raw spread, mean of 2019–2021 backtest folds** (3 × 20,060 rows, 360 dates; registered, 10,000 resamples) | **+0.0036 [−0.0059, +0.0127]** | **no** |
| model − true IFS ENS, calibrated (20,060 rows, 120 dates) | +0.0408 [+0.0226, +0.0573] | yes |
| model − true IFS ENS, raw spread (superseded: 40-date subsample, 6,732 rows) | +0.0251 [−0.0083, +0.0580] | no |

**The honest reading.** Beating the cheap proxy is established, in 2022 and on
average over 2019–2021. Beating a real 50-member operational ensemble is
established for 2022 only: there the margin is +0.0316 with an interval above
zero, on a test registered before the data existed. Averaged over 2019–2021 it
is +0.0036, an interval that contains zero, and in 2019 the ensemble wins. The
raw spread is still the fair comparator, because calibration lowers ENS AUROC.
Full method and the intervals for Brier and ECE:
[`DECISIONS.md` D-022](DECISIONS.md); the 2022 settlement: D-025; the backtest:
D-026.

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
* **The real ensemble is a much stronger baseline than our proxy.** In 2022
  our margin over a genuine 50-member IFS ENS is **+0.0316 [+0.0141, +0.0485]
  AUROC**, not the +0.082 the proxy comparison suggests. Outside 2022 it does
  not hold up: averaged over 2019–2021 it is +0.0036 [−0.0059, +0.0127], and in
  2019 the ensemble outranks the model. The ensemble also adds information on
  top of the model. See D-014, D-025 and D-026.
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

No mobile app, no chatbot, no login, no microservices, no CNN/transformer/GNN
in the served product (neural candidates are tested under a registered rule and
none is served; D-027), no Kafka, no Kubernetes, no real-time streaming, and
**no attempt to forecast the weather better**. See LOGIC.md §14.
