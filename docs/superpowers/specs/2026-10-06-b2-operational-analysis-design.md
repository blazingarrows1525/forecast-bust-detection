# B2 — operational-analysis substitution, with derived TCWV — design

**Date:** 2026-10-06
**Status:** approach chosen by the owner in chat ("derive TCWV from humidity");
this file fixes the details and the gates **before any derived TCWV was
compared with anything**.
**Backlog:** `docs/ML_RESEARCH_BACKLOG.md` B2.
**Concurrency note:** a separate, uncommitted draft in the main checkout
(`src/fbd/ingest/hres_t0.py`) takes the opposite position, refusing any
synthesised TCWV. This study uses different file names
(`fbd.features.tcwv`, `scripts/fetch_analysis_t0.py`,
`data/raw/wb2/analysis_t0/`) so the two cannot collide. Only one of them
should be merged.

---

## 1. The question

> Rebuilt from the operational HRES initial state instead of ERA5, do the 30
> analysis-derived inputs cost the bust model no material ranking skill?

It is a **non-inferiority** question. 30 of the 52 served inputs come from a
reanalysis that does not exist at issue time
(`docs/FEATURE_AVAILABILITY_MATRIX.md`). If the operational state does as
well, those inputs have a live, legal source.

## 2. Facts verified before designing

| fact | source |
|---|---|
| WB2 `hres_t0` 240×121 (1.5°) covers 2016–2022 6-hourly with geopotential, specific humidity, u, v on 13 levels (50–1000 hPa), MSLP and surface pressure | store metadata, read 2026-10-06 |
| It has **no total column water vapour** | the same metadata |
| TCWV feeds 11 of the 30 inputs: `tcwv`, `india_tcwv`, `india_tcwv_z`, `india_tcwv_d1`, `india_tcwv_d3` and, through the classifier, every regime probability and the regime entropy | `fbd.features.era5`, `fbd.regime.classify` |
| On the store's 1.5° grid the monsoon domain is exactly the ERA5 cache's 37 × 47 grid | coordinates |
| Every analysis-derived input reads the 00Z state of the issue day (and earlier days for tendencies) | `fbd.features.era5` |

## 3. Derived TCWV

`fbd.features.tcwv.column_water_vapour`: (1/g) ∫ q dp, trapezoid on the
levels above the ground, the cut segment integrated to the surface with q
interpolated linearly in pressure, q held at 1000 hPa below that level, and
nothing above 50 hPa. Tested exactly against constant and linear profiles.

## 4. Gates, fixed now (before any comparison)

The derived field must pass both, or the study stops and records why.
Evaluated at 00Z over the India box (6–38°N, 66–100°E), on **2016 and 2017
only**: training years in every fold, never a test year.

| gate | what it isolates | pass |
|---|---|---|
| G1 integration error | ERA5's own humidity (13 levels) and surface pressure, integrated, against ERA5's own TCWV, on the same grid and times | RMSE ≤ 2.0 kg m⁻² and \|mean bias\| ≤ 1.5 kg m⁻² |
| G2 source agreement | `hres_t0` derived TCWV against ERA5 TCWV | Pearson r ≥ 0.95 and RMSE ≤ 4.0 kg m⁻² |

G1 asks whether 13 levels can carry the column at all. G2 asks whether the
operational state plus the approximation tracks the reanalysis the model was
trained on. Neither gate looks at a label.

## 5. The comparison

- **Folds, labels and rows:** S1b's four strict folds (test 2019–2022), the
  same bust labels, and S1b's comparison rows (Day 3–7 test rows with a label
  and ENS spread, 20,060 a year).
- **Candidate:** each fold rebuilt with the 30 analysis-derived inputs from
  `hres_t0` (derived TCWV), everything else identical. That includes the
  regime classifier refitted on the fold's training years, the same
  standardisation years, ERA5's static orography and land–sea mask (constant
  fields, available live), `DEFAULT_PARAMS` XGBoost on the CPU exactly as the
  incumbent was trained, and isotonic on the validation year.
- **Incumbent:** the pinned S1b fold models (`backtest.json`).
- **Primary:** the mean over 2019–2022 of the within-year margin
  AUROC(candidate) − AUROC(incumbent). Stratified paired cluster bootstrap over
  init dates, 10,000 resamples, seed 20260919, two-sided 95% interval.
- **Margin:** δ = 0.005 AUROC.
- **Verdict:**
  - lower > −δ: **non-inferior**
  - upper < −δ: **inferior**
  - otherwise: **inconclusive**
- **Secondary** (reported, cannot decide): the margin per year (2,000
  resamples); Brier skill on the decision band; the refusal rate under the
  same OOD rule; per-feature agreement between the two sources on the
  training years.

## 6. Consequences, committed now

| verdict | recorded as |
|---|---|
| non-inferior | the `hres_t0` features (with derived TCWV) become the live feature set's source; FEATURE_AVAILABILITY_MATRIX moves the 30 inputs to "operational source, validated" |
| inferior or inconclusive | the live path cannot claim the full model; it ships a reduced model or stays archive-only, and says so |
| a gate fails | derived TCWV is not fit for use; B2 falls back to the owner's other option (a reduced-input study) |

Nothing served changes in this study. The served store stays ERA5-built.

## 7. Before registration: the audit (`scripts/audit_b2.py`)

1. **Neutrality:** with `source="era5"`, the changed builder reproduces each
   pinned fold dataset byte for byte (`backtest.json` hashes).
2. **Gates G1 and G2** (§4), on 2016–2017.
3. **Causality:** the rebuilt features for a sample of 2018 rows are unchanged
   when every `hres_t0` state after the issue time is poisoned.
4. **Determinism:** two trainings of the fold-2019 candidate give
   bit-identical validation-year (2018) output.

Then `docs/PREREGISTRATION_B2.md`, committed and pushed with CI green,
pinning the input files, the code and the margin. Then one scoring run.
