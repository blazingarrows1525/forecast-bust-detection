# S1 — Settle the ENS question — design

**Date:** 2026-09-23
**Status:** approved in chat (four sections); corrected against the code during self-review (§3, §5.1, §5.2); awaiting spec review
**Part of:** a five-way decomposition of the "research + implementation master
prompt" (§2). This is the first sub-project and it gates the others.

---

## 1. The question

> Does the model outrank a real 50-member operational ensemble at telling which
> forecasts are about to bust?

Today's answer is **undecided**: +0.0251 AUROC [−0.0083, +0.0580] on 40 init
dates of the 2022 held-out season (DECISIONS.md D-014, D-022). The interval
contains zero.

It matters more than anything else on the roadmap. Forecasters already have the
ensemble. If the model does not beat ensemble spread, the honest operational
answer is "use the spread", and deep learning, real-time serving and shadow
deployment would all be refinements of the wrong product.

## 2. Where this sits

The master prompt listed ten capabilities. They reduce to five sub-projects
with one dependency spine:

```
S1  Audit + settle the ENS question      ← this document
 ├─ S2  Model-source disagreement
 ├─ S3  Experimentation framework (temporal / spatial / DL), with a promotion rule
 └─ S4  Near-real-time inference (ECMWF Open Data, local, shadow mode)
      └─ S5  Live frontend + MLOps
```

The prompt's own rule, *information gained per unit of additional complexity*,
puts S1 first: it is one existing script, a resumable fetch, and a
pre-registered test.

## 3. Facts verified before designing

| fact | source |
|---|---|
| 279,650 rows, 57 features, 854 init dates across JJAS 2016–2022 | `data/processed/dataset.parquet`, `model/train.py` |
| Existing ENS: 2022 every 3rd date (41), 2019 and 2020 every 6th | `data/raw/wb2/ens/` |
| ENS store `ifs_ens/2018-2022-240x121…` holds **all 122 JJAS 00Z dates** for every year 2018–2022 | queried the store |
| Store leads are 6-hourly (61 steps), so daily leads 1–10 touch **6 chunks** per date, **≈123 MB per date** at ~20.5 MB/chunk | queried the store; chunk size from `ingest/wb2.py` |
| Both downstream scripts glob and concatenate every `ens_spread_*.parquet` — overlapping files would **silently double-count dates** | `compute_confidence_intervals.py:113`, `evaluate_ens_baseline.py` |
| Calibrated-ENS fit currently pools train + val ENS rows; the model is calibrated on 2021 alone | `evaluate_ens_baseline.py:73`, `config.py` |
| Published intervals used 2,000 resamples, seed 20260919 | `evaluate/uncertainty.py:45-46` |
| `landing.html:215` hardcodes the 40-date margin; its four stat cards are hardcoded too | `web/landing.html` |
| **The published margin scores the frozen `bust_model.joblib` directly on dataset rows**; it never reads `test_predictions.parquet` (which holds 2022 only) | `compute_confidence_intervals.py:140-141` |
| The raw model returns a probability for every row, including out-of-distribution rows the served product refuses | same |
| The ENS comparator is chosen on the test rows: the stronger of raw and relative spread by AUROC | `compute_confidence_intervals.py:151-153` |
| 380 GB free on disk | `df -h` |

A cost correction made during design: the first estimate for the remaining 2022
dates was 3–4 GB, on an assumption of 2 chunks per date. Measured, it is 6
chunks, so ≈10 GB at all ten leads.

## 4. Scope

**Fetch:** all remaining 2022 dates (81) at all ten leads, then all 122 dates of
2021 at all ten leads. ≈10 GB + ≈15 GB = **≈25 GB** of transfer. Only reduced
per-subdivision statistics are written to disk.

**In:** the audit, the shared loader, the resumable fetch, the registration,
`settle_ens.py`, the verdict's consequences in docs and on the landing page.

**Out:** retraining, new features, new models, anything in S2–S5.

## 5. The registered analysis

Lives in `docs/PREREGISTRATION_S1.md`, committed **with `settle_ens.py`** and
pushed before the fetch runs.

### 5.1 Frozen inputs

- **Model:** `data/artifacts/bust_model.joblib`, scored directly on dataset rows,
  exactly as the published margin was. **Data:** `data/processed/dataset.parquet`.
  The SHA-256 of both is recorded in the registration; `settle_ens.py` refuses to
  run if either differs. The model is not retrained during S1; 2021 probabilities
  for secondary b come from the same frozen file by inference alone.
  (An earlier draft of this spec named `test_predictions.parquet`. The audit of
  the code showed the published number never used it.)
- **ENS spread:** the raw standard deviation across all 50 members of each
  subdivision's area-mean 24-hour rainfall, as `fetch_ens.py` computes it. No
  transform. (The relative spread, used by mistake once, scored AUROC 0.554 and
  produced a flattering margin of +0.28.)

### 5.2 Primary — one test, one verdict

| | |
|---|---|
| quantity | model AUROC − raw ENS-spread AUROC |
| rows | 2022 JJAS, Day 3–7, where ENS spread and the label exist — the rule in `ens_margin()` (`compute_confidence_intervals.py`), confirmed by the audit. The model scores **every** such row, including rows the served product would refuse as out-of-distribution: this measures ranking skill, not the served subset, and is stated as such |
| comparator | the stronger, by AUROC on the test rows, of raw spread and relative spread — the published rule, kept because it is conservative (the model always faces the stronger competitor). With the 40-date data this is raw spread, 0.807 vs 0.554 |
| interval | paired cluster bootstrap over **init dates**; 10,000 resamples; seed 20260919; 95% percentile |
| requires | all 122 init dates of 2022 present |
| verdict | lower > 0 → **the model outranks raw ENS spread over the full held-out season**; upper < 0 → **ENS spread outranks the model**; otherwise → **not distinguishable on 122 dates** |

The third outcome is final for this season. No re-slicing to find a significant
subset afterwards.

10,000 resamples at the same seed, versus the published 2,000, reduces Monte
Carlo noise in the endpoints. It is not a re-roll: the audit reproduces the
published number at 2,000 first.

### 5.3 Secondary — reported, cannot overturn the primary

- **a. Calibrated ENS vs model.** ENS isotonic fitted on **2021 only**, matching
  the model's own calibration year (D-010). ΔBrier, ΔBSS, Δ decision cost, each
  with a paired interval.
- **b. Does the model add anything on top of ENS?** Logistic regression on
  `[logit(p_model), log1p(ens_spread)]`, fitted on 2021's Day 3–7 rows, tested on
  2022 against raw ENS alone (paired ΔAUROC) and against the model alone, with the
  primary's bootstrap. Caveat, stated in the output: the model's isotonic
  calibration was fitted on 2021, so the combination is fitted on probabilities
  that are in-sample for calibration. The evaluation on 2022 remains out of sample.
- **c. Continuity.** The old pooled calibration (2019 + 2020 + 2021), so existing
  published numbers can be compared.

### 5.4 Exploratory — labelled, no claims drawn

- Margin by lead day (1–10) and by month.
- **The original 40 dates vs the 81 new dates.** Tests whether the earlier
  subsample was representative. A large difference is reported as a finding.

## 6. Data pipeline

- **Shards:** `data/raw/wb2/ens/shards/<year>/<YYYY-MM-DD>.parquet`, one per init
  date, all leads × subdivisions. Written to `.tmp` and renamed, so an interrupted
  write never looks complete.
- **Resumable:** a date is done if its shard exists or it is inside a legacy
  file. Legacy files are never modified.
- **Order:** 2022, then 2021.
- **Retries:** five attempts per date, exponential backoff 2 → 32 s; failures
  logged and skipped; a closing summary lists missing dates.
- **Per-shard integrity:** exactly 50 members (the count is stored; a short date
  is flagged, never silently averaged); per-subdivision NaN counts.
- **Determinism check:** two dates from the legacy every-3 file are re-fetched to
  scratch and compared value by value (≈250 MB).
- **Shared loader** `fbd.evaluate.ens.load_ens()`, used by
  `evaluate_ens_baseline.py`, `compute_confidence_intervals.py` and
  `settle_ens.py`. Deduplicates on `(subdivision_id, init_date, lead_day)`:
  equal duplicates keep one, **conflicting duplicates raise**. Reports dates per
  year.
- Runs in the background with a progress log; the ETA is measured from the first
  dates, not estimated.

## 7. Audit (runs first)

1. Reproduce +0.0251 [−0.0083, +0.0580] exactly (2,000 resamples, seed
   20260919) through the new loader. **A mismatch stops S1.**
2. Write down the row rule from the code.
3. Record the SHA-256 of `bust_model.joblib` and `dataset.parquet`.
4. Re-derive every published number the verdict touches: 0.832 / 0.807 / 0.731 on
   6,732 rows; calibrated 0.792 and +0.0403.
5. Record doc–code discrepancies and fix them. Known so far:
   - `docs/FRONTEND_BUILT.md` claims every landing-page claim reads live from the
     API, which is false for the ENS sentence and the four stat cards.
   - The README describes the ENS comparison as "on identical rows" without saying
     the model scores rows the product would refuse; the row rule gets stated.

Audit results are written into the registration as "baseline reproduced before
registration".

## 8. Outputs

- `data/artifacts/ens_settlement.json` — registration hash, model and dataset
  hashes, dates and rows, primary interval and verdict, secondaries, exploratory
  results.
- `docs/figures/ens_settlement.{png,svg}` — all intervals on one axis, zero marked.
- `/api/metrics` gains the settlement block (null when the file is absent).
- `landing.html` reads the ENS sentence and the four stat cards from
  `/api/metrics`; missing values render "—" with a note.

## 9. Consequences, committed before the data

| verdict | README / landing / FRONTEND_LOGIC §8 | D-025 | S2–S5 |
|---|---|---|---|
| lower > 0 | "outranks a real 50-member ensemble over the full held-out season, +x [lo, hi], 122 dates" moves to *may claim* | LOCKED | build on an established edge |
| contains 0 | "not distinguishable from a real 50-member ensemble over a full season"; "comparable" stays, now evidenced | DISCLOSED | S3's target becomes model + ENS vs ENS alone (secondary b) |
| upper < 0 | "ENS spread outranks the model over the full season", stated plainly; value proposition rewritten around secondary b, or its absence | DISCLOSED | reopen the product question before building anything else |

## 10. Tests

All synthetic, so they run in CI without network or data.

| file | proves |
|---|---|
| `tests/test_ens_loader.py` | equal duplicates keep one; conflicting duplicates raise; legacy + shards combine; per-year counts |
| `tests/test_fetch_ens_resume.py` | to-fetch = all − legacy − shards; `.tmp` then rename; short member count flagged |
| `tests/test_settle_ens.py` | refuses on a model or dataset hash mismatch, on < 122 dates, on an uncommitted registration; verdict mapping for all three outcomes; same seed → same result |
| `tests/test_web_pages.py` | no hardcoded margin or stat-card values in `landing.html`; reads `/api/metrics` |
| API test | settlement block present when the file exists, null when absent |

New test files are added to the CI test list.

## 11. Failure handling

- Network failure: retries, then logged and skipped; the primary refuses to run
  on an incomplete season.
- Interrupted write: `.tmp` + rename means no partial shard is ever counted.
- Changed model or dataset: hash mismatch → refusal.
- Changed plan: `settle_ens.py` refuses to run with an uncommitted registration,
  and records the registration's hash in its output.
- Missing results on the landing page: "—" with a note, never a stale number.

## 12. Commit order

```
1  shared loader + resumable fetch + audit + tests        (no new data)
2  PREREGISTRATION_S1.md + settle_ens.py, pushed          (the analysis exists before the data)
3  fetch 2022, then 2021                                   (background)
4  run settle_ens.py → json, figure, D-025, README,
   landing, FRONTEND_LOGIC §8                              (whichever way it falls)
```
