# S3a — Candidate harness, promotion rule, and the MLP control — design

**Date:** 2026-09-25
**Status:** approved in chat (scope, first piece, promotion rule A, three design sections); awaiting spec review
**Follows:** S1b (`docs/superpowers/specs/2026-09-25-s1b-rolling-origin-backtest-design.md`, D-026)

---

## 1. The question

> On identical inputs, does a neural network rank busts better than the
> XGBoost model, across four held-out seasons?

S3 was decomposed in chat into three candidates — S3a an MLP on the existing
features, S3b a temporal sequence model, S3c a spatial CNN on the gridded
fields — each scored under one promotion rule registered here, before any
candidate exists. The MLP comes first because it is the control: if S3b or S3c
later beats XGBoost, the MLP result says whether the architecture or the new
information did it.

## 2. Facts verified before designing

| fact | source |
|---|---|
| The four S1b strict fold datasets and XGBoost fold models are on disk; every SHA-256 matches `backtest.json` | `data/processed/backtest/`, checked |
| Incumbent per-year AUROC on the 20,060 comparison rows: 0.7829 / 0.8415 / 0.8154 / 0.8408 (2019–2022) | `backtest.json` |
| 52 features; 11 have structural missing values: the lagged-ensemble group (`lagged_spread`, `lagged_spread_rel`, `lagged_range`, `jumpiness`, `fcst_prev_run`) 10.4% — no lagged members at Day 10 or season edges — and the tendency features (`*_d1`, `*_d3`) 0.85–2.6% — undefined on the first days of a season | fold 2022 dataset |
| XGBoost handles missing values natively; a neural network does not | — |
| PyTorch is not installed; RTX 3060 Laptop 6 GB, 16 CPUs, 15 GB RAM | environment |
| ENS covers 2018–2022; complete for 2019–2022 | `E.dates_by_year` |

## 3. The candidate slate and the promotion rule

Registered once, in `docs/PREREGISTRATION_S3.md`, before any candidate is
trained.

- **Slate, declared in full now:** `mlp` (S3a), `temporal` (S3b), `spatial`
  (S3c). No candidate outside the slate can be scored by the harness; adding one
  needs a new registration and widens the correction.
- **Incumbent, fixed for all three:** the S1b strict XGBoost fold models, pinned
  by the SHA-256 of `backtest.json`. A promoted candidate does not become the
  comparator for the next one.
- **Each candidate's architecture and hyperparameters** are frozen by the SHA-256
  of its parameter dictionary: the MLP's in this registration, S3b's and S3c's in
  their own addenda (`PREREGISTRATION_S3B.md`, `…_S3C.md`), each committed
  before that candidate is scored.

### 3.1 Primary, per candidate

| | |
|---|---|
| quantity | mean over **2019, 2020, 2021, 2022** of the within-year margin AUROC(candidate) − AUROC(incumbent) |
| rows | each fold's comparison rows (S1b rule): Day 3–7 test rows with a label and ENS spread, 20,060 per year |
| interval | stratified cluster bootstrap over init dates (S1b); paired; 10,000 resamples; seed 20260919; **percentile interval at 1 − 0.05/3 (98.33%)** — Bonferroni over the three declared candidates |
| verdict | lower > 0 → **promoted**; upper < 0 → **the incumbent outranks the candidate**; otherwise → **not distinguishable from the incumbent** |

Bonferroni, not Holm: candidates are scored weeks apart, and Holm's step-down
cannot decide any candidate until all three p-values exist. All four years are
used because no candidate has been scored on any of them; the incumbent's
per-year results are known (S1b) and that is disclosed (§7).

### 3.2 Secondary — reported, cannot promote

- Per year, candidate − incumbent (2,000 resamples, 95%).
- **Candidate − ENS spread**, mean over 2019–2021 (the S1b primary years, 10,000
  resamples, 95%) and per year, with S1b's per-year rule: a year whose upper
  bound is below zero is stated as "ENS spread outranks the <candidate> in
  <year>". Whatever this shows, it does not reopen D-026, which was registered
  for XGBoost.

### 3.3 Exploratory — labelled, no claims drawn

- Candidate − incumbent by month within each year.
- Seed spread: each seed's own AUROC per year, to show how much the 5-seed
  average smooths.

## 4. The harness

- `scripts/promote.py --candidate NAME` — guards, then for each fold: load the
  strict fold dataset and the incumbent fold model (both hash-checked against
  `backtest.json`), fit the candidate on the fold's train and validation years,
  score the comparison rows, and write `data/artifacts/candidates/NAME.json`.
- Candidates implement the `BustModel` interface: `fit(train, val, features)`,
  `predict_proba(df)`, `save(path)`. A registry maps slate names to classes;
  S3a registers only `mlp`, and the harness refuses slate names without a
  registered parameter hash.
- `fbd.evaluate.backtest_stats.mean_margin` gains an `alpha` argument (default
  0.05) so the primary can use 0.05/3 through the existing stratified bootstrap.
- Guards, in order: registration committed as-is; `backtest.json` hash; every
  fold dataset and incumbent model hash from it; candidate in the slate;
  candidate parameter hash; all four ENS years complete.

## 5. The MLP, frozen now

| | |
|---|---|
| inputs | the 52 features of the incumbent, in the same order |
| missing values | per feature: training-rows median; plus a 0/1 missing indicator for every feature with any missing value in the fold's training rows |
| scaling | mean and standard deviation from training rows, after imputation; a zero SD becomes 1 |
| network | inputs → 128 → 64 → 1; ReLU; dropout 0.2 after each hidden layer |
| loss | binary cross-entropy with `pos_weight` = negatives / positives on the training rows (mirrors XGBoost's `scale_pos_weight`) |
| optimiser | AdamW, learning rate 1e-3, weight decay 1e-4, batch 1,024, shuffled by a seeded generator |
| stopping | at most 60 epochs; stop after 5 epochs without improvement in validation-year loss (same weighted BCE); restore the best epoch |
| seeds | 5 networks, seeds 20260920 + 0…4; their probabilities averaged |
| calibration | isotonic on the validation year, fitted to the averaged probability — as `BustModel` does |
| device | CPU; `torch.use_deterministic_algorithms(True)`; 8 threads; float32 |

The validation year is used twice — early stopping and calibration — as it is
for any early-stopped model; no test-year row is touched by fitting. CPU keeps
training bit-reproducible and keeps "no GPU is used anywhere" true; the GPU
question belongs to S3c. Torch is imported lazily inside the MLP module, so
nothing that CI or the serving image imports depends on it.

## 6. Audit (runs first, before registration)

1. **Incumbent reproduction:** the harness, scoring the S1b fold models on the
   comparison rows, reproduces every per-fold model AUROC and ENS AUROC in
   `backtest.json` exactly. A mismatch stops S3a.
2. **MLP determinism:** fitting the fold-2019 MLP twice gives bit-identical
   validation-year probabilities. Uses training and validation years only.
3. **MLP smoke test:** its validation-year AUROC on fold 2019 is recorded, as
   evidence the pipeline learns. Below 0.60 stops S3a as a defect. The audit may
   fix defects; it may not change §5.

Results go into the registration.

## 7. Blinding, stated honestly

- No MLP has been trained; no candidate prediction exists for any test year.
- The incumbent's per-year results (D-026) and the late-season weakness are
  known. The MLP was specified after them, with identical inputs and an
  architecture fixed without looking at any MLP result.
- The audit looks at fold 2019's validation year (2018) only.

## 8. Consequences, committed before the data

| MLP primary | recorded as | README / FRONTEND_LOGIC §8 |
|---|---|---|
| promoted | the neural architecture outranks trees on identical inputs; S3b and S3c results are read against it | README "Other model families" states it; §8: "an MLP outranks the served XGBoost in a registered backtest; it is not served" |
| not distinguishable | architecture alone adds nothing detectable; any S3b/S3c gain is attributable to information | README states it; §8: "no neural model is served; an MLP was tested and was not distinguishable (D-027)" |
| incumbent outranks | trees beat the network on these inputs | README states it; §8: "no neural model is served; an MLP was tested and XGBoost outranked it (D-027)" |

In every row the served product is unchanged; promotion into the product is a
separate decision after S3c.

## 9. Pipeline changes

| file | change |
|---|---|
| `src/fbd/model/params.py` | `MLP_PARAMS` (the §5 table as a dict); `params_sha256` already generic |
| `src/fbd/model/mlp.py` | `MLPModel` — fit / predict_proba / save / load; torch imported inside |
| `src/fbd/evaluate/promotion.py` | pure: `SLATE`, `ALPHA_FAMILY`, `alpha_per_candidate()`, `verdict()`, `VERDICT_TEXT`, `check_candidate()` |
| `src/fbd/evaluate/backtest_stats.py` | `mean_margin(..., alpha=0.05)` |
| `scripts/audit_s3a.py` | §6 → `data/artifacts/s3a_audit.json` |
| `scripts/promote.py` | the harness → `data/artifacts/candidates/NAME.json` |
| `scripts/plot_candidate.py` | per-year and primary intervals → `docs/figures/candidate_NAME.{png,svg}` |
| `docs/PREREGISTRATION_S3.md` | the registration |
| `requirements.txt` | `torch` (CPU build) — not `requirements-dev.txt` or `requirements-serve.txt` |

## 10. Tests

| file | CI? | proves |
|---|---|---|
| `tests/test_promotion.py` | yes | slate is exactly mlp/temporal/spatial; per-candidate alpha is 0.05/3; verdict mapping for all three outcomes; undeclared candidate refused; MLP params hash stable and sensitive |
| `tests/test_backtest_stats.py` (extended) | skips | `mean_margin` with a smaller alpha gives a wider interval around the same point |
| `tests/test_mlp.py` | skips without torch | same seed → identical fit; imputation and scaling use training rows only (poison test); missing indicators exactly for features missing in training; probabilities in [0, 1] and calibrated monotone; save/load round trip |

## 11. Failure handling

- Any guard fails → `REFUSED: …`, exit 2, nothing trained.
- Incumbent reproduction or MLP determinism fails → stop before registration.
- A fold fails to train → exit non-zero before any statistic; no partial JSON.

## 12. Commit order

```
1  params + promotion rule (pure) + tests
2  MLPModel + tests (torch installed)
3  mean_margin alpha, promote.py, plot_candidate.py
4  audit → s3a_audit.json
5  PREREGISTRATION_S3.md, CI list — pushed                  (before any MLP is scored)
6  promote.py --candidate mlp, once → json, figure
7  D-027, README, FRONTEND_LOGIC §8, consistency test; final audit; push
```
