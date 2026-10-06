# Pre-registration — S3c: the spatial candidate (addendum to S3)

**Registered:** 2026-10-06, before any spatial model produced a prediction for
any of 2019–2022.
**Code:** `src/fbd/model/{grids,spatial,params}.py`, `scripts/promote.py`,
`scripts/audit_s3c.py`, committed before this file.
**Design:** `docs/superpowers/specs/2026-09-28-s3c-spatial-candidate-design.md`
**Parent:** `docs/PREREGISTRATION_S3.md`. The slate, the incumbent, the primary
and the Bonferroni level are S3's and are not restated as new choices here.
`promote.py` merges this file's registration block into S3's and refuses a key
that would override one of S3's.

---

## 1. The question

> Does a CNN that also sees the forecast's spatial pattern around the
> subdivision and the synoptic map at issue time rank busts better than the
> XGBoost model, across 2019–2022?

S3c is `spatial`, the last of the three candidates S3 declared. The
incumbent's 52 inputs see the forecast only as a subdivision average and the
atmosphere only through hand-built indices. S3a found that the architecture
alone helps on average (D-027, MLP − XGBoost +0.0088 [+0.0023, +0.0157]); S3b
found that 14 days of history adds nothing (D-031). D-027 requires S3c to be
read against the MLP: a gain counts as *spatial information* only if S3c also
clears the MLP.

## 2. Reproduced before registration

`scripts/audit_s3c.py`, output `data/artifacts/s3c_audit.json`.

| check | result |
|---|---|
| the changed harness re-scores the MLP (into a scratch directory) and reproduces `mlp.json` | **exact** in all four folds: candidate, incumbent and ENS AUROC (MLP 0.814818 / 0.846506 / 0.839082 / 0.815320) |
| the same for the temporal model and `temporal.json` | **exact** in all four folds (0.814928 / 0.844923 / 0.830162 / 0.814809) |
| the pinned S3a and S3b fold models | untouched; SHA-256 unchanged |
| poisoning every run issued after t and every analysis after t 00Z, on 20 real 2018 rows | **0 violations** in the window and the map |
| the GPU's window cut against the reference (numpy) cut, same 20 rows | identical (largest gap 0.0) |
| two fits of the fold-2019 spatial model (train 2016–2017, validation 2018), on the GPU | **bit-identical** validation-year output; 73 s and 55 s per fit |
| that model's validation-year (2018) AUROC | **0.8307** (defect floor 0.60) |
| the 23 gridded inputs | hashed into `s3c_inputs.json` |

The five seeds reached their best validation loss at epochs 2, 3, 2, 2, 1:
the same near-immediate peak the MLP and the temporal model showed. That is
recorded, not tuned; §3 is fixed. Environment: torch 2.14.0+cu126, CUDA 12.6,
NVIDIA GeForce RTX 3060 Laptop GPU, Python 3.10.11, Windows.

## 3. The candidate, frozen

`fbd.model.params.SPATIAL_PARAMS`, hash below. Everything is the spec's,
with one amendment, made before any spatial model was built or scored.

**The amendment: the device.** The spec said CPU. On the owner's request to
use the local GPU, the registered device is **CUDA** (an NVIDIA GeForce RTX
3060 Laptop GPU, torch 2.14.0+cu126). It runs deterministically:
deterministic algorithms, cuDNN deterministic, TF32 off (true float32), a
fixed cuBLAS workspace, and weights initialised on the CPU before the move.
The adaptive pool is written as two pooling matrices, because PyTorch's
built-in adaptive pool has no deterministic backward pass on the GPU; a test
holds the two equal. The audit's two fits are bit-identical (§2). A result
produced on another device is not the registered result, and the model
refuses to run on a CPU silently.

**The window branch:** 13 × 13 cells of the HRES 0.703° grid (~1,000 km),
centred on the subdivision's area-weighted centre (rounded half up), zero-
padded outside the HRES domain. 10 channels:

| channel | content | scaling |
|---|---|---|
| forecast, this run | log(1 + tp24) of the run issued on t, lead L (the forecast being judged) | standardised |
| forecast, previous run | log(1 + tp24) of the run issued on t − 1, lead L + 1 (the same target day) | standardised; 0 when absent |
| previous run present | 1 where that run's value is present (not Day 10, not 1 June, in the domain) | unscaled |
| outline | fraction of each cell's area inside s, from the cached weights | unscaled |
| orography | ERA5 surface geopotential, bilinear to 0.703° | standardised |
| land–sea | ERA5 land–sea mask, bilinear | unscaled |
| in domain | 1 inside the HRES domain, 0 in the padding | unscaled |
| TCWV, u850, v850 | ERA5 at t 00Z, bilinear to 0.703° | standardised |

"Standardised": per channel, with the mean and SD of the present, in-domain
entries of the training rows' windows; absent entries are 0 after scaling.

**The map branch:** the ERA5 map at t 00Z, 37 × 47 at 1.5° over −9–45°N,
40.5–109.5°E. TCWV, MSLP, u850, v850, q850, z500 and u200 as anomalies (each
cell's mean over the training issue days removed, divided by the channel's
SD of those anomalies); orography standardised; the land–sea mask unscaled.

**The network and its training.**

| | |
|---|---|
| static inputs | S3a's preprocessing of the incumbent's 52 features, unchanged |
| window encoder | conv 3×3 (10→16), ReLU, conv 3×3 (16→32), ReLU, max-pool 2, conv 3×3 (32→32), ReLU, global average pool → 32. Padding 1 |
| map encoder | conv 3×3 (9→16), ReLU, max-pool 2, conv 3×3 (16→32), ReLU, max-pool 2, conv 3×3 (32→32), ReLU, average pool to 4 × 4, linear 512→32, ReLU → 32. Padding 1 |
| head | [static ; window 32 ; map 32] → 128 → 64 → 1, ReLU, dropout 0.2 after each hidden layer |
| training | BCE with `pos_weight` = negatives / positives on training rows; AdamW lr 1e-3, weight decay 1e-4; batch 1,024, seeded shuffle; ≤ 60 epochs, patience 5 on validation-year weighted BCE, best epoch restored |
| seeds | 20260920–20260924, probabilities averaged |
| calibration | isotonic on the validation year, on the averaged probability |

**Variants.** The same class with branches (window) and (map) gives
**window-only** and **map-only**. The head's input drops the missing branch's
32 values. Everything else is identical.

**Inputs pinned file by file.** `data/artifacts/s3c_inputs.json` lists the
SHA-256 of the 23 gridded files S3c reads (HRES × 7, ERA5 2-D × 7, ERA5 3-D ×
7, ERA5 static, the HRES weights). This file pins that manifest's own hash,
and `promote.py` refuses if any listed file differs.

## 4. Primary

S3's §4, unchanged. The quantity is the mean over **2019, 2020, 2021, 2022** of
the within-year margin AUROC(spatial) − AUROC(incumbent). The rows are S1b's
comparison rows (Day 3–7 test rows with a label and ENS spread, 20,060 a
year). The interval is a stratified, paired cluster bootstrap over init dates:
**10,000** resamples, seed **20260919**, percentile interval at **98.33%**.
Verdict: lower > 0 → **promoted**; upper < 0 → **the incumbent outranks the
spatial model**; otherwise → **not distinguishable from the incumbent**.

## 5. Secondary — reported, cannot promote

- **Spatial − MLP**, using the pinned S3a fold models (each verified by
  SHA-256 and AUROC against `mlp.json` before use). Mean over 2019–2022
  (10,000 resamples, 95%) and per year (2,000, 95%). D-027's check.
- **Window-only − XGBoost** and **map-only − XGBoost**, each as a mean over
  2019–2022 (10,000, 95%) and per year (2,000). These show which image
  carries any gain.
- **Spatial − ENS spread**, S3's rule: mean over 2019–2021 (10,000, 95%) and
  per year (2,000).
- Spatial − incumbent per year (2,000, 95%).

## 6. Exploratory — labelled, no claims drawn

- **Group shuffle.** On each fold's test rows, the AUROC with the windows
  permuted across rows, and then the maps permuted across issue days (seed
  20260919). Not an ablation; nothing is refitted.
- Spatial − incumbent by month; each seed's AUROC per year; each variant's
  seed AUROC.

## 7. Blinding

- No spatial model has produced a prediction for 2019, 2020, 2021 or 2022.
- Known in advance: the incumbent's per-year results (D-026), the MLP's
  (D-027, D-028) and the temporal model's (D-031), including that networks so
  far lose to XGBoost in 2022. The inputs and the architecture were fixed in
  chat on 2026-09-28, before any code existed, without any spatial result.
- The audit, and one two-epoch timing fit made while building it, looked at
  fold 2019's validation year (2018) only.

## 8. Consequences, committed now

| primary | spatial − MLP (mean) | recorded as |
|---|---|---|
| promoted | lower > 0 | the forecast's spatial pattern or the synoptic map adds information beyond being a network. The variants say which |
| promoted | interval contains 0, or upper < 0 | the gain is the network, not the images |
| not distinguishable | — | the images add nothing detectable over XGBoost |
| incumbent outranks | — | the XGBoost incumbent outranks the spatial model |

In every row, README "Other model families" states the result, and
`FRONTEND_LOGIC.md` §8 keeps saying what is *served*. Nothing is served. S3c
completes the slate; whether anything from S3 should change the served model
(for example the MLP inside the served combination, D-029/D-030) is a separate
decision (backlog B5), taken with S3's results in hand.

## 9. What would invalidate this

- Changing `SPATIAL_PARAMS`, `mlp.json`, `s3c_inputs.json` or any file it
  lists, `backtest.json`, the fold files, this file or S3's after the spatial
  model has been scored. `promote.py` refuses each of these.
- Changing the harness, the grid builder or the model code after scoring.
- Scoring before this file is committed, or running the primary more than once.
- Promoting on a secondary, per-year, variant or exploratory result.

## 10. Registration block

Read by `promote.py` and merged into S3's. Do not edit after registration.

```registration
spatial_params_sha256: 3653d1de7227acc48c2347dd988e4171abb315bec2de5ce9ac5e75660732b2aa
mlp_json_sha256: b8d8b911cee6f2e157c9e27045c9aae1dd87f0bd58ea3e1669ab755ac9b33ae1
spatial_inputs_sha256: 1152df9f60a756cded24557d32e0034cd9bad0b76dd542bcea7bc981d1adab6b
```
