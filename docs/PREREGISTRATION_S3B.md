# Pre-registration — S3b: the temporal candidate (addendum to S3)

**Registered:** 2026-09-27, before any temporal model produced a prediction for
any of 2019–2022.
**Code:** `src/fbd/model/{sequence,temporal,params}.py`, `scripts/promote.py`,
`scripts/audit_s3b.py`, committed before this file.
**Design:** `docs/superpowers/specs/2026-09-27-s3b-temporal-candidate-design.md`
**Parent:** `docs/PREREGISTRATION_S3.md`. The slate, the incumbent, the primary
and the Bonferroni level are S3's and are not restated as new choices here.
`promote.py` merges this file's registration block into S3's and refuses a key
that would override one of S3's.

---

## 1. The question

> Does a network that also reads the last 14 days rank busts better than the
> XGBoost model, across 2019–2022?

S3a showed that the architecture alone helps on average: the MLP on the
incumbent's 52 inputs outranks XGBoost by +0.0088 [+0.0023, +0.0157] (D-027),
although not in 2022. D-027 requires S3b to be read against that. A gain counts
as *history* only if S3b also clears the MLP.

## 2. Reproduced before registration

`scripts/audit_s3b.py`, output `data/artifacts/s3b_audit.json`.

| check | result |
|---|---|
| the changed harness re-scores the MLP (into a scratch directory) and reproduces `mlp.json` | **exact** in all four folds: candidate, incumbent and ENS AUROC (MLP 0.814818 / 0.846506 / 0.839082 / 0.815320) |
| the pinned S3a fold models, loaded, reproduce their AUROC; their files are untouched | **exact**; SHA-256 unchanged |
| poisoning every observation after t − 2 and every weather value after t, on 20 real 2018 sequences | **0 violations** |
| two fits of the fold-2019 temporal model (train 2016–2017, validation 2018) | **bit-identical** validation-year output; 49 s per fit |
| that model's validation-year (2018) AUROC | **0.8304** (defect floor 0.60) |

The five seeds reached their best validation loss at epochs 1, 3, 2, 1, 1, the
same near-immediate peak the MLP showed. That is recorded, not tuned; §3 is
fixed.

## 3. The candidate, frozen

`fbd.model.params.TEMPORAL_PARAMS`, hash below.

**The sequence.** For each (subdivision, issue day t), 14 daily steps
d = t − 13 … t. Step d holds what had arrived by the morning of day d:

| group | channels | cutoff |
|---|---|---|
| weather (17) | `moisture_flux_850`, `wind_shear`, `u850`, `v850`, `z500`, `tcwv`, `mslp`, `onshore_wind`, `somali_jet_z`, `monsoon_trough_mslp_z`, `nw_z500_z`, `india_shear_z`, `india_tcwv_z`, `india_q850_z`, `mcz_q850_z`, `bob_vorticity_max_z`, `bob_mslp_min_z`, analysed on day d | ≤ t |
| verification (7) | observed rain on day d − 2; for leads 1, 3, 5, the error (forecast − observed) and bust flag of the forecast that verified on day d − 2 | observed ≤ t − 2 |
| masks (5) | weather present; observation present; lead 1, 3, 5 verification present | — |

d − 2 is the last day whose IMD observation (03Z–03Z, D-005) is complete at a
00Z issue on day d. Steps before 1 June have no data and are masked. Value
channels are standardised with the training rows' present entries; absent
entries are 0.

**The network and its training.**

| | |
|---|---|
| static inputs | S3a's preprocessing of the incumbent's 52 features, unchanged |
| encoder | one-layer GRU, 32 hidden units, over 14 steps × 29 channels; final hidden state |
| head | [static ; GRU state] → 128 → 64 → 1, ReLU, dropout 0.2 after each hidden layer |
| training | BCE with `pos_weight` = negatives / positives on training rows; AdamW lr 1e-3, weight decay 1e-4; batch 1,024, seeded shuffle; ≤ 60 epochs, patience 5 on validation-year weighted BCE, best epoch restored |
| seeds | 20260920–20260924, probabilities averaged |
| calibration | isotonic on the validation year, on the averaged probability |
| device | CPU, deterministic algorithms, 8 threads, float32 |

## 4. Primary

S3's §4, unchanged. The quantity is the mean over **2019, 2020, 2021, 2022** of
the within-year margin AUROC(temporal) − AUROC(incumbent). The rows are S1b's
comparison rows (Day 3–7 test rows with a label and ENS spread, 20,060 a
year). The interval is a stratified, paired cluster bootstrap over init dates:
**10,000** resamples, seed **20260919**, percentile interval at **98.33%**.
Verdict: lower > 0 → **promoted**; upper < 0 → **the incumbent outranks the
temporal model**; otherwise → **not distinguishable from the incumbent**.

## 5. Secondary — reported, cannot promote

- **Temporal − MLP**, using the pinned S3a fold models (each verified by
  SHA-256 and AUROC against `mlp.json` before use). Mean over 2019–2022 (10,000
  resamples, 95%) and per year (2,000, 95%). This is D-027's check.
- **Temporal − ENS spread**, S3's rule: mean over 2019–2021 (10,000, 95%) and
  per year (2,000).
- Temporal − incumbent per year (2,000, 95%).

## 6. Exploratory — labelled, no claims drawn

- **Group shuffle.** On each fold's test rows, the AUROC with the weather group,
  and then the verification group, permuted across sequences (seed 20260919).
  The drop says which history the model leans on. It is not an ablation, and
  nothing is refitted.
- Temporal − incumbent by month; each seed's AUROC per year.

## 7. Blinding

- No temporal model has produced a prediction for 2019, 2020, 2021 or 2022.
- Known in advance: the incumbent's per-year results (D-026), the MLP's (D-027)
  and its 2018 confirmation (D-028). The sequence content and the architecture
  were fixed in chat before any code existed, without any temporal result.
- The audit looked at fold 2019's validation year (2018) only.

## 8. Consequences, committed now

| primary | temporal − MLP (mean) | recorded as |
|---|---|---|
| promoted | lower > 0 | recent history adds information beyond being a network |
| promoted | interval contains 0, or upper < 0 | the gain is the network, not the history |
| not distinguishable | — | 14 days of history adds nothing detectable over XGBoost |
| incumbent outranks | — | the XGBoost incumbent outranks the temporal model |

In every row, README "Other model families" states the result, and
`FRONTEND_LOGIC.md` §8 keeps saying what is *served*. Nothing is served: promotion
into the product is decided after S3c.

## 9. What would invalidate this

- Changing `TEMPORAL_PARAMS`, `mlp.json`, `backtest.json`, the fold files, this
  file or S3's after the temporal model has been scored. `promote.py` refuses
  each of these.
- Changing the harness or the sequence builder after scoring.
- Scoring before this file is committed, or running the primary more than once.
- Promoting on a secondary, per-year or exploratory result.

## 10. Registration block

Read by `promote.py` and merged into S3's. Do not edit after registration.

```registration
temporal_params_sha256: e484ab556b3a5d63af6d75b2ec3eaee0bcadd543c8e7f61b060d731e1733f96e
mlp_json_sha256: b8d8b911cee6f2e157c9e27045c9aae1dd87f0bd58ea3e1669ab755ac9b33ae1
```
