# S3b — the temporal candidate: a GRU over 14 days of history — design

**Date:** 2026-09-27
**Status:** approved in chat (sequence content: verification + weather; architecture A; all seven sections); awaiting spec review
**Follows:** S3 registration (`docs/PREREGISTRATION_S3.md`), S3a (D-027), S3a-C (D-028)

---

## 1. The question

> Does a network that also reads the last 14 days rank busts better than the
> XGBoost model, across 2019–2022?

S3 declared three candidates before any was built. S3b is `temporal`. S3a
showed that the architecture alone helps on average: the MLP on the incumbent's
52 inputs outranks XGBoost by +0.0088 [+0.0023, +0.0157] (D-027), although not
in 2022. D-027 requires S3b and S3c to be read against that. A temporal gain
counts as *history* only if it also clears the MLP.

## 2. Facts verified before designing

| fact | source |
|---|---|
| A forecast issued on day t at lead L verifies on t + L − 1 | every fold dataset: `valid − init − lead = −1` on all rows |
| IMD's rainfall day runs 03Z → 03Z (D-005), so the last observation complete at a 00Z issue on day t is for day t − 2 | D-005 |
| `error` = forecast − observed; bust thresholds are fitted on each fold's own training years | `fbd.labels.bust.label` |
| The 17 weather fields (8 subdivision ERA5 fields, 9 national `*_z` indices) take one value per (subdivision, issue day) and are never missing | fold 2022 dataset |
| Each fold dataset holds every year up to its test year, all splits, all ten leads | fold 2019: 2016–2019, 159,800 rows; fold 2022: 2016–2022, 279,650 rows |
| Seasons are 1 June – 30 September only; there is no May data | fold datasets |
| `promote.py` gives a candidate only the rows being scored (Day 3–7 comparison rows), not the fold dataset | `scripts/promote.py` |
| The S3a fold models are saved and pinned by SHA-256 in `data/artifacts/candidates/mlp.json` | `data/processed/backtest/candidate_mlp_<year>.joblib` |
| S3a fit times: 12–32 s per fold for five seeds, CPU | `mlp.json` |

## 3. The sequence

One sequence per (subdivision, issue day t). There are 14 daily steps,
d = t − 13 … t, and step d holds exactly what had arrived by the morning of
day d. Every lead's row for that (subdivision, t) shares the sequence; the lead
stays among the 52 inputs.

| group | channels | taken from | cutoff |
|---|---|---|---|
| weather (17) | `moisture_flux_850`, `wind_shear`, `u850`, `v850`, `z500`, `tcwv`, `mslp`, `onshore_wind`, `somali_jet_z`, `monsoon_trough_mslp_z`, `nw_z500_z`, `india_shear_z`, `india_tcwv_z`, `india_q850_z`, `mcz_q850_z`, `bob_vorticity_max_z`, `bob_mslp_min_z` | the row with `init_date = d` | d ≤ t |
| verification (7) | `obs_rain_mm` on day d − 2; for L ∈ {1, 3, 5}, the `error` and `bust` of the forecast at lead L that verified on day d − 2 | rows with `valid_date = d − 2` | observed ≤ t − 2 |
| masks (5) | weather present; observation present; lead-1, lead-3 and lead-5 verification present | — | — |

29 channels in all.

- **Within a season only.** A step whose data falls before 1 June has no row:
  its values are zero and its masks are 0.
- **Scaling.** Each of the 24 value channels is standardised with the mean and
  SD of its *present* entries in the training rows' sequences. Absent entries
  are 0 after scaling. Masks are not scaled.
- **Causality.** The verification row for day d − 2 at lead L was issued on
  d − 1 − L, which is before t. Both cutoffs are enforced by construction and
  tested by poisoning (§8).

## 4. The model

`fbd.model.temporal.TemporalModel`: S3a's model with a history encoder in
front of its head.

| | |
|---|---|
| static inputs | S3a's `Preprocessor` on the incumbent's 52 features, unchanged (median imputation, missing indicators, training-rows standardisation) |
| encoder | one-layer GRU, 32 hidden units, over the 14 steps × 29 channels; its final hidden state |
| head | [static ; GRU state] → 128 → 64 → 1, ReLU, dropout 0.2 after each hidden layer (S3a's head) |
| training | S3a's settings exactly: BCE with `pos_weight` = negatives / positives on training rows; AdamW lr 1e-3, weight decay 1e-4; batch 1,024, seeded shuffle; ≤ 60 epochs, patience 5 on validation-year weighted BCE, best epoch restored |
| seeds | 20260920–20260924, probabilities averaged |
| calibration | isotonic on the validation year, on the averaged probability |
| device | CPU, deterministic algorithms, 8 threads, float32 |

Frozen as `fbd.model.params.TEMPORAL_PARAMS`, which includes the channel lists,
the window (14), the verification lag (2) and the leads (1, 3, 5), so that
changing any of them changes `temporal_params_sha256`.

The model needs the fold dataset to find each row's history, so it takes it at
construction: `TemporalModel(dataset=ds)`, with `needs_dataset = True` on the
class. (`history` is already the per-seed training log that `promote.py`
records, as on the MLP.) `save` stores the weights, the scaler and the
calibrator, not the dataset; `load(path, dataset=ds)` takes it again.

### Alternatives considered

- **Temporal CNN** (two 1-D convolutions and pooling): about the same on 14
  steps, and no reason to prefer it.
- **Small transformer:** too many parameters for about 20,000 unique sequences
  per fold, and the attribution to S3a is weaker.

## 5. Harness changes (additive, proven neutral)

1. **Dataset hook.** `score_fold` builds `cls(dataset=ds)` when the class sets
   `needs_dataset`, and `cls()` otherwise. The MLP path is unchanged.
2. **Output directory.** `score_fold` takes an `out_dir` for the saved fold
   model, which defaults to today's `FOLD_DIR`. The audit writes to a scratch
   directory so it never overwrites the pinned MLP fold models.
3. **Addenda.** `guards` reads the candidate's addendum (`temporal` →
   `docs/PREREGISTRATION_S3B.md`), applies the same committed-and-unchanged
   guard to it, and merges its registration block into S3's. A key present in
   both is refused: an addendum adds and never overrides.
4. **MLP secondary.** For any candidate other than `mlp`, each fold also loads
   `candidate_mlp_<year>.joblib`, checks its SHA-256 against `mlp.json`, and
   requires its AUROC to reproduce `mlp.json` exactly before using it.
5. **Group shuffle (exploratory).** A candidate that declares `SHUFFLE_GROUPS`
   exposes `predict_proba(df, shuffle=<group>, seed=...)`. On the test rows,
   `promote.py` records the AUROC with each group permuted across sequences
   (seed 20260919). The MLP declares none.

The S3 registration counts changing the harness after a scored candidate as
invalidating (§10). So the audit re-scores the MLP through the changed harness
into a scratch directory, and requires every fold's candidate, incumbent and
ENS AUROC to equal `mlp.json` exactly.

## 6. Registration — `docs/PREREGISTRATION_S3B.md`

- **Primary:** S3's, unchanged. The mean over 2019–2022 of the within-year
  AUROC margin over the XGBoost incumbent, on S1b's comparison rows (20,060 a
  year). Stratified cluster bootstrap, 10,000 resamples, seed 20260919,
  percentile interval at 98.33%. Verdict: lower > 0 → promoted; upper < 0 →
  the incumbent outranks it; otherwise not distinguishable.
- **Secondary** (reported, cannot promote):
  - S3b − MLP, as a mean over 2019–2022 (10,000 resamples, 95%) and per year
    (2,000). This is D-027's check.
  - S3b − ENS spread, S3's rule (mean over 2019–2021 and per year).
  - Candidate − incumbent per year (2,000, 95%).
- **Exploratory** (labelled; no claims):
  - Shuffle one channel group (weather or verification) across sequences in
    the test rows and report the AUROC drop, for each group.
  - By month, and each seed's AUROC per year.
- **Registration block:** `temporal_params_sha256` and `mlp_json_sha256`.

## 7. Consequences, committed before the data

| primary | S3b − MLP | recorded as |
|---|---|---|
| promoted | lower > 0 | recent history adds information beyond being a network |
| promoted | interval contains 0, or upper < 0 | the gain is the network, not the history |
| not distinguishable | — | 14 days of history adds nothing detectable over XGBoost |
| incumbent outranks | — | stated as such |

In every row, README "Other model families" states the result and
`FRONTEND_LOGIC.md` §8 keeps saying what is *served*. Nothing is served; promotion
into the product is decided after S3c.

## 8. Tests

| file | CI | proves |
|---|---|---|
| `tests/test_temporal_sequence.py` | yes | poisoning every observation after t − 2 and every weather value after t with 1000 leaves the sequence unchanged; step d holds d's weather and d − 2's verification; steps before 1 June have values 0 and masks 0; a lead missing on d − 2 zeroes only that lead's channels and mask |
| `tests/test_promotion.py` | yes | `TEMPORAL_PARAMS` hash is stable and changes when any entry changes; an addendum key clashing with S3's is refused; a candidate without `needs_dataset` is built with no arguments |
| `tests/test_temporal_model.py` | skips without torch | fit/predict on a small synthetic frame; two fits give bit-identical probabilities; save/load round-trip |

## 9. Audit before registering — `scripts/audit_s3b.py`

Output `data/artifacts/s3b_audit.json`. Registration waits on every check.

1. The harness reproduces `mlp.json` exactly for all four folds (§5), and the
   loaded MLP fold models reproduce their AUROCs.
2. The causality poison holds on the real fold 2019 dataset.
3. Two fits of the fold 2019 temporal model (train 2016–2017, validation 2018)
   give bit-identical validation-year output. Its validation AUROC is recorded,
   with a defect floor of 0.60, and so is the fit time.

The audit looks at 2018 only. No temporal model produces a prediction for 2019–2022
before registration.

## 10. Blinding

- No temporal model has been built or scored.
- Known in advance: the incumbent's per-year results (D-026), the MLP's
  (D-027) and its 2018 confirmation (D-028). The sequence content and the
  architecture were fixed in chat before any code existed, without looking at
  any temporal result.

## 11. Commit order

```
1  params (TEMPORAL_PARAMS) + fbd.model.temporal (sequence builder, model) + tests
2  harness: history hook, out_dir, addenda, MLP secondary + tests
3  audit_s3b.py; run it; commit s3b_audit.json
4  PREREGISTRATION_S3B.md; commit and push; CI green
5  promote.py --candidate temporal, once; commit candidates/temporal.json + figure
6  D-031, README, FRONTEND_LOGIC §8, HANDOFF; push
```
