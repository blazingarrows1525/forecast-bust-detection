# S3c — the spatial candidate: CNNs on the forecast window and the monsoon map — design

**Date:** 2026-09-28
**Status:** approved in chat (inputs: local window + whole-region map, with each alone as a secondary; architecture A; stacked on S3b); awaiting spec review
**Follows:** S3 registration (`docs/PREREGISTRATION_S3.md`), S3a (D-027), S3b (D-031)

---

## 1. The question

> Does a CNN that also sees the forecast's spatial pattern around the
> subdivision and the synoptic map at issue time rank busts better than the
> XGBoost model, across 2019–2022?

S3c is `spatial`, the last of the three candidates S3 declared. The
incumbent's 52 inputs see the forecast only as a subdivision average and the
atmosphere only through hand-built indices. S3b found that 14 days of history
adds nothing (D-031). D-027 requires every candidate to be read against the
MLP: a gain counts as *spatial information* only if S3c also clears the MLP.

## 2. Facts verified before designing

| fact | source |
|---|---|
| HRES 24 h forecast rain is on disk for 2016–2022: 122 issue days × 10 leads per season, a 45 × 49 grid at 0.703° over 6.7–37.6°N, 66.1–99.8°E, in mm | `data/raw/wb2/hres/hres_tp24_india_<year>.nc` |
| ERA5 is on disk for 2016–2022, 6-hourly, on a 37 × 47 grid at 1.5° over −9–45°N, 40.5–109.5°E. It has TCWV and MSLP, plus geopotential, specific humidity, u and v at 200/500/850 hPa, and static orography and land–sea mask. Every season has all 122 00Z analyses | `data/raw/wb2/era5/era5_{2d,3d}_<year>.nc`, `era5_static.nc` |
| Area weights from each HRES cell to each subdivision are cached. Averaging the grid with them reproduces the dataset's `fcst_rain_mm` exactly (Assam & Meghalaya, issue day 2022-06-14, Day 4: 83.69 mm both ways) | `data/interim/weights_45x49_…parquet` |
| The 34 modelled subdivisions span at most 10 × 12 HRES cells. A 13 × 13 window centred on each subdivision's area-weighted centre contains all of it, and only 2 windows cross the domain edge | computed |
| Both grids have latitude ascending, and the NetCDF variables are stored (time, [level/lead], longitude, latitude) | files |
| S3b's scoring-script changes (dataset hook, addenda, MLP secondary, group shuffle) exist on `s3b-temporal` (PR #3, open) | branch |

## 3. What the network sees

One example per row (subdivision s, issue day t, lead L). Nothing issued or
analysed after t 00Z is used.

### Window branch: 13 × 13 cells of the HRES grid (~1,000 km), centred on s

| channel | content | scaling |
|---|---|---|
| forecast, this run | log(1 + tp24) of the run issued on t, lead L (the forecast being judged) | standardised |
| forecast, previous run | log(1 + tp24) of the run issued on t − 1, lead L + 1 (the same target day) | standardised; 0 when absent |
| previous run present | 1 where that run exists (not Day 10, not 1 June) | unscaled |
| outline | fraction of each cell's area inside s, from the cached weights | unscaled |
| orography | ERA5 surface geopotential, interpolated bilinearly to 0.703° | standardised |
| land–sea | ERA5 land–sea mask, interpolated | unscaled |
| in domain | 1 inside the HRES domain, 0 in the zero-padding | unscaled |
| TCWV, u850, v850 | ERA5 at t 00Z, interpolated to 0.703° | standardised |

10 channels. "Standardised" means per channel, using the mean and SD of the
present, in-domain entries of the training rows' windows; absent entries are
0 after scaling.

### Map branch: the ERA5 monsoon-region map at t 00Z (37 × 47 at 1.5°)

The region covers the Somali jet, the Arabian Sea, the Bay of Bengal and the
Tibetan plateau.

| channels | scaling |
|---|---|
| TCWV, MSLP, u850, v850, q850, z500, u200 | anomaly: subtract each cell's mean over the training issue days, then divide by the channel's SD of those anomalies |
| orography | standardised per channel |
| land–sea mask | unscaled |

9 channels. The map is the same for every subdivision and lead on a given
issue day: at most 610 distinct maps in a fold's training years.

### Causality

The two runs used are issued on t and t − 1, and the analysis is at t 00Z;
the 52 features follow the same "state at issue time" rule (LOGIC.md 7.1). The
test poisons every run issued after t and every analysis after t 00Z with
1000 and requires identical inputs.

## 4. The model

`fbd.model.spatial.SpatialModel(dataset=ds, branches=("window", "map"))`:
S3a's model with two image encoders in front of its head.

| | |
|---|---|
| static inputs | S3a's `Preprocessor` on the incumbent's 52 features, unchanged |
| window encoder | Conv 3×3 (10→16), ReLU, Conv 3×3 (16→32), ReLU, MaxPool 2, Conv 3×3 (32→32), ReLU, global average pool → 32 |
| map encoder | Conv 3×3 (9→16), ReLU, MaxPool 2, Conv 3×3 (16→32), ReLU, MaxPool 2, Conv 3×3 (32→32), ReLU, adaptive average pool to 4 × 4, Linear (512→32), ReLU → 32. The 4 × 4 grid keeps *where* a pattern is |
| head | [static ; window 32 ; map 32] → 128 → 64 → 1, ReLU, dropout 0.2 after each hidden layer (S3a's head) |
| training | S3a's settings exactly: BCE with `pos_weight` = negatives / positives on training rows; AdamW lr 1e-3, weight decay 1e-4; batch 1,024, seeded shuffle; ≤ 60 epochs, patience 5 on validation-year weighted BCE, best epoch restored |
| seeds | 20260920–20260924, probabilities averaged |
| calibration | isotonic on the validation year, on the averaged probability |
| device | CPU, deterministic algorithms, 8 threads, float32 |

**Variants.** The same class with `branches=("window",)` and
`branches=("map",)` gives window-only and map-only. The head's input then
drops the missing branch's 32 values. Everything else is identical.

`fbd.model.params.SPATIAL_PARAMS` freezes all of it: channels, window size,
scalings, both encoders, the head, training, the primary's branches and the
two variants. `spatial_params_sha256` pins it.

**Efficiency.** Windows are cut per batch from padded per-season arrays,
never materialised for every row. Each map is encoded once per distinct issue
day in a batch and gathered to its rows.

### Alternatives considered

- **A pretrained image backbone (e.g. ResNet):** built for photographs, not
  10-channel weather fields; heavy on CPU; weak attribution to S3a.
- **U-Net or ConvLSTM:** far larger than about 600 distinct maps per fold can
  support.

## 5. Scoring-script changes (additive, proven neutral)

Built on S3b's changes.

1. **Variants.** `SPATIAL_PARAMS["variants"]` names window-only and map-only.
   Each fold also fits them, adds `p_window_only` and `p_map_only` to the rows,
   and saves them beside the primary model.
2. **Pinned inputs.** `data/artifacts/s3c_inputs.json` lists the SHA-256 of
   the 23 gridded files S3c reads (HRES × 7, ERA5 2d × 7, ERA5 3d × 7, ERA5
   static, the HRES weights). The addendum pins that manifest's own hash, and
   `guards` refuses if any listed file differs.
3. **Addenda with their own pins.** `ADDENDA[name]` becomes (path,
   {registration key: file}). S3b's entry keeps exactly the pin it has
   (`mlp_json_sha256`); S3c's adds `spatial_inputs_sha256`.
4. **Group shuffle.** The primary declares `SHUFFLE_GROUPS = ("window",
   "map")`: windows are permuted across rows, maps across issue days (S3b's
   mechanism).

Both S3 and S3b count changing the scoring script after a scored candidate as
invalidating. So the audit re-scores **both** the MLP and the temporal model
through the changed script, into a scratch directory. Every fold's candidate,
incumbent, ENS and (for S3b) MLP AUROC must equal `mlp.json` and
`temporal.json` exactly.

## 6. Registration — `docs/PREREGISTRATION_S3C.md`

- **Primary:** S3's, unchanged. The mean over 2019–2022 of the within-year
  AUROC margin over the XGBoost incumbent, on S1b's comparison rows (20,060 a
  year). Stratified cluster bootstrap, 10,000 resamples, seed 20260919,
  percentile interval at 98.33%. Verdict: lower > 0 → promoted; upper < 0 →
  the incumbent outranks it; otherwise not distinguishable.
- **Secondary** (reported, cannot promote):
  - S3c − MLP, as a mean over 2019–2022 (10,000, 95%) and per year (2,000).
    D-027's check.
  - Window-only − XGBoost and map-only − XGBoost, each as a mean over
    2019–2022 (10,000, 95%) and per year (2,000). These show which image
    carries any gain.
  - S3c − ENS spread, S3's rule.
  - S3c − incumbent per year.
- **Exploratory** (labelled; no claims): the AUROC with each branch's input
  shuffled (seed 20260919); by month; each seed's AUROC per year.
- **Registration block:** `spatial_params_sha256`, `mlp_json_sha256`,
  `spatial_inputs_sha256`.

## 7. Consequences, committed before the data

| primary | S3c − MLP | recorded as |
|---|---|---|
| promoted | lower > 0 | the forecast's spatial pattern or the synoptic map adds information beyond being a network. The variants say which |
| promoted | interval contains 0, or upper < 0 | the gain is the network, not the images |
| not distinguishable | — | the images add nothing detectable over XGBoost |
| incumbent outranks | — | the XGBoost incumbent outranks the spatial model |

README "Other model families" states the result, and `FRONTEND_LOGIC.md` §8
keeps saying what is *served*. Nothing is served. S3c completes the slate, so
a separate decision follows on whether anything from S3 should change the
served model, for example the MLP inside the served combination (D-029/D-030).
That decision is taken with S3's results in hand and is not pre-empted here.

## 8. Tests

| file | CI | proves |
|---|---|---|
| `tests/test_spatial_inputs.py` | yes (numpy/xarray only, synthetic grids) | poisoning every run after t and every analysis after t 00Z leaves the window and the map unchanged; the window is centred on the subdivision; the forecast channels equal the (t, L) and (t − 1, L + 1) grid slices; the previous run is absent at Day 10 and on 1 June, with its plane 0; padding and the in-domain plane at the edge; scaling uses present entries only; map anomalies use training days only |
| `tests/test_promotion.py` | yes | the `SPATIAL_PARAMS` hash is stable and changes when any entry changes; the variants are exactly window-only and map-only; an addendum's own pins are verified; a changed manifest file is refused |
| `tests/test_spatial_model.py` | skips without torch | fit and predict on a small synthetic fold for all three branch sets; two fits bit-identical; save and load round-trip; branch shuffles seeded |

## 9. Audit before registering — `scripts/audit_s3c.py`

Output `data/artifacts/s3c_audit.json`. Registration waits on every check.

1. Re-scored through the changed script into a scratch directory, the MLP and
   the temporal model reproduce `mlp.json` and `temporal.json` exactly. The
   pinned fold models are untouched.
2. The causality poison holds on the real 2018 grids for 20 sampled rows.
3. Two fits of the fold-2019 primary (train 2016–2017, validation 2018) give
   bit-identical validation-year output. Its validation AUROC is recorded,
   with a defect floor of 0.60, and so are the fit time and each seed's best
   epoch.
4. `s3c_inputs.json` is written from the files on disk.

The audit looks at 2018 only. No spatial model produces a prediction for 2019–2022
before registration.

## 10. Blinding

- No spatial model has been built or scored.
- Known in advance: the incumbent's per-year results (D-026), the MLP's (D-027,
  D-028) and the temporal model's (D-031), including that networks so far lose
  to XGBoost in 2022. The inputs and the architecture were fixed in chat before
  any code existed, without looking at any spatial result.

## 11. Branch and commit order

`s3c-spatial` is branched from `s3b-temporal` and its PR targets `master`. It
shows #3's commits until #3 merges.

```
1  params (SPATIAL_PARAMS) + fbd.model.grids (inputs, windows, maps, manifest) + tests
2  fbd.model.spatial (encoders, head, variants, shuffle, save/load) + tests
3  scoring script: variants, addenda with their own pins, manifest guard + tests
4  audit_s3c.py; run it; commit s3c_audit.json and s3c_inputs.json
5  PREREGISTRATION_S3C.md; commit and push; CI green
6  promote.py --candidate spatial, once; commit candidates/spatial.json + figure
7  D-032, README, FRONTEND_LOGIC §8, HANDOFF; push
```

The scoring run is about 45–60 minutes of CPU: three models × four folds ×
five seeds.
