# Feature availability matrix

For every model input and every label field: where it comes from, when it
exists, and whether a live system could know it at the moment of decision.
This is master prompt §9.2, written 2026-10-05 from `src/fbd/model/train.py`,
`src/fbd/features/*`, `src/fbd/labels/bust.py` and `DECISIONS.md`.

The decision time is the bulletin deadline: 03:00 UTC (08:30 IST), with a
second at 06:30 UTC (12:00 IST) (LOGIC.md). The forecast judged is the
00 UTC IFS HRES run of the issue day `T_init`.

**Classes** (master prompt §9.2):
- **FT**: forecast-time input
- **FD**: forecast-derived future-valid-time field, legal because it is part
  of the issued forecast
- **VO**: verification-only label or metric
- **RA**: retrospective-analysis variable
- **SM**: static metadata
- **UV**: unknown until verified. **Not allowed into a live prediction**
  until its timing is confirmed.

## 1. The 52 served inputs, by group

| features | group | raw source / version | variable, units | aggregation | source time → available | class | training transform | missing | potential leakage path | tests |
|---|---|---|---|---|---|---|---|---|---|---|
| `lead_day`, `month`, `day_of_season` | calendar | none | integers | none | known at issue | **SM** | none | never | none | none needed |
| `fcst_rain_mm` | forecast | WB2 IFS HRES, 0.703° (D-003) | `total_precipitation_24hr`, mm/day | area-mean via `weights_*.parquet` (EPSG:7755 overlap) | 00 UTC run → disseminated hours later | **FD** | none | row dropped if absent | none; it is the issued forecast | `test_core.py::test_area_mean_*` |
| `fcst_anomaly`, `fcst_rel_to_p90` | forecast | HRES + training-year climatology | mm/day, ratio | as above | same as the forecast | **FD** | climatology fitted on training years only (`forecast.py:105`) | NaN → XGBoost default branch | climatology fit on all years would leak; it is train-only | `test_fold_leakage.py::test_forecast_climatology_ignores_later_years` |
| `lagged_spread`, `lagged_spread_rel`, `lagged_mean`, `lagged_range`, `lagged_n_members`, `spread_growth` | lagged-ensemble proxy | earlier HRES runs verifying the same day (leads ≥ L) | mm/day | area-mean | issued **before** `T_init` (`forecast.py:11-13`) | **FD** | none | structurally undefined at Day 10 (D-011) → NaN | using lead < L runs would be future runs; excluded by construction | `test_core.py::test_lagged_ensemble_uses_only_leads_at_or_beyond_L` |
| `jumpiness`, `fcst_prev_run` | run-to-run change | the previous day's HRES run for the same valid day | mm/day | area-mean | issued `T_init − 1` | **FD** | none | NaN at the longest lead | none | none dedicated (gap) |
| `clim_obs_mean`, `clim_obs_p90`, `clim_fcst_mean`, `clim_bust_rate` | climatology | IMD obs + HRES + labels, **training years** | mm/day, rate | per subdivision × month (× lead for the bust rate) | fixed before the season | **SM** (from the training period) | fitted on training years only (`forecast.py:122-132`) | backs off to coarser groups | fitting on test years would leak; it is train-only | `test_fold_leakage.py::test_climatological_bust_rate_ignores_later_years` |
| `moisture_flux_850`, `wind_shear`, `tcwv`, `z500`, `mslp`, `u850`, `v850` | ERA5 subdivision state | WB2 ERA5 `1959-2023_01_10`, 1.5° (240×121) | SI units | area-mean per subdivision | 00 UTC analysis of `T_init` → ERA5 published days later | **RA** (stands in for an FT state) | none | NaN → default branch | **ERA5 analysis window**: a reanalysis 00 UTC analysis may assimilate observations from after 00 UTC. A live system must use the operational analysis instead. Treated as **UV** for live use (backlog B2). | no dedicated test of the analysis window (gap) |
| `somali_jet_z`, `monsoon_trough_mslp_z`, `nw_z500_z`, `india_shear_z`, `india_tcwv_z`, `india_q850_z`, `mcz_q850_z`, `bob_vorticity_max_z`, `bob_mslp_min_z` | ERA5 national indices | as above | standardised anomalies | national boxes | as above | **RA** → **UV** for live | z-scored against training-year climatology (`era5.py:145`) | NaN → default branch | same window question; the standardisation is train-only | `test_fold_leakage.py::test_standardise_ignores_rows_outside_the_mask` |
| `somali_jet_d1`, `somali_jet_d3`, `bob_vorticity_max_d1`, `monsoon_trough_mslp_d1`, `nw_z500_d1`, `india_tcwv_d1` | ERA5 tendencies | as above | 1- and 3-day changes | national boxes | uses `T_init` and earlier analyses | **RA** → **UV** for live | differences | NaN at season start | none beyond the window question | none dedicated (gap) |
| `regime_active_monsoon`, `regime_break_monsoon`, `regime_monsoon_depression`, `regime_western_disturbance`, `regime_orographic`, `regime_coastal`, `regime_entropy`, `regime_top_prob` | soft regime | derived from the ERA5 fields above (`regime/classify.py`) | probabilities, entropy | national and local | as ERA5 | **RA** → **UV** for live | softmax of standardised scores | NaN → default branch | **Disclosed leak:** the published (`legacy`) pipeline standardises regime scores over **every date, test year included**. Measured on fold 2022: +0.0009 [−0.0025, +0.0045], no detectable effect. Regime features add no skill (D-012). `strict` mode fits on training years. | `test_fold_leakage.py::test_standardise_within_*` |
| `elevation_m`, `terrain_roughness_m`, `coastal_index`, `orographic_index` | static geography | region build | m, indices | per subdivision | fixed | **SM** | none | never | none | `test_core.py::test_subdivision_config_is_an_exact_partition_of_districts` |

## 2. Fields stored beside the prediction (not inputs)

| field | source | class | rule |
|---|---|---|---|
| `ens_spread` / `ens_spread_mm` | WB2 IFS ENS, 50 members, 0.703° | **FD** (the ENS is issued with the HRES cycle) | Input to the **served combination** (D-030), not to the 52-feature model. Short ensembles are refused, never averaged (`EXPECTED_MEMBERS = 50`). |
| `model_probability` | the model | output | shown only as a labelled ingredient |
| `baseline_probability` | ENS spread + isotonic on 2021 | output (comparator) | same calibration year as the model (D-010) |
| `bust_probability` | combiner (2021 fit) | **the served output** | `NULL` unless `status = OK` |
| `ood_distance` | Mahalanobis on the training features | output | threshold = training 99.5th percentile |
| `observed_rain_mm` | IMD 0.25° gridded, 03Z→03Z | **VO** | **Never an input.** Display and evaluation only. Arrives after the valid day, and IMD gridded products are finalised later still. |
| `actual_bust` | `labels/bust.py` | **VO** | Uses the observation and training-year P95 thresholds (`test_core.py::test_error_threshold_is_fitted_on_training_years_only`, `test_fold_leakage.py::test_bust_thresholds_ignore_later_years`). Never an input. |
| `forecast_rain_mm` | HRES | **FD** | duplicate of `fcst_rain_mm`, kept for the verification log |

## 3. What this means for a live system

- **Ready for live use (FT, FD, SM):** the forecast block, the lagged
  proxy, run-to-run change, climatology, static geography, and the ENS
  spread. Each has a clear issue-time existence.
- **Must be substituted before live use (RA → UV):** every ERA5-derived
  feature (7 local + 9 national + 6 tendencies + 8 regime = 30 of the 52).
  - A live system has the **operational IFS analysis**, available a few
    hours after 00 UTC, not ERA5.
  - **Tested (B2, D-033):** these features were rebuilt from WB2 `hres_t0`
    (the operational initial state), with total column water vapour
    integrated from its humidity, then retrained on the same folds and
    compared under a registration.
    - Non-inferiority at 0.005 AUROC was **not established**: −0.0033
      [−0.0057, −0.0010].
    - The cost is small but real, and ERA5 is clearly better in 2021 and
      2022.
    - The derived TCWV itself passed both validation gates.
  - So the live path still must refuse to serve a probability that depends
    on them, or serve a model built on the operational state **labelled as
    such**, with its measured cost stated.
- **Never inputs (VO):** `observed_rain_mm` and `actual_bust`.
- **The 3-hour window offset (D-005)** affects labels, not inputs. It is
  accepted and applies equally to every baseline.
