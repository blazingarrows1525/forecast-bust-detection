# LOGIC.md — Engineering Contract & System Blueprint (SIH26079)

**Problem Statement:** SIH26079 · Medium-Range Forecast Bust Detection for Weather Forecasts  
**Ministry / Stakeholder:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)  
**System Class:** Operational Meteorological Decision-Support Meta-Model  
**Status:** Canonical Engineering Specification (LOCKED)

---

## 1. Problem Statement & Core Philosophy

### 1.1 The Operational Problem
Medium-range weather forecasts (Day 1–10) over the Indian subcontinent are produced by numerical weather prediction (NWP) models (e.g., ECMWF IFS HRES, NCMRWF, IMD GFS). While skill has steadily improved, **unannounced forecast busts**—sudden catastrophic forecast failures where the NWP model predicts heavy rain that never arrives, or misses a localized extreme torrential event—carry devastating socioeconomic and disaster-management costs.

Forecasters cannot manually scrutinize all 36 subdivisions across 10 lead times and 50 ensemble members every single morning before the 08:30 IST / 12:00 IST bulletin deadline.

### 1.2 The Solution: Meta-Modeling Forecast Reliability
This system is **NOT** a weather forecasting model:
> **"We do not compete with IMD's weather forecast; we tell IMD's duty forecasters which forecasts to double-check."**

The system trains a machine learning meta-model on historical forecast errors (2016–2020), conditioned on:
1. **Forecast characteristics** (magnitude, anomaly against local climatology, run-to-run disagreement / jumpiness, lagged ensemble spread).
2. **Atmospheric state dynamics** (Somali Jet intensity, Bay of Bengal vorticity, monsoon trough position, moisture flux, vertical wind shear).
3. **Synoptic monsoon regimes** (Active, Break, Monsoon Depression, Western Disturbance, Orographic, Coastal) and regime ambiguity (entropy).

### 1.3 Authority & Non-Interference Invariant
- The system is purely **decision support**.
- It **never** alters, suppresses, or automatically issues public weather warnings.
- The human duty forecaster remains the sole and final authority.

---

## 2. Operational Workflow & User Persona

### 2.1 The Duty Forecaster Persona
The primary user is the IMD National Weather Forecasting Centre (NWFC) or Regional Meteorological Centre (RMC) duty meteorologist who has ~45 minutes to review NWP output and issue district/subdivision bulletins.

### 2.2 Operational Cycle
```
00:00 UTC (05:30 IST) ──► NWP Run Initialized (HRES / ENS / IMD GFS)
02:30 UTC (08:00 IST) ──► Raw NWP fields available
02:35 UTC (08:05 IST) ──► FBD Meta-Model executes in <30 seconds
02:40 UTC (08:10 IST) ──► Forecaster opens Review Queue on Dashboard:
                           - Top-ranked high-risk subdivisions flagged
                           - Plain-language meteorological reasons displayed
                           - Prediction intervals & OOD warnings inspected
03:00 UTC (08:30 IST) ──► Forecaster incorporates confidence flags into official bulletin
```

---

## 3. Spatial & Temporal Domain

### 3.1 Geographic Bounds
- **Mainland India Bounding Box:** `[lat_min: 6.0°N, lat_max: 38.0°N, lon_min: 66.0°E, lon_max: 100.0°E]`
- **Monsoon Circulation Domain:** `[lat_min: -10.0°S, lat_max: 45.0°N, lon_min: 40.0°E, lon_max: 110.0°E]`
  - *Rationale:* Key monsoon drivers (Somali Jet over the Arabian Sea, depressions in the Bay of Bengal, mid-latitude troughs over Iran/Afghanistan) originate far outside the Indian landmass. Restricting atmospheric state to India would decapitate crucial physical signals.

### 3.2 Target Season
- **Monsoon Season (JJAS):** June, July, August, September (Months 6, 7, 8, 9).
  - *Rationale:* JJAS accounts for >70% of India's annual precipitation and >90% of high-impact flooding / bust events.

### 3.3 Strict Temporal Split (No Temporal Leakage)
- **Training Set:** 2016, 2017, 2018, 2019, 2020 (5 full JJAS seasons; ~200,000 subdivision-lead rows).
- **Validation Set:** 2021 (used strictly for threshold fitting, hyperparameter selection, and Isotonic calibration).
- **Test Set:** 2022 (completely held-out year, untouched until final evaluation).
- *Strict Rule:* No random k-fold cross validation. Temporal splits preserve the sequential nature of climate variability and prevent autocorrelated data leakage.

---

## 4. Spatial Unit & The Mathematical Bust Definition

### 4.1 Spatial Aggregation: 36 IMD Meteorological Subdivisions
- Forecasts and observations are aggregated to **36 IMD meteorological subdivisions**, constructed as a strict partition of the 641 Census-2011 districts (`config/imd_subdivisions.json`).
- **Exact Polygon-Cell Weighted Area Aggregation:**
  - Raster cells are mapped to vector polygons using equal-area projection (`EPSG:7755` / Indian 1975 UTM) to compute fractional area weights $W_{i,s}$.
  - Grid cell centroid approximation is strictly forbidden.
- **Island Subdivision Exclusion (Mainland Gate):**
  - IMD 0.25° gridded daily rainfall is a mainland-only gauge product.
  - Andaman & Nicobar Islands and Lakshadweep have 0 land gauge cells in the gridded product and are explicitly excluded from modeling (resulting in 34 modeled subdivisions).

### 4.2 Lead Times
- **Horizon:** Day 1 to Day 10 ($L \in \{1, 2, \dots, 10\}$).
- **Day 1 Definition:** Day of issue ($T_{valid} = T_{init} + 0\text{ days}$, 24-hr accumulation $[T_{init}, T_{init} + 24\text{h}]$).
- **Core Decision Band:** **Day 3 to Day 7** ($L \in [3, 7]$). This is where medium-range forecast uncertainty is actionable by civil authorities (evacuation, reservoir management, NDRF staging).

### 4.3 The Mathematical Bust Definition (§4.3)
A forecast bust is defined as an error that is both **statistically extreme** and **operationally decision-altering**. A bust label $Y_{s,t,L} \in \{0, 1\}$ occurs if and only if all three conditions are met:

$$\text{Bust} = \mathbb{I}\Big( |F - O| \ge \max(10.0\text{ mm}, P_{95}(s, m)) \Big) \land \mathbb{I}\Big( \text{Cat}(F) \ne \text{Cat}(O) \Big) \land \mathbb{I}\Big( \max(F, O) \ge 15.6\text{ mm/day} \Big)$$

Where:
1. **Magnitude Condition:** Absolute area-mean error $|F - O|$ exceeds the 95th percentile error threshold $P_{95}(s, m)$ for subdivision $s$ in month $m$, with an absolute floor of $10.0\text{ mm/day}$.
   - **Critical Invariant:** $P_{95}(s, m)$ is fitted **strictly on training years (2016–2020)**.
2. **Category Flip Condition:** Forecast category $\text{Cat}(F)$ differs from Observed category $\text{Cat}(O)$ based on official IMD rainfall intensity classes:
   - `No Rain`: $[0.0, 2.5)\text{ mm}$ (Index 0)
   - `Light Rain`: $[2.5, 15.6)\text{ mm}$ (Index 1)
   - `Moderate Rain`: $[15.6, 64.5)\text{ mm}$ (Index 2)
   - `Heavy Rain`: $[64.5, 115.5)\text{ mm}$ (Index 3)
   - `Very Heavy Rain`: $[115.5, 204.5)\text{ mm}$ (Index 4)
   - `Extremely Heavy Rain`: $\ge 204.5\text{ mm}$ (Index 5)
3. **Operational Significance Floor:** At least one of $F$ or $O$ must be in the `Moderate Rain` band or higher ($\ge 15.6\text{ mm/day}$). This prevents trivial false alarms (e.g., $0.1\text{ mm}$ vs $3.0\text{ mm}$ shifting from `No Rain` to `Light Rain`).

---

## 5. Data Sources & Storage

| Dataset | Provider / Store | Resolution / Grid | Variables Used | Role in System |
|---|---|---|---|---|
| **IMD 0.25° Gridded Rainfall** | IMD Pune NetCDF archive | 0.25° lat/lon (~27 km), Daily 03Z–03Z | `rain` (mm/day) | Observed Ground Truth ($O$) |
| **IFS HRES Forecasts** | WeatherBench 2 (GCS Zarr) | 0.703° (512×256), 00 UTC, Leads 1–10 | `total_precipitation_24hr` | Primary NWP Forecast ($F$) |
| **IFS ENS (Ensemble)** | WeatherBench 2 (GCS Zarr) | 0.703° (512×256), 50 members | `total_precipitation_24hr` | Operational Ensemble Spread Baseline |
| **ERA5 Reanalysis** | WeatherBench 2 (GCS Zarr, `1959-2023_01_10`) | 240×121 (1.5°), 00/06/12/18 UTC | `u`, `v`, `z`, `q`, `tcwv`, `msl` at 850, 500, 200 hPa | Initial Atmospheric State Features & Regimes |
| **Census-2011 Districts** | DataMeet Maps (GeoJSON/SHP) | 641 district vector boundaries | Geometry | Boundary Partition for 36 Subdivisions |

---

## 6. Model Architecture & Family Selection

### 6.1 Model Choice: Gradient Boosted Trees (XGBoost) + Isotonic Calibration
- **Architecture:** `XGBClassifier` with `max_depth=5`, `n_estimators=600`, `learning_rate=0.04`, `subsample=0.85`, `colsample_bytree=0.75`, `min_child_weight=20`, `reg_lambda=2.0`, `tree_method="hist"`, and `scale_pos_weight` set to the class ratio (~23.6 on the current training split). These are the values in `src/fbd/model/train.py` and stored in `bust_model.joblib`; the code is authoritative if this line ever drifts again.
- **Post-Hoc Probability Calibration:** Isotonic Regression fitted strictly on the 2021 Validation year.
- **Why NOT Deep Learning / CNN / Transformers:**
  1. Tabular regime with $\sim 10^4$ rows per lead where tree ensembles consistently outperform neural approaches.
  2. Native TreeSHAP computation (`pred_contribs=True`) provides exact, mathematically consistent per-prediction feature attributions in milliseconds.
  3. Auditable, lightweight execution without GPU requirements.

---

## 7. Feature Engineering & Concept Families

All features are grouped into distinct conceptual families to prevent multicollinearity and ensure clear explanations:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FEATURE SET (52 Features)                       │
├────────────────────────┬───────────────────────┬───────────────────────┤
│ 1. FORECAST AMOUNT     │ 2. FORECAST DISAGREE- │ 3. CLIMATOLOGY &      │
│    & ANOMALY (4 feats) │    MENT (8 feats)     │    LOCATION (6 feats) │
│ - fcst_rain_mm         │ - lagged_spread       │ - clim_fcst_mean      │
│ - fcst_rel_to_p90      │ - lagged_mean         │ - clim_bust_rate      │
│ - fcst_anomaly         │ - spread_growth       │ - fcst_anomaly_sd     │
│ - fcst_category        │ - jumpiness           │ - day_of_season       │
├────────────────────────┼───────────────────────┼───────────────────────┤
│ 4. ERA5 REGIONAL DYN-  │ 5. ERA5 SYNOPTIC MON- │ 6. SYNOPTIC REGIMES & │
│    AMICS (14 feats)    │    SOON STATE (12)    │    ENTROPY (8 feats)  │
│ - mcz_q850_z, tcwv_z   │ - somali_jet_u850_z   │ - prob_active         │
│ - shear_200_850_z      │ - bob_vort_850_z      │ - prob_break          │
│ - u850_flux_z, w850_z  │ - trough_lat_error    │ - prob_depression     │
│ - local vorticity/flux │ - national_tcwv_z     │ - prob_wd, prob_orog  │
│ - tendency_24h metrics │ - jet_shear_z         │ - regime_entropy      │
└────────────────────────┴───────────────────────┴───────────────────────┘
```

### 7.1 Causality Rules (Zero-Leakage Invariant)
1. **Initial Date Binding:** All ERA5 atmospheric state features are sampled at $T_{init}$ (never $T_{valid}$). The atmosphere on the verification day does not exist when the forecast is issued.
2. **Lagged Ensemble Causality:** The lagged ensemble spread for lead $L$ is computed using only forecasts initialized at or before $T_{init}$ (leads $\ge L$). Forecasts for shorter leads that verify on the same day are issued *later* in time and cannot be accessed.

---

## 8. Baselines, Evaluation Protocol & Decision Economics

### 8.1 Benchmark Baselines
To prove genuine scientific skill over operational standards, the model is benchmarked against 4 competitive baselines:
1. **Climatology Prior:** Historical bust rate per (subdivision, month, lead).
2. **Forecast Rain Amount Only:** Logistic model mapping raw predicted rainfall volume to bust probability.
3. **Logistic Regression (Spread + Lead):** Standard linear predictability model.
4. **Operational Ensemble Spread:** Calibrated ensemble spread (Hoffman & Kalnay lagged proxy + IFS ENS 50-member spread).

### 8.2 Equal Calibration Protocol
Every baseline and the XGBoost model are calibrated using the exact same Isotonic Regression on the 2021 validation set before evaluation.

### 8.3 Statistical Metrics
- **AUROC (Area Under ROC Curve):** Discriminative ranking power across all operating thresholds.
- **Brier Score ($BS$) & Brier Skill Score ($BSS$):** Mean squared probability error relative to climatology:
  $$BSS = 1 - \frac{BS_{\text{model}}}{BS_{\text{clim}}}$$
- **Expected Calibration Error (ECE):** Equal-count reliability binning (10 quantile bins for rare-event calibration).

### 8.4 Asymmetric Decision Cost & Relative Value
Forecasting errors in disaster management are severely asymmetric:
- **Cost of Missed Bust ($C_{\text{miss}}$):** $10.0$ (disaster unpreparedness, unannounced flood/drought, emergency mobilization failure).
- **Cost of False Alarm ($C_{\text{fa}}$):** $1.0$ (forecaster spends ~10 minutes inspecting secondary ensemble products).

$$\text{Total Cost} = C_{\text{miss}} \cdot \text{FN} + C_{\text{fa}} \cdot \text{FP}$$

$$\text{Economic Value } V = \frac{\text{Cost}_{\text{climatology}} - \text{Cost}_{\text{model}}}{\text{Cost}_{\text{climatology}} - \text{Cost}_{\text{perfect}}}$$

---

## 9. API Contract & Output Data Schema

The FastAPI backend exposes the strict Pydantic contract defined in `src/fbd/api/schema.py`:

```json
{
  "region": "Assam & Meghalaya",
  "region_id": "ASSAM_MEGHALAYA",
  "lead_day": 4,
  "init_date": "2022-06-14",
  "valid_date": "2022-06-17",
  "status": "OK",
  "bust_probability": 0.706,
  "confidence_in_estimate": 0.88,
  "prediction_interval": [0.621, 0.784],
  "dominant_factors": [
    "forecast rainfall is +61.9 mm above seasonal normal (top 1% for this subdivision)",
    "this subdivision and lead time historically bust frequently (5.0% base rate)",
    "column moisture across India is below normal (-0.8 sd)"
  ],
  "regime": {
    "active_monsoon": 0.12,
    "break_monsoon": 0.08,
    "monsoon_depression": 0.65,
    "western_disturbance": 0.01,
    "orographic": 0.10,
    "coastal": 0.04,
    "entropy": 0.42
  },
  "data_quality": "OK",
  "input_age_hours": 2.4,
  "ood_distance": 3.81,
  "baseline_probability": 0.112
}
```

---

## 10. Explainability Engine (TreeSHAP & Reason Generation)

1. **Native TreeSHAP:** Computed directly via XGBoost C++ API (`pred_contribs=True`).
2. **Concept Family Deduplication:** Top SHAP attributions are filtered so that at most one factor per concept family is reported (preventing redundant reasons such as listing 3 collinear moisture metrics).
3. **Direction-Aware Templates:** Each feature template uses explicit `(label, high_phrase, low_phrase)` triples to guarantee meteorologically sound sentence construction.
4. **O(1) Percentile Knots:** Precomputed 101-quantile knot arrays turn historical rank queries into fast binary searches.

---

## 11. Out-of-Distribution (OOD) Detection & Refusal to Guess

### 11.1 The Refusal Invariant
AI weather models degrade significantly on unprecedented climatic extremes. When an atmospheric state lies outside historical training distribution:
> **The system refuses to emit a pseudo-confident number, returning `status: "OUT_OF_DISTRIBUTION"` and `bust_probability: null`.**

### 11.2 Mahalanobis Distance Engine
- Fitted on standardized training features with shrinkage covariance:
  $$D_M(x) = \sqrt{(x - \mu)^T (\Sigma + \lambda I)^{-1} (x - \mu)}$$
- **Threshold:** 99.5th percentile of training distances.
- **Earned Refusal Property:** Validated on test sets—refused instances exhibit a **23.4% bust rate** (vs 3.4% for accepted data), proving the detector catches genuine failure regimes.

---

## 12. Robustness & Quality Gates

1. **Input Dropout (30% missing data):** AUROC degrades gracefully from 0.843 to 0.745; ECE increases from 0.010 to 0.021 without producing silent overconfidence.
2. **Synthetic Bust Injection:** Recall on injected busts scales monotonically from 78% to 93% with error magnitude.
3. **Data Staleness Gate:** If inputs are $>30$ hours old, API sets `data_quality: "STALE"` and disables live confidence scoring.

---

## 13. System Architecture & Offline Guarantee

```
┌────────────────────────────────────────────────────────┐
│                   BROWSER CLIENT                       │
│    Leaflet Map · Review Queue · Replay Slider          │
│    (Vendored JS/CSS - 100% Offline / Zero CDN)         │
└───────────────────────────▲────────────────────────────┘
                            │ HTTP JSON / REST
┌───────────────────────────┴────────────────────────────┐
│                    FASTAPI BACKEND                     │
│    /api/bulletin · /api/review-queue · /api/health     │
└───────────────────────────▲────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│               LOCAL SQLITE DATASTORE                   │
│    bulletins.sqlite (Precomputed 79,900 rows)          │
│    Artifacts: bust_model.joblib, gpkg boundaries       │
└────────────────────────────────────────────────────────┘
```
- **Air-Gapped Operation:** The entire serving layer (FastAPI + Leaflet + SQLite) runs completely offline without internet or external CDN dependencies.

---

## 14. Explicit Non-Goals & Architectural Boundaries

To preserve scientific rigor and avoid hackathon scope bloat, the following are strictly **out of scope**:
1. **No Raw Weather Forecasting:** The system does not simulate fluid equations or output rainfall grids.
2. **No Black-Box Deep Learning (CNN/Transformers/GNN):** Tabular gradient boosting is chosen for interpretability, speed, and sample efficiency.
3. **No Distributed Cloud Bloat:** No Kubernetes, Kafka, Spark, or microservice meshes; single-node reproducibility is mandated.
4. **No Direct Public Alerting:** No automated SMS/social media alerts; all recommendations route to duty meteorologists.
5. **No CDN Dependencies:** All frontend assets are vendored locally.
