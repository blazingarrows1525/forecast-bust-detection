"""Generate a self-contained Colab/Jupyter notebook for the bust-detection model.

The notebook trains the exact XGBoost + isotonic pipeline from src/fbd/model on
the real committed dataset (pulled from GitHub raw), reproduces the held-out-year
headline numbers, shows TreeSHAP reasons, and includes an honest TIGGE/cdsapi
section showing how a live IMD forecast would enter the system.

Run:  python scripts/build_colab_notebook.py
Out:  notebooks/fbd_xgboost_colab.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "fbd_xgboost_colab.ipynb"
RAW = "https://github.com/blazingarrows1525/forecast-bust-detection/raw/master/data/processed/dataset.parquet"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# Forecast Bust Detection — XGBoost pipeline (Colab-ready)

**SIH 2026 · SIH26079 · Ministry of Earth Sciences**

This notebook trains the **exact** model used in the project — a class-weighted
XGBoost classifier with isotonic calibration — on the real committed dataset, and
reproduces the held-out-year headline result.

It predicts, per **IMD subdivision × lead day × initialisation date**, the
probability that an existing medium-range rainfall forecast is about to **bust**
(be badly and decision-alteringly wrong). It does **not** forecast the weather.

**Part 1** (Sections 1–6): the full ML pipeline. Runs in ~2 minutes on a free
Colab CPU, no credentials, no GPU.

**Part 2** (Section 7): how a *live* IMD forecast is pulled from ECMWF TIGGE via
`cdsapi` — the data-ingestion bridge — with an honest note on what connects and
what does not.

Repo: https://github.com/blazingarrows1525/forecast-bust-detection
""")

# ---------------------------------------------------------------- 1 install
md("## 1. Install dependencies")
code(r"""
# Colab already has pandas / numpy / scikit-learn. We add xgboost + pyarrow.
!pip -q install "xgboost>=2.0" pyarrow scikit-learn 2>/dev/null
import xgboost, sklearn, pandas as pd, numpy as np
print("xgboost", xgboost.__version__, "| sklearn", sklearn.__version__, "| pandas", pd.__version__)
""")

# ---------------------------------------------------------------- 2 data
md(r"""
## 2. Load the real dataset

`dataset.parquet` is the project's committed modelling frame — **279,650 rows ×
93 columns**, one row per (subdivision, init date, lead day), already labelled and
feature-engineered. We pull it straight from the GitHub repo.

Every row is derived from real data: **IMD 0.25° gauge rainfall** (observed
truth), **WeatherBench 2 IFS HRES** (the forecast whose busts we predict), and
**ERA5** (atmospheric state). See `DATA.md` in the repo for provenance.
""")
code(f"""
URL = "{RAW}"
ds = pd.read_parquet(URL)   # ~34 MB; a few seconds on Colab
print("shape:", ds.shape)
print("splits:", ds.split.value_counts().to_dict())
print("overall bust rate: {{:.2%}}".format(ds.bust.mean()))
ds.head(3)
""")

# ---------------------------------------------------------------- 3 label
md(r"""
## 3. What the label means (read this before trusting any number)

A **bust** ( `bust = 1` ) is defined by three conditions that must **all** hold —
this is the heart of the project:

1. **Magnitude:** `|forecast − observed| ≥ max(10 mm, P95(subdivision, month))`,
   where P95 is fitted **on training years only**.
2. **Category flip:** forecast and observed fall in different IMD rainfall
   intensity classes (no-rain / light / moderate / heavy / very-heavy / extreme).
3. **Operational significance:** at least one side is ≥ 15.6 mm/day (moderate+).

All three ⇒ *a large error that would have flipped a real alert decision*. The
model never sees `obs_rain_mm`, `error`, or `bust` as inputs — those are truth,
used only to build the label.
""")
code(r"""
# The demo case: Assam & Meghalaya floods, init 2022-06-14. Day 4 is the bust.
demo = ds[(ds.subdivision_id == "ASSAM_MEGHALAYA") &
          (ds.init_date.astype(str).str.startswith("2022-06-14"))]
demo[["subdivision_id","lead_day","fcst_rain_mm","obs_rain_mm","error","bust","bust_type"]].head(6)
""")

# ---------------------------------------------------------------- 4 features
md(r"""
## 4. Features and the temporal split

**52 features** in 6 concept families: forecast amount/anomaly, forecast
disagreement (lagged ensemble spread, jumpiness), climatology, ERA5 regional
dynamics, ERA5 synoptic monsoon state, and monsoon-regime soft-probabilities.

**Strict temporal split — never random:** train on 2016–2020, calibrate on 2021,
test on the fully held-out 2022. A random split would leak autocorrelated weather
between train and test and inflate every score.
""")
code(r"""
FORECAST = ["lead_day","fcst_rain_mm","fcst_anomaly","fcst_rel_to_p90","lagged_spread",
    "lagged_spread_rel","lagged_mean","lagged_range","lagged_n_members","spread_growth",
    "jumpiness","fcst_prev_run","clim_obs_mean","clim_obs_p90","clim_fcst_mean",
    "clim_bust_rate","day_of_season","month"]
ERA5_LOCAL = ["moisture_flux_850","wind_shear","tcwv","z500","mslp","u850","v850"]
ERA5_NAT = ["somali_jet_z","monsoon_trough_mslp_z","nw_z500_z","india_shear_z","india_tcwv_z",
    "india_q850_z","mcz_q850_z","bob_vorticity_max_z","bob_mslp_min_z","somali_jet_d1",
    "somali_jet_d3","bob_vorticity_max_d1","monsoon_trough_mslp_d1","nw_z500_d1","india_tcwv_d1"]
REGIME = ["regime_active_monsoon","regime_break_monsoon","regime_monsoon_depression",
    "regime_western_disturbance","regime_orographic","regime_coastal","regime_entropy","regime_top_prob"]
STATIC = ["elevation_m","terrain_roughness_m","coastal_index","orographic_index"]

FEATURES = [c for c in FORECAST + ERA5_LOCAL + ERA5_NAT + REGIME + STATIC if c in ds.columns]
print(f"{len(FEATURES)} features present")

train = ds[ds.split == "train"].dropna(subset=["bust"])
val   = ds[ds.split == "val"].dropna(subset=["bust"])
test  = ds[ds.split == "test"].dropna(subset=["bust"])
print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
print("bust rate  train={:.2%}  val={:.2%}  test={:.2%}".format(
      train.bust.mean(), val.bust.mean(), test.bust.mean()))
""")

# ---------------------------------------------------------------- 5 train
md(r"""
## 5. Train the model — XGBoost + isotonic calibration

This is the exact configuration from `src/fbd/model/train.py`.

Two things a jury will ask about:
- **`scale_pos_weight = neg / pos ≈ 25`** — busts are ~4% of the data, so without
  class weighting the model would just predict "no bust" every time (96% accuracy,
  zero usefulness).
- **Isotonic calibration fitted on the *validation* year** — class weighting makes
  raw scores over-confident, so we calibrate; afterwards a "30%" flag busts about
  30% of the time.
""")
code(r"""
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

X = train[FEATURES].to_numpy(np.float32)   # NaN kept — XGBoost routes it natively
y = train.bust.to_numpy(int)
pos, neg = max(int(y.sum()), 1), int((y == 0).sum())

params = dict(
    n_estimators=600, max_depth=5, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.75, min_child_weight=20, reg_lambda=2.0,
    scale_pos_weight=neg / pos,          # rare-event correction (~25)
    eval_metric="logloss", tree_method="hist", random_state=20260920, n_jobs=0,
)

booster = xgb.XGBClassifier(**params)
booster.fit(X, y,
            eval_set=[(val[FEATURES].to_numpy(np.float32), val.bust.to_numpy(int))],
            verbose=False)

# Isotonic calibration on the VALIDATION year only (never on test)
raw_val = booster.predict_proba(val[FEATURES].to_numpy(np.float32))[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(raw_val, val.bust.to_numpy(float))

def predict(df):
    raw = booster.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1]
    return calibrator.predict(raw)

print("trained:", booster.n_estimators, "trees, scale_pos_weight =",
      round(params["scale_pos_weight"], 1))
""")

# ---------------------------------------------------------------- 6 eval
md(r"""
## 6. Evaluate on the held-out 2022 year

Never report raw accuracy — with a 4% base rate, always-"no bust" scores 96%.
We use **AUROC** (rank separation, threshold-free), **Brier** (probability
accuracy), and a **reliability** check (is a 30% flag busting ~30%?).

**Expected headline: AUROC ≈ 0.84 overall, ≈ 0.83 on the Day 3–7 decision band**,
beating a lagged-ensemble spread baseline by a clear margin.
""")
code(r"""
from sklearn.metrics import roc_auc_score, brier_score_loss

p_test = predict(test)

def report(mask, name):
    yt = test.bust.to_numpy()[mask]; pp = p_test[mask]
    print(f"{name:22s}  n={mask.sum():>6,}  AUROC={roc_auc_score(yt, pp):.3f}  "
          f"Brier={brier_score_loss(yt, pp):.4f}  base={yt.mean():.2%}")

all_rows = np.ones(len(test), bool)
band = test.lead_day.isin([3,4,5,6,7]).to_numpy()
report(all_rows, "XGBoost — all leads")
report(band,     "XGBoost — Day 3-7")

# Baseline to beat: lagged ensemble spread, calibrated the same way
from sklearn.isotonic import IsotonicRegression as Iso
sp_tr = train.dropna(subset=["lagged_spread"])
base_iso = Iso(out_of_bounds="clip", y_min=0, y_max=1).fit(
    sp_tr.lagged_spread.to_numpy(float), sp_tr.bust.to_numpy(float))
p_base = base_iso.predict(np.nan_to_num(test.lagged_spread.to_numpy(float),
                                        nan=float(train.bust.mean())))
print(f"\n{'ensemble spread — Day 3-7':22s}  AUROC="
      f"{roc_auc_score(test.bust.to_numpy()[band], p_base[band]):.3f}   <- the baseline we beat")
""")
code(r"""
# Reliability: equal-count bins. Observed frequency should track predicted probability.
import numpy as np
order = np.argsort(p_test); yt = test.bust.to_numpy()[order]; pp = p_test[order]
print("predicted  observed   n")
for b in np.array_split(np.arange(len(pp)), 10):
    print(f"  {pp[b].mean():.3f}     {yt[b].mean():.3f}    {len(b)}")
""")

# ---------------------------------------------------------------- 6b shap
md(r"""
### 6b. Explainability — why each flag fires (TreeSHAP)

The problem statement *requires* explainable output. Native TreeSHAP gives an
exact per-prediction attribution. Here are the top factors for the Assam Day-4
bust — no external `shap` package needed, this uses XGBoost's own C++ path.
""")
code(r"""
row = test[(test.subdivision_id=="ASSAM_MEGHALAYA") &
           (test.init_date.astype(str).str.startswith("2022-06-14")) &
           (test.lead_day==4)]
if len(row):
    import xgboost as xgb
    dm = xgb.DMatrix(row[FEATURES].to_numpy(np.float32), feature_names=FEATURES)
    contribs = booster.get_booster().predict(dm, pred_contribs=True)[0][:-1]  # drop base term
    top = sorted(zip(FEATURES, contribs), key=lambda t: -t[1])[:5]
    print(f"Assam & Meghalaya, Day 4 — calibrated bust probability = {predict(row)[0]:.1%}")
    print(f"forecast {row.fcst_rain_mm.iloc[0]:.1f} mm/day vs observed "
          f"{row.obs_rain_mm.iloc[0]:.1f} mm/day  ->  actual bust = {int(row.bust.iloc[0])}\n")
    print("top factors RAISING bust probability (feature, SHAP contribution):")
    for f, c in top:
        print(f"  {f:26s} {c:+.3f}")
else:
    print("row not in this split")
""")

# ---------------------------------------------------------------- 7 tigge
md(r"""
---
## 7. Live IMD forecast via ECMWF TIGGE (the data-ingestion bridge)

Everything above uses the **archived** WeatherBench 2 forecasts. To score
*today's* forecast you need a live source. **ECMWF TIGGE** (via the Copernicus
CDS) publishes real **IMD-origin** medium-range forecasts — control **and**
perturbed (ensemble) members — which is exactly the operational feed this system
is designed to sit on top of.

**Honest scope note — read before running:**
- TIGGE needs a free **CDS account** and a one-time **TIGGE licence acceptance**;
  `cdsapi.Client()` reads your key from `~/.cdsapirc`.
- A single field (one variable, one level, one lead) is **not** a drop-in to the
  trained model. The model consumes **52 features aggregated to IMD subdivisions**
  — subdivision area-means of rainfall, lagged-ensemble spread, ERA5 state, regime
  probabilities. Turning raw TIGGE GRIB into those features is the job of the
  repo's `src/fbd/ingest/` and `src/fbd/features/` modules.
- So this cell demonstrates **ingestion**, not end-to-end live scoring. The full
  live path (TIGGE → subdivision features → model) is the "live IMD adapter"
  listed as future work in the repo.
""")
code(r"""
# One-time setup (uncomment and fill in on Colab):
# 1. Register at https://cds.climate.copernicus.eu and accept the TIGGE licence.
# 2. Write your key file:
# import os
# os.makedirs(os.path.expanduser("~"), exist_ok=True)
# with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
#     f.write("url: https://cds.climate.copernicus.eu/api\n")
#     f.write("key: <YOUR_UID>:<YOUR_API_KEY>\n")
# !pip -q install cdsapi cfgrib eccodes xarray

import cdsapi

dataset = "tigge-forecasts"
request = {
    "origin": "imd",                       # IMD's own forecast — the operational feed
    "year": "2025", "month": "05", "day": "15",
    "time": "12:00",
    "level_type": "pressure",
    "variable": ["temperature"],
    "forecast_type": "control_forecast",   # use 'perturbed_forecast' for the ensemble spread
    "leadtime_hour": ["0"],
    "level_value": ["200_hpa"],
    "data_format": "grib",
}

# client = cdsapi.Client()
# path = client.retrieve(dataset, request).download()   # writes a .grib file
# import xarray as xr
# field = xr.open_dataset(path, engine="cfgrib")        # then aggregate to subdivisions
# print(field)
print("TIGGE request prepared. Uncomment the retrieve() lines after CDS setup.")
print("For ensemble spread (a key feature), set forecast_type='perturbed_forecast'")
print("and request all members, then compute the per-subdivision std across members.")
""")

# ---------------------------------------------------------------- 8 close
md(r"""
## 8. How this maps to the full repository

| This notebook | Repo module |
|---|---|
| feature list | `src/fbd/model/train.py` (`ALL_FEATURES`) |
| XGBoost + isotonic | `src/fbd/model/train.py` (`BustModel`) |
| the bust label | `src/fbd/labels/bust.py` |
| baselines | `src/fbd/model/baselines.py` |
| metrics | `src/fbd/evaluate/metrics.py` |
| TreeSHAP → sentences | `src/fbd/explain/reasons.py` |
| out-of-distribution refusal | `src/fbd/ood/detector.py` |
| TIGGE-style ingestion | `src/fbd/ingest/` + `src/fbd/features/` |

The repo also serves an offline dashboard (2D map + 3D command centre) from a
precomputed SQLite store, has an out-of-distribution "refuse to guess" layer, and
ships 68 passing tests. This notebook is the **model core**, runnable anywhere.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}  ({len(cells)} cells)")
