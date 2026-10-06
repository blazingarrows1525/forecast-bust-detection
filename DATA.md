# DATA.md — what is in this repo, what is not, and how to get the rest

This project uses five datasets. **None of the raw third-party data is redistributed here** — it is external data under its own licence, and this repo pulls it straight from the primary sources with `scripts/fetch_*.py`. What *is* committed is (a) our own derived artifacts, small enough to live in git, and (b) a precomputed results store shipped as a GitHub Release asset.

The result: you can clone the repo and **run the tests and the offline dashboard immediately**, without downloading a single byte of raw NWP data.

---

## 1. What is committed directly in the repo (~42 MB)

These are **our derived products**, not third-party data. Safe to redistribute, small enough for git, and they make the repo runnable.

| Path | Size | What it is |
|---|---|---|
| `data/processed/dataset.parquet` | 34 MB | The full labelled modelling frame: 279,650 rows × 93 columns (subdivision × init × lead, with the bust label, all 52 features, and the split tag). This is the single file `train_model.py` needs. |
| `data/interim/imd_subdivisions.gpkg` | 6 MB | The 34 modelled IMD subdivision polygons (built as unions of 641 Census-2011 districts). The API serves `/api/regions` from this. |
| `data/interim/truth_subdivision_daily.parquet` | 180 KB | Observed subdivision-daily rainfall (area-mean), derived from the IMD gridded product. |
| `data/interim/weights_*.parquet` | ~100 KB | Cached grid↔polygon area-overlap weights, keyed by grid geometry. |
| `data/artifacts/bust_model.joblib` | 1.5 MB | The trained XGBoost booster + isotonic calibrator + feature list. |
| `data/artifacts/results.json` | 7.5 KB | Held-out-year evaluation table (all baselines + model, per-lead, reliability). |
| `data/artifacts/*.csv` | <5 KB each | Ablation, SHAP importance, stress-test, and ENS-baseline comparison tables. |
| `data/artifacts/reference_distributions.npz` | 134 KB | Training-year feature distributions, used by the drift monitor. |

**Derived-data provenance note:** `dataset.parquet` and `truth_subdivision_daily.parquet` contain an observed-rainfall column derived from IMD's 0.25° gridded product (area-averaged to subdivisions, reformatted, and joined with model outputs). This is a transformative derivative work under IMD's research-use terms; attribution is in `LOGIC.md` §5. It is not a copy of the IMD product and cannot be used to reconstruct it.

---

## 2. What ships as a GitHub Release asset (~74 MB)

| Asset | Size | What it is |
|---|---|---|
| `bulletins.sqlite` | 74 MB | The precomputed bulletin store: 79,900 pre-scored (subdivision × init × lead) rows, each with the served bust probability (spatial network + ENS spread since v0.3.0; XGBoost + ENS in v0.2.0), the model alone, the ENS spread, bagged prediction interval, OOD status, SHAP reason strings, regime vector, and the baseline probability. This is what the offline dashboard serves. |

It is a Release asset rather than a committed file because 74 MB of binary would bloat every clone of the git history. It is **fully regenerable** from the committed `dataset.parquet`, `bust_model.joblib` and `combiner.json` by running `scripts/generate_bulletins.py` (~100 s), so the Release is a convenience, not a dependency. Since v0.2.0 (D-030) the served probability combines the model with the 50-member ENS spread, so regenerating also needs the ENS spread for 2021 and 2022 on disk (`scripts/fetch_ens.py --year 2021` and `--year 2022`, about 7 MB of derived parquet under `data/raw/wb2/ens/`, gitignored).

**Get it** (either works):

```bash
# One command — downloads and verifies the checksum:
PYTHONPATH=src python scripts/fetch_release_artifacts.py

# ...or regenerate it locally from the committed artifacts:
PYTHONPATH=src python scripts/generate_bulletins.py
```

SHA-256 of `bulletins.sqlite` (v0.3.0): `2d7ab6db82786998a0d857e25eed087125584cafd69c26869df72ea0c791bb1f`.
Release v0.3.0 also carries the served spatial model, `candidate_spatial_2022.joblib` (SHA-256 `76fb33e7b074f6335dc94fdcf2549a0b13cf079178fb14e452827b1d2d2d6ad4`).
Regenerating v0.3.0 needs that model and the 23 public WB2 grids pinned in `data/artifacts/s3c_inputs.json`.

Release v0.2.0 (the XGBoost + ENS store, SHA-256 `deeac846b62e88e1169c349b75b4dacdd2a7f8b8b99191026de0bdc9f002a468`) stays published.

Release v0.1.0 (the model-alone store, SHA-256 `5cbb61350c2ba63dbcf2dc00bc1d13b9a3d2f9a70e66d8317121cf996e6f241c`) stays published. The API still serves it without error, with no combination shown and the landing page's combination sentence hidden.

---

## 3. What is NOT in the repo and never will be (~570 MB of raw data)

This is external third-party data. We do not rehost it — we cite it and reproduce it from the primary source. Rehosting it would be a licensing grey area and would teach the wrong lesson about how research artifacts are shared.

| Dataset | Local size | Primary source | Fetch with |
|---|---|---|---|
| IMD 0.25° gridded rainfall (2016–2022) | 170 MB | `imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html` | `scripts/fetch_imd.py` |
| WeatherBench 2 IFS HRES forecasts | 72 MB (of ~9 GB transferred) | `gs://weatherbench2/datasets/hres/...` (anonymous GCS) | `scripts/fetch_hres.py` |
| WeatherBench 2 ERA5 analysis | ~450 MB | `gs://weatherbench2/datasets/era5/...` | `scripts/fetch_era5.py` |
| WeatherBench 2 IFS ENS (50-member) | 360 KB (subsampled) | `gs://weatherbench2/datasets/ifs_ens/...` | `scripts/fetch_ens.py` |
| Census-2011 district boundaries | 10 MB | `github.com/datameet/maps` | `scripts/fetch_boundaries.py` |

Full reproduction from nothing:

```bash
python scripts/fetch_imd.py
python scripts/fetch_hres.py
python scripts/fetch_era5.py
python scripts/fetch_ens.py --year 2019 --every 6
python scripts/fetch_ens.py --year 2020 --every 6
python scripts/fetch_ens.py --year 2022 --every 3
python scripts/fetch_boundaries.py
# then rebuild everything:
python scripts/build_truth.py
python scripts/build_dataset.py
python scripts/train_model.py
python scripts/generate_bulletins.py
```

Total download time is dominated by the WeatherBench 2 HRES pull (~20 min) because WB2's zarr chunks span the globe (you download the world and slice to India locally — see `DECISIONS.md` D-002).

---

## 4. The three tiers, at a glance

```
┌─ committed in git (~42 MB) ──────────────────┐   clone the repo -> tests pass,
│  our derived products: dataset.parquet,      │   model loads, pipeline runs
│  trained model, geometry, eval tables        │
└───────────────────────────────────────────────┘
┌─ GitHub Release asset (~74 MB) ──────────────┐   one command -> offline
│  bulletins.sqlite (precomputed dashboard)    │   dashboard runs
└───────────────────────────────────────────────┘
┌─ never in the repo (~570 MB, external) ──────┐   scripts/fetch_*.py ->
│  raw IMD / WeatherBench 2 / Census data      │   reproduce from source
└───────────────────────────────────────────────┘
```

This is the standard pattern for a reproducible-research repo: **cite the primary source, ship the reproduction script, host only your own derivatives.** It is what makes the repo credible rather than merely convenient.
