# Real-time source matrix

Which live or near-real-time sources could extend Forecast Bust
Intelligence without breaking its offline default (master prompt §11).
Every provider fact below was checked against the provider's own pages on
**2026-10-05** (links in §4). Anything not checked is marked **unverified**.

**Status of the live path: designed only.** No adapter exists. `mode=live`
reports `STALE` by design, and the offline demo must keep working with the
network unplugged.

## 1. Candidates

| source | what it would add | history for training? | access and licence (verified 2026-10-05) | latency and cadence | fit | decision |
|---|---|---|---|---|---|---|
| **ECMWF IFS open data** (HRES / control + ENS) | the **same model family** the product was trained on, live | **No open archive.** Open data keeps only the latest 12 runs (about 2–3 days). Training history stays WB2 (2016–2022). | CC-BY-4.0 with attribution; `data.ecmwf.int/forecasts/`, also AWS, Azure and GCP mirrors; `ecmwf-opendata` Python client | 00/06/12/18 UTC; 00/12 UTC steps 0–144 h by 3 h, then to 360 h by 6 h; IFS released at the end of the dissemination schedule | **Best fit**: `tp` is published and ENS spread can be computed. Resolution is 0.25°; we regrid to 0.703° to match training. | **First live source (phase G1)** |
| **ECMWF AIFS** (single + ENS) | an AI forecast to compare against | no open archive (same rule) | CC-BY-4.0, same portal | released as soon as produced | interesting as a second opinion; no training history to judge busts | research-only, later |
| **WB2 `hres_t0`** (HRES initial conditions) | the **operational analysis** a live system really has, replacing ERA5 features | **Yes:** `gs://weatherbench2/datasets/hres_t0/`, 2016–2022, 6-hourly, includes `512x256_equiangular_conservative` | anonymous GCS, the same channel as today's WB2 reads | archive only | **Directly solves the ERA5 availability gap** (`FEATURE_AVAILABILITY_MATRIX.md` §3) | **Backlog B2, before any live work** |
| **NOAA GEFS** | an independent ensemble (multi-model disagreement) | **Yes:** AWS Open Data `noaa-gefs-pds` from 2017; 0.25°–0.5° depending on version | NOAA Open Data Dissemination; no account | 4 cycles a day to 16 days; 21 members (31 in GEFSv13) | needs its own bust baseline and alignment; GEFS changed versions within the record | backlog B6 (registered study) |
| **NOAA GFS** | a second deterministic forecast | AWS `noaa-gfs-bdp-pds` | open | 4 cycles a day | lower value than GEFS for spread | defer |
| **NASA GPM IMERG Early** | low-latency satellite rain (monitoring, not truth) | V07 archive | GES DISC; also AWS Open Data `nasa-gpm3imergde` | **about 4 h** after the UTC day closes (Early Run) | **verification-time context only.** It is *not* the IMD truth label and must never become a forecast-time input unless its publication time precedes the decision. | phase G3: a monitoring overlay |
| **MOSDAC / INSAT-3D/3DR** | Indian satellite QPE, cloud | archive on request | **registration** (SSO) required. General users get Level-2+ in near real time and Level-1 after 3 days. Rainfall products free for **non-commercial** use. | NRT for registered NRT users | licensing permits personal research use; public redistribution needs review | defer: terms need a careful read first |
| **IMD observations** | the truth label, live | IMD gridded archive (in use) | **unverified** for a programmatic live feed | daily gridded product lags | needed for live verification, not for prediction | phase G4: verification log only |

## 2. Forecast-time vs verification-time (non-negotiable)

- **Forecast-time** (may feed a prediction):
  - the 00 UTC HRES/ENS run, which is the forecast being judged
  - its ensemble statistics
  - the operational analysis at 00 UTC (from `hres_t0` historically, from
    IFS open data step 0 live)
  - static metadata
- **Verification-time** (may only label, monitor or display):
  - IMD gridded rainfall
  - IMERG (any run)
  - final reanalyses (ERA5)
  - later model cycles
- **The publication-time test.** A value counts as forecast-time only if its
  publication time is before the decision (03:00 or 06:30 UTC) and that
  time is recorded in the run identity (§3).

## 3. The live path, designed only (phase G)

```
discover_runs(as_of) → fetch_run → validate → normalize → features → score → store(live rows)
        ECMWF open data adapter (G1)       hres_t0-trained model (B2)   separate table, status-flagged
```

- **Provider protocol, as master prompt §11.3.** Adapters own discovery,
  file naming, retries, parsing and quality rules; shared code owns
  regridding to 0.703° and the subdivision area means through the existing
  `weights_*.parquet`.
- **Run identity:**
  - provider and product
  - cycle and model version
  - `init_utc`
  - member
  - lead and `valid_utc`
  - `retrieved_utc` and `processing_version`
- **State machine:** `DISCOVERED → DOWNLOADING → VALIDATING → READY |
  PARTIAL | FAILED | QUARANTINED`, plus `STALE` by age.
  - A run is `READY` only with all 50 ENS members (the existing
    `ShortEnsemble` refusal) and every step to 240 h present.
- **Serving stays offline-first:**
  - live rows go to their own table and carry `data_quality` and
    `input_age_hours`
  - the API never downloads on a request
  - with no network, `mode=live` is `STALE` and replay is untouched
- **Feature legality:** until B2 proves the operational-analysis swap, a
  live prediction must not use ERA5-derived features. It is either served
  from an `hres_t0`-trained model or refused with a reason.
- **Retention:** open data keeps only about 12 runs, so the adapter must
  fetch within that window and keep raw runs immutably on local disk.
  Cloud storage needs your approval.

## 4. Sources (accessed 2026-10-05)

- ECMWF open data, products, cycles, steps, licence and retention:
  <https://www.ecmwf.int/en/forecasts/datasets/open-data>
- WeatherBench 2 data guide (`hres_t0`, ERA5 versions):
  <https://weatherbench2.readthedocs.io/en/latest/data-guide.html>
- GPM IMERG Early Run latency and V07:
  <https://gpm.nasa.gov/data/imerg>,
  <https://registry.opendata.aws/nasa-gpm3imergde/>
- NOAA GEFS:
  <https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast>,
  <https://github.com/awslabs/open-data-registry/blob/main/datasets/noaa-gefs.yaml>
- NOAA GFS on AWS: <https://registry.opendata.aws/noaa-gfs-bdp-pds/>
- MOSDAC access and terms: <https://www.mosdac.gov.in/terms-conditions>,
  <https://mosdac.gov.in/faq-page>
