"""FastAPI service.  One service, not microservices (LOGIC.md sec 11/14).

Serves precomputed daily bulletins from SQLite.  The system is decision support:
it never issues or suppresses a warning, it ranks district-days for a human
forecaster to look at twice.

Two clock modes, because honesty about staleness is a core requirement:
  * ``mode=replay`` -- the requested init date *is* "now", which is what the
    demo and any historical backtest need.
  * ``mode=live``   -- compares the newest available init date with the real
    wall clock.  Since this build serves a 2016-2022 archive, live mode
    correctly reports STALE and greys the map.  That is the intended behaviour,
    not a bug: a stale high-confidence number is the worst thing this system
    could produce (LOGIC.md sec 1.3).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from fbd import config
from fbd.api.schema import (
    Bulletin,
    BustPrediction,
    DataQuality,
    HealthResponse,
    OverrideRequest,
    PredictionStatus,
    RegimeVector,
    ReviewQueueItem,
    ReviewTier,
)
from fbd.quality import escalation

DB = config.ARTIFACTS / "bulletins.sqlite"
WEB_DIR = config.ROOT / "web"

app = FastAPI(
    title="Forecast Bust Detection — SIH26079",
    description=(
        "Predicts when a medium-range rainfall forecast is likely to fail over "
        "India: which subdivision, which lead day, and why. Decision support for "
        "a duty forecaster — the system never issues or suppresses a warning."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _con() -> sqlite3.Connection:
    if not DB.exists():
        raise HTTPException(
            503, f"bulletin store missing ({DB}); run scripts/generate_bulletins.py"
        )
    con = sqlite3.connect(DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _meta() -> dict:
    try:
        con = _con()
        rows = con.execute("SELECT key, value FROM meta").fetchall()
        con.close()
        return {r["key"]: r["value"] for r in rows}
    except HTTPException:
        return {}


def _quality(init_date: str, mode: str) -> tuple[DataQuality, float | None, str | None]:
    """Data quality + input age + banner text for a given clock mode."""
    if mode == "replay":
        # A 00Z run is typically on the forecaster's desk by ~06Z.
        return DataQuality.OK, 6.0, None
    con = _con()
    latest = con.execute("SELECT MAX(init_date) AS m FROM bulletins").fetchone()["m"]
    con.close()
    if latest is None:
        return DataQuality.UNAVAILABLE, None, "No bulletins available."
    age_h = (
        datetime.now(timezone.utc) - pd.Timestamp(latest).tz_localize("UTC")
    ).total_seconds() / 3600.0
    if age_h > config.STALE_INPUT_HOURS:
        return (
            DataQuality.STALE,
            age_h,
            f"STALE INPUT — newest forecast is {age_h/24:.0f} days old "
            f"(issued {latest}). Confidence values are not current; showing "
            f"archive data. Do not use operationally.",
        )
    return DataQuality.OK, age_h, None


def _to_prediction(r: sqlite3.Row, age_h: float | None, dq: DataQuality) -> BustPrediction:
    regime = None
    if r["regime_json"]:
        d = json.loads(r["regime_json"])
        regime = RegimeVector(
            **{k: v for k, v in d.items() if k in RegimeVector.model_fields}
        )
    pi = (
        [r["pi_low"], r["pi_high"]]
        if r["pi_low"] is not None and r["pi_high"] is not None
        else None
    )
    tier = escalation.classify_tier(
        bust_probability=r["bust_probability"],
        ood_flag=(r["status"] != PredictionStatus.OK.value),
    )
    return BustPrediction(
        review_tier=ReviewTier(tier.value),
        tier_guidance=escalation.guidance(tier),
        region=r["region"],
        region_id=r["region_id"],
        lead_day=r["lead_day"],
        init_date=r["init_date"],
        valid_date=r["valid_date"],
        status=PredictionStatus(r["status"]),
        bust_probability=r["bust_probability"],
        confidence_in_estimate=r["confidence_in_estimate"],
        prediction_interval=pi,
        dominant_factors=json.loads(r["dominant_factors"]) if r["dominant_factors"] else [],
        regime=regime,
        data_quality=dq,
        input_age_hours=age_h,
        ood_distance=r["ood_distance"],
        baseline_probability=r["baseline_probability"],
        observed_rain_mm=r["observed_rain_mm"],
        forecast_rain_mm=r["forecast_rain_mm"],
        actual_bust=r["actual_bust"],
        model_version=r["model_version"],
    )


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    meta = _meta()
    notes: list[str] = []
    try:
        con = _con()
        n = con.execute("SELECT COUNT(*) AS c FROM bulletins").fetchone()["c"]
        rng = con.execute(
            "SELECT MIN(init_date) AS a, MAX(init_date) AS b FROM bulletins"
        ).fetchone()
        con.close()
    except HTTPException as exc:
        return HealthResponse(
            status="unavailable", model_version=meta.get("model_version", "?"),
            model_loaded=False, n_bulletin_rows=0, init_date_range=None,
            latest_init_date=None, input_age_hours=None,
            data_quality=DataQuality.UNAVAILABLE, notes=[str(exc.detail)],
        )

    dq, age_h, banner = _quality(rng["b"], "live")
    if banner:
        notes.append(banner)
    notes.append(
        "This build serves a 2016-2022 reanalysis archive, so live mode reports "
        "STALE by design. Use mode=replay for the historical demo."
    )
    try:
        from fbd.genai.settings import load as _genai_load

        if _genai_load().enabled:
            notes.append("GenAI assistant ENABLED (see /api/assistant/status).")
        else:
            notes.append("GenAI assistant disabled; serving fully offline (default).")
    except Exception:  # noqa: BLE001
        pass
    drift_status = None
    if meta.get("drift_status"):
        try:
            drift_status = json.loads(meta["drift_status"])
            if drift_status.get("status") in {"WATCH", "DRIFT"}:
                notes.append(
                    f"INPUT DRIFT {drift_status['status']}: {drift_status.get('note', '')}"
                )
        except (ValueError, TypeError):
            drift_status = None

    return HealthResponse(
        drift_status=drift_status,
        status="ok" if n else "degraded",
        model_version=meta.get("model_version", "0.1.0"),
        model_loaded=(config.ARTIFACTS / "bust_model.joblib").exists(),
        n_bulletin_rows=n,
        init_date_range=[rng["a"], rng["b"]] if rng["a"] else None,
        latest_init_date=rng["b"],
        input_age_hours=age_h,
        data_quality=dq,
        notes=notes,
    )


@app.get("/api/replay/dates")
def replay_dates() -> dict:
    con = _con()
    rows = con.execute(
        "SELECT DISTINCT init_date FROM bulletins ORDER BY init_date"
    ).fetchall()
    con.close()
    return {"init_dates": [r["init_date"] for r in rows]}


@app.get("/api/regions")
def regions() -> JSONResponse:
    """Subdivision polygons as GeoJSON for the map.

    Serves a precomputed, already-simplified GeoJSON when one is present. The
    geometry never changes at runtime, so reading and simplifying a GeoPackage
    per request bought nothing and forced GDAL/GEOS/PROJ into the serving image
    (D-020). The GeoPackage path remains as a fallback for development
    checkouts that have geopandas installed but have not run the precompute.
    """
    precomputed = config.INTERIM / "regions.geojson"
    if precomputed.exists():
        return JSONResponse(json.loads(precomputed.read_text(encoding="utf-8")))

    if not config.SUBDIVISION_GPKG.exists():
        raise HTTPException(
            503,
            "subdivision geometry missing; run `python -m fbd.regions.build` then "
            "`python scripts/precompute_geo_assets.py`",
        )
    import geopandas as gpd

    g = gpd.read_file(config.SUBDIVISION_GPKG, layer="subdivisions")
    # Simplify for the browser: full-resolution district unions are ~10 MB.
    g["geometry"] = g.geometry.simplify(0.02, preserve_topology=True)
    return JSONResponse(json.loads(g.to_json()))


@app.get("/api/voxel-grid")
def voxel_grid() -> JSONResponse:
    """Region index per grid cell, for the volumetric view.

    Static: the geometry never changes at runtime, so this is precomputed by
    ``scripts/precompute_voxel_grid.py`` from the same exact EPSG:7755
    polygon-cell overlap the feature pipeline uses. Serving it rather than
    rasterising per request keeps the geo stack out of the image (D-020) and
    puts the rendered volume on the model's own grid rather than a second grid
    that merely resembles it.
    """
    path = config.INTERIM / "voxel_grid.json"
    if not path.exists():
        raise HTTPException(
            503,
            "voxel grid missing; run `PYTHONPATH=src python "
            "scripts/precompute_voxel_grid.py`",
        )
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/bulletin", response_model=Bulletin)
def bulletin(
    init_date: str = Query(..., description="YYYY-MM-DD"),
    lead_day: int | None = Query(None, ge=1, le=10),
    mode: str = Query("replay", pattern="^(replay|live)$"),
) -> Bulletin:
    dq, age_h, banner = _quality(init_date, mode)
    con = _con()
    sql = "SELECT * FROM bulletins WHERE init_date = ?"
    params: list = [init_date]
    if lead_day is not None:
        sql += " AND lead_day = ?"
        params.append(lead_day)
    sql += " ORDER BY region"
    rows = con.execute(sql, params).fetchall()
    con.close()
    if not rows:
        raise HTTPException(404, f"no bulletin for init_date={init_date}")

    return Bulletin(
        init_date=init_date,
        issued_at=datetime.now(timezone.utc).isoformat(),
        lead_day=lead_day,
        n_regions=len({r["region_id"] for r in rows}),
        data_quality=dq,
        input_age_hours=age_h,
        banner=banner,
        predictions=[_to_prediction(r, age_h, dq) for r in rows],
    )


@app.get("/api/bulletin/{region_id}", response_model=list[BustPrediction])
def region_bulletin(
    region_id: str,
    init_date: str = Query(...),
    mode: str = Query("replay", pattern="^(replay|live)$"),
) -> list[BustPrediction]:
    dq, age_h, _ = _quality(init_date, mode)
    con = _con()
    rows = con.execute(
        "SELECT * FROM bulletins WHERE region_id = ? AND init_date = ? ORDER BY lead_day",
        (region_id, init_date),
    ).fetchall()
    con.close()
    if not rows:
        raise HTTPException(404, f"no data for {region_id} at {init_date}")
    return [_to_prediction(r, age_h, dq) for r in rows]


@app.get("/api/review-queue", response_model=list[ReviewQueueItem])
def review_queue(
    init_date: str = Query(...),
    top: int = Query(20, ge=1, le=200),
    decision_band_only: bool = Query(True),
) -> list[ReviewQueueItem]:
    """Ranked district-days for the duty forecaster to check twice.

    This is the actual product: not a map, a *queue*.  Ordered by bust
    probability, restricted by default to the Day 3-7 decision band where a
    forecast is trusted but sometimes should not be.
    """
    con = _con()
    sql = "SELECT * FROM bulletins WHERE init_date = ? AND bust_probability IS NOT NULL"
    params: list = [init_date]
    if decision_band_only:
        marks = ",".join("?" * len(config.DECISION_BAND))
        sql += f" AND lead_day IN ({marks})"
        params += list(config.DECISION_BAND)
    sql += " ORDER BY bust_probability DESC LIMIT ?"
    params.append(top)
    rows = con.execute(sql, params).fetchall()
    con.close()
    return [
        ReviewQueueItem(
            rank=i + 1,
            region=r["region"],
            region_id=r["region_id"],
            lead_day=r["lead_day"],
            valid_date=r["valid_date"],
            bust_probability=r["bust_probability"],
            status=PredictionStatus(r["status"]),
            forecast_rain_mm=r["forecast_rain_mm"],
            dominant_factors=json.loads(r["dominant_factors"]) if r["dominant_factors"] else [],
        )
        for i, r in enumerate(rows)
    ]


@app.get("/api/verification")
def verification(init_date: str = Query(...), lead_day: int = Query(..., ge=1, le=10)) -> dict:
    """What actually happened -- model flag vs baseline vs observed bust.

    Powers the demo's truth-reveal beat and, more importantly, the verification
    log an operational system would be audited against.
    """
    con = _con()
    rows = con.execute(
        "SELECT region, region_id, valid_date, bust_probability, baseline_probability,"
        " forecast_rain_mm, observed_rain_mm, actual_bust, status"
        " FROM bulletins WHERE init_date = ? AND lead_day = ? ORDER BY region",
        (init_date, lead_day),
    ).fetchall()
    con.close()
    if not rows:
        raise HTTPException(404, "no verification data")
    recs = [dict(r) for r in rows]
    labelled = [r for r in recs if r["actual_bust"] is not None]
    return {
        "init_date": init_date,
        "lead_day": lead_day,
        "n_regions": len(recs),
        "n_busts_observed": sum(r["actual_bust"] for r in labelled),
        "records": recs,
    }


@app.get("/api/metrics")
def metrics() -> dict:
    """Held-out-year evaluation, so the UI can show it is not a cherry-pick."""
    path = config.ARTIFACTS / "results.json"
    if not path.exists():
        raise HTTPException(503, "results.json missing; run scripts/train_model.py")
    return json.loads(path.read_text())


@app.post("/api/override")
def override(req: OverrideRequest) -> dict:
    """Forecaster override, logged immutably with user and reason."""
    con = _con()
    con.execute(
        "INSERT INTO overrides (region_id, init_date, lead_day, action, reason, user,"
        " created_at) VALUES (?,?,?,?,?,?,?)",
        (
            req.region_id, req.init_date, req.lead_day, req.action, req.reason,
            req.user, datetime.now(timezone.utc).isoformat(),
        ),
    )
    con.commit()
    n = con.execute("SELECT COUNT(*) AS c FROM overrides").fetchone()["c"]
    con.close()
    return {"ok": True, "n_overrides": n,
            "note": "Override recorded. The system does not change any public warning."}


@app.get("/api/overrides")
def list_overrides(limit: int = Query(100, ge=1, le=1000)) -> dict:
    con = _con()
    rows = con.execute(
        "SELECT * FROM overrides ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return {"overrides": [dict(r) for r in rows]}


# --------------------------------------------------------------------------
# Observability and the optional GenAI layer.
#
# Both are mounted BEFORE the static files handler, because that handler is
# mounted at "/" and would otherwise swallow every path below it.
# --------------------------------------------------------------------------
from fastapi import Request                                    # noqa: E402
from fastapi.responses import PlainTextResponse                # noqa: E402

from fbd.obs.metrics import REGISTRY, get_logger, log_event    # noqa: E402

_LOG = get_logger("fbd.api")


@app.middleware("http")
async def _observe(request: Request, call_next):
    """Count and time every request, and never let telemetry break serving."""
    import time as _time

    start = _time.perf_counter()
    response = await call_next(request)
    elapsed = _time.perf_counter() - start
    route = request.url.path
    # Collapse high-cardinality paths so the metric does not explode into one
    # series per region id.
    if route.startswith("/api/bulletin/"):
        route = "/api/bulletin/{region_id}"
    try:
        REGISTRY.counter(
            "fbd_http_requests_total", "HTTP requests served",
            labels={"route": route, "status": response.status_code},
        )
        REGISTRY.observe(
            "fbd_http_latency_seconds", elapsed, "HTTP request latency",
            labels={"route": route},
        )
    except Exception:  # noqa: BLE001 - telemetry must never break the request
        pass
    return response


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def prometheus_metrics() -> str:
    """Prometheus exposition. Note: /api/metrics is the model benchmark table;
    this is operational telemetry, which is a different thing."""
    try:
        con = _con()
        n = con.execute("SELECT COUNT(*) AS c FROM bulletins").fetchone()["c"]
        con.close()
        REGISTRY.gauge("fbd_bulletin_rows", n, "Rows in the bulletin store")
    except HTTPException:
        REGISTRY.gauge("fbd_bulletin_rows", 0, "Rows in the bulletin store")
    return REGISTRY.render()


_CENTROIDS: dict[str, tuple[float, float]] | None = None


def _centroids() -> dict[str, tuple[float, float]]:
    """Representative interior point per subdivision, for placing 3-D columns.

    ``representative_point`` rather than ``centroid``: a centroid can fall
    outside a concave polygon, which would float Konkan & Goa's risk column out
    over the Arabian Sea.

    Prefers the precomputed centroids file so the serving image does not need
    geopandas (D-020); falls back to the GeoPackage for development checkouts.
    """
    global _CENTROIDS
    if _CENTROIDS is not None:
        return _CENTROIDS

    precomputed = config.INTERIM / "centroids.json"
    if precomputed.exists():
        raw = json.loads(precomputed.read_text(encoding="utf-8"))
        _CENTROIDS = {sid: (float(xy[0]), float(xy[1])) for sid, xy in raw.items()}
        return _CENTROIDS

    import geopandas as gpd

    g = gpd.read_file(config.SUBDIVISION_GPKG, layer="subdivisions")
    pts = g.geometry.representative_point()
    _CENTROIDS = {
        sid: (float(p.x), float(p.y))
        for sid, p in zip(g.subdivision_id, pts)
    }
    return _CENTROIDS


@app.get("/api/risk-cube")
def risk_cube(
    init_date: str = Query(..., description="YYYY-MM-DD"),
    mode: str = Query("replay", pattern="^(replay|live)$"),
) -> dict:
    """The space x lead-day risk field, in one compact payload.

    Exists because the 2-D choropleth can only show one lead day at a time,
    while problem-statement deliverable 3 asks which regions *and lead times*
    are unreliable -- a two-dimensional field.  The 3-D command centre renders
    it as colour up a vertical column per subdivision (D-018).

    Deliberately lean: probabilities as parallel arrays indexed by lead day, no
    reason strings or regime vectors.  Those are fetched per region on click,
    so the whole-country view stays a single small request.
    """
    dq, age_h, banner = _quality(init_date, mode)
    con = _con()
    rows = con.execute(
        "SELECT region_id, region, lead_day, status, bust_probability,"
        " baseline_probability, actual_bust, forecast_rain_mm, observed_rain_mm,"
        " ood_distance, valid_date"
        " FROM bulletins WHERE init_date = ? ORDER BY region_id, lead_day",
        (init_date,),
    ).fetchall()
    con.close()
    if not rows:
        raise HTTPException(404, f"no bulletin for init_date={init_date}")

    leads = list(config.LEAD_DAYS)
    slot = {L: i for i, L in enumerate(leads)}
    cents = _centroids()

    regions: dict[str, dict] = {}
    for r in rows:
        rid = r["region_id"]
        reg = regions.get(rid)
        if reg is None:
            lon, lat = cents.get(rid, (float("nan"), float("nan")))
            n = len(leads)
            reg = regions[rid] = {
                "region_id": rid,
                "region": r["region"],
                "lon": lon,
                "lat": lat,
                "p": [None] * n,
                "baseline": [None] * n,
                "status": ["UNAVAILABLE"] * n,
                "actual": [None] * n,
                "fcst_mm": [None] * n,
                "obs_mm": [None] * n,
                "valid_date": [None] * n,
            }
        i = slot.get(r["lead_day"])
        if i is None:
            continue
        reg["p"][i] = r["bust_probability"]
        reg["baseline"][i] = r["baseline_probability"]
        reg["status"][i] = r["status"]
        reg["actual"][i] = r["actual_bust"]
        reg["fcst_mm"][i] = r["forecast_rain_mm"]
        reg["obs_mm"][i] = r["observed_rain_mm"]
        reg["valid_date"][i] = r["valid_date"]

    return {
        "init_date": init_date,
        "lead_days": leads,
        "decision_band": list(config.DECISION_BAND),
        "n_regions": len(regions),
        "data_quality": dq.value,
        "input_age_hours": age_h,
        "banner": banner,
        "regions": sorted(regions.values(), key=lambda d: d["region"]),
    }


try:
    from fbd.api import genai_routes

    _GENAI_MOUNTED = genai_routes.attach(app)
except Exception as exc:  # noqa: BLE001 - the optional layer must never
    # prevent the offline serving path from starting. That is the whole point
    # of the default-off contract in D-015.
    _GENAI_MOUNTED = False
    log_event(_LOG, "genai_mount_failed", error=str(exc))


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
