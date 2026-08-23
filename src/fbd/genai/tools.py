"""Tool definitions and dispatch for the LLM layer.

Every tool here is **read-only**.  There is no write tool, no override tool and
no alerting tool, and that is a design decision rather than an omission: the
non-interference invariant (LOGIC.md 1.3) says the system never issues or
suppresses a warning, so the model is given no capability that could.  A
forecaster override already exists as a human-authenticated REST endpoint and
is deliberately not reachable from here.

Each dispatch returns the tool result **and the set of numbers it contains**.
That second return value is what makes the numeric-grounding guardrail
possible: the caller accumulates grounded numbers across tool calls and then
checks that the model's prose invented none of its own.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable

from fbd import config

DB = config.ARTIFACTS / "bulletins.sqlite"

# Tool schemas.  strict=True with additionalProperties=false and an explicit
# required list means the API guarantees the input validates, so the dispatch
# functions below do not need to defend against malformed arguments.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_bulletin",
        "description": (
            "Return the forecast-bust assessment for one IMD subdivision on one "
            "initialisation date, across all available lead days. Use this when "
            "asked about a specific region and date. Returns calibrated bust "
            "probabilities, prediction intervals, and the TreeSHAP-derived "
            "meteorological reasons. These numbers are authoritative; never "
            "restate them approximately and never compute your own."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "region_id": {
                    "type": "string",
                    "description": "Subdivision id, e.g. ASSAM_MEGHALAYA, ODISHA, KONKAN_GOA.",
                },
                "init_date": {
                    "type": "string",
                    "description": "Forecast initialisation date, YYYY-MM-DD.",
                },
            },
            "required": ["region_id", "init_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_review_queue",
        "description": (
            "Return the ranked review queue for one initialisation date: the "
            "subdivision-lead pairs most likely to bust, highest risk first. "
            "This is the primary operational product. Use it for questions like "
            "'what should I look at today' or 'which regions are risky'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "init_date": {"type": "string", "description": "YYYY-MM-DD."},
                "top": {
                    "type": "integer",
                    "description": "How many items to return, 1-25.",
                },
            },
            "required": ["init_date", "top"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_project_docs",
        "description": (
            "Search this project's own engineering documentation -- the locked "
            "specification (LOGIC.md), the decision log with evidence "
            "(DECISIONS.md), and the public writeup. Use this for any question "
            "about methodology, why a choice was made, known limitations, or "
            "benchmark provenance. Prefer it over your own knowledge: the "
            "decision log is the authority on what this system actually does."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_model_metrics",
        "description": (
            "Return held-out benchmark metrics for the model and every baseline "
            "(AUROC, Brier, BSS, ECE, decision cost, economic value), on the "
            "2022 test year. Use when asked how well the system performs or how "
            "it compares to ensemble spread."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "band": {
                    "type": "string",
                    "description": "'decision_band' for Day 3-7, or 'overall' for all leads.",
                },
            },
            "required": ["band"],
            "additionalProperties": False,
        },
    },
]


def _con() -> sqlite3.Connection:
    if not DB.exists():
        raise FileNotFoundError(f"bulletin store missing: {DB}")
    con = sqlite3.connect(DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _collect_numbers(obj: Any, out: list[float]) -> None:
    """Walk a tool result and record every number it contains."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_numbers(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_numbers(value, out)
    elif isinstance(obj, str):
        # Reasons carry numbers inside prose ("+61.9 mm above normal"), and
        # those are grounded too -- they came from TreeSHAP, not the model.
        import re

        for token in re.findall(r"-?\d+(?:\.\d+)?", obj):
            out.append(float(token))


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------
def get_bulletin(region_id: str, init_date: str) -> dict:
    con = _con()
    rows = con.execute(
        "SELECT region, region_id, lead_day, valid_date, bust_probability, "
        "       status, forecast_rain_mm, pi_low, pi_high, dominant_factors "
        "FROM bulletins WHERE region_id = ? AND init_date = ? ORDER BY lead_day",
        (region_id, init_date),
    ).fetchall()
    con.close()
    if not rows:
        return {"error": f"no bulletin for {region_id} at {init_date}"}

    out = []
    for r in rows:
        item = {k: r[k] for k in r.keys()}
        if isinstance(item.get("dominant_factors"), str):
            try:
                item["dominant_factors"] = json.loads(item["dominant_factors"])
            except (ValueError, TypeError):
                pass
        out.append(item)
    return {"region_id": region_id, "init_date": init_date, "predictions": out}


def get_review_queue(init_date: str, top: int = 10) -> dict:
    top = max(1, min(int(top), 25))
    con = _con()
    rows = con.execute(
        "SELECT region, region_id, lead_day, valid_date, bust_probability, "
        "       status, forecast_rain_mm, dominant_factors "
        "FROM bulletins WHERE init_date = ? AND bust_probability IS NOT NULL "
        "ORDER BY bust_probability DESC LIMIT ?",
        (init_date, top),
    ).fetchall()
    con.close()
    if not rows:
        return {"error": f"no bulletin rows for {init_date}"}

    items = []
    for rank, r in enumerate(rows, start=1):
        item = {"rank": rank, **{k: r[k] for k in r.keys()}}
        if isinstance(item.get("dominant_factors"), str):
            try:
                item["dominant_factors"] = json.loads(item["dominant_factors"])
            except (ValueError, TypeError):
                pass
        items.append(item)
    return {"init_date": init_date, "items": items}


def search_project_docs(query: str) -> dict:
    from fbd.genai import retrieval
    from fbd.genai.guardrails import sanitise_untrusted

    index = retrieval.build_index()
    hits = index.search(query, k=4)

    passages, warnings = [], []
    for passage, score in hits:
        clipped, report = sanitise_untrusted(passage.text)
        if not report.ok:
            # Surface it; do not obey it, and do not silently drop it either.
            warnings.extend(report.violations)
            continue
        passages.append(
            {"citation": passage.citation, "relevance": round(score, 3), "text": clipped}
        )
    return {"query": query, "passages": passages, "warnings": warnings}


def get_model_metrics(band: str = "decision_band") -> dict:
    path = config.ARTIFACTS / "results.json"
    if not path.exists():
        return {"error": "results.json missing; run scripts/train_model.py"}
    results = json.loads(path.read_text(encoding="utf-8"))
    key = band if band in results else "decision_band"
    return {"band": key, "rows": results.get(key, [])}


DISPATCH: dict[str, Callable[..., dict]] = {
    "get_bulletin": get_bulletin,
    "get_review_queue": get_review_queue,
    "search_project_docs": search_project_docs,
    "get_model_metrics": get_model_metrics,
}


def dispatch(name: str, arguments: dict) -> tuple[dict, list[float]]:
    """Run a tool and return (result, grounded numbers it contained).

    Tool inputs arrive as JSON from the model and are parsed by the caller with
    json.loads -- never string-matched (the escaping varies by model).
    """
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}, []
    try:
        result = fn(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}, []
    except (FileNotFoundError, sqlite3.Error) as exc:
        return {"error": f"{name} failed: {exc}"}, []

    grounded: list[float] = []
    _collect_numbers(result, grounded)
    return result, grounded
