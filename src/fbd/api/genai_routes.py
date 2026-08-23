"""GenAI routes, mounted only when the layer is explicitly enabled.

These endpoints do not exist on a default build.  ``attach()`` is a no-op
unless ``FBD_GENAI_ENABLED=1``, which is what keeps the air-gapped default
honest: with the flag off there is no route that could attempt egress, not
merely a route that declines to.

Every response carries ``authoritative: false``.  The numbers a forecaster acts
on come from /api/bulletin and /api/review-queue; this surface is an assistant
over those, and the schema says so rather than leaving it to be inferred.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fbd.genai import agent, retrieval
from fbd.genai.client import availability
from fbd.genai.settings import load
from fbd.obs.metrics import REGISTRY, get_logger, log_event, timed

LOG = get_logger("fbd.genai")
router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AskResponse(BaseModel):
    ok: bool
    answer: str
    tools_used: list[str] = []
    guardrail_violations: list[str] = []
    turns: int = 0
    authoritative: bool = Field(
        False,
        description=(
            "Always false. Operational values come from /api/bulletin and "
            "/api/review-queue; this endpoint narrates them."
        ),
    )


@router.get("/status")
def status() -> dict:
    """Whether the layer could actually serve a request, and why not if not."""
    settings = load()
    check = availability()
    return {
        "settings": settings.describe(),
        "backend": check.as_dict(),
        "guardrails": {
            "numeric_grounding": True,
            "non_interference": True,
            "injection_screening": True,
            "note": "Enforced in-process. A guardrail that fails open is not a guardrail.",
        },
    }


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    settings = load()
    if not settings.enabled:
        raise HTTPException(503, "assistant disabled; this build serves offline by default")

    REGISTRY.counter("fbd_assistant_requests_total", "Assistant questions received")
    with timed("fbd_assistant_latency_seconds", "Assistant end-to-end latency"):
        try:
            result = agent.run(req.question, settings=settings)
        except RuntimeError as exc:
            # Backend unreachable (no credentials, no SDK). Report it plainly.
            REGISTRY.counter("fbd_assistant_errors_total", "Assistant backend errors")
            raise HTTPException(503, str(exc)) from exc

    if result.violations:
        REGISTRY.counter(
            "fbd_guardrail_violations_total",
            "Generated answers blocked by a guardrail",
            labels={"kind": result.violations[0].split(":")[0][:40]},
        )
    log_event(
        LOG, "assistant_answer", ok=result.ok, turns=result.turns,
        tools=result.tool_calls, violations=result.violations,
    )
    return AskResponse(**result.as_dict())


@router.get("/search")
def search(q: str, k: int = 4) -> dict:
    """Retrieval only -- no model call, so this works with no credentials.

    Useful on its own: it answers "where is that documented?" against the
    project's decision log, which is most of what a reviewer wants.
    """
    settings = load()
    if not settings.rag:
        raise HTTPException(503, "retrieval disabled")
    hits = retrieval.build_index().search(q, k=max(1, min(k, 10)))
    return {
        "query": q,
        "hits": [
            {"citation": p.citation, "relevance": round(s, 3), "text": p.text[:1200]}
            for p, s in hits
        ],
    }


def attach(app) -> bool:
    """Mount the router if the layer is enabled.  Returns whether it mounted."""
    settings = load()
    if not settings.enabled:
        return False
    app.include_router(router)
    log_event(LOG, "genai_routes_mounted", **settings.describe())
    return True
