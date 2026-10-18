"""HTTP surface for the research assistant. Mounted only when enabled.

    GET  /api/assistant/research/status   what is configured (never a key)
    POST /api/assistant/research/chat     ChatRequest -> SSE stream, or JSON when
                                          response_mode == "complete"

Why SSE and not a WebSocket: the traffic is one request and a stream of
server events, which is what SSE is; it rides plain HTTP through any proxy the
deployment already has, and FastAPI serves it with a StreamingResponse and no
new dependency. Every event carries an increasing ``id``, so a client that
receives the same event twice (a retry, a flaky proxy) can drop the duplicate.

Cancellation: when the browser aborts the request, the server stops iterating
the pipeline. A provider call already in flight finishes in its worker thread
and its result is discarded; no further search, fetch or model call is made.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from fbd.genai.research import pipeline
from fbd.genai.research.providers import build_llm, build_search
from fbd.genai.research.schemas import ChatRequest
from fbd.genai.research.settings import load
from fbd.genai.research.state import QUOTA, SEARCH_CACHE

router = APIRouter(prefix="/api/assistant/research", tags=["research-assistant"])

# Tests replace this to run the real route against mock providers.
DEPS_FACTORY = None


def _client(request: Request) -> str:
    return (request.client.host if request.client else "unknown")[:64]


@router.get("/status")
def status() -> dict:
    s = load()
    llm = build_llm(s.spec())
    search = build_search(s.search_provider) if s.web else None
    if llm:
        llm_ok = llm.available()
    elif s.llm.strip().lower() == "none":
        llm_ok = (False, "no model configured: answers come from the archive and project documents only")
    else:
        llm_ok = (False, f"FBD_ASSISTANT_LLM={s.llm!r} is not a known provider")
    search_ok = search.available() if search else (False, "web search is off" if not s.web else "unknown provider")
    return {
        "settings": s.describe(),
        "llm_ready": {"ok": llm_ok[0], "reason": llm_ok[1]},
        "search_ready": {"ok": search_ok[0], "reason": search_ok[1]},
        "cache": {"hits": SEARCH_CACHE.hits, "misses": SEARCH_CACHE.misses},
        "streaming": "server-sent events; stages stream, the answer arrives once, validated",
        "authoritative": False,
    }


def _foreign_origin(request: Request) -> bool:
    """True if a browser on another site sent this request.

    The app's CORS policy is open (read-only data, D-021), which would let any
    web page make its visitors' browsers spend this deployment's model and
    search budget. Chat is therefore same-origin unless the operator lists an
    origin in FBD_RESEARCH_ALLOWED_ORIGINS. Requests with no Origin header
    (curl, server-to-server, the eval harness) are not browsers and pass; the
    quotas still apply to them.
    """
    import os
    import urllib.parse
    origin = request.headers.get("origin")
    if not origin:
        return False
    allowed = {o.strip().rstrip("/") for o in os.environ.get("FBD_RESEARCH_ALLOWED_ORIGINS", "").split(",") if o.strip()}
    if origin.rstrip("/") in allowed:
        return False
    return urllib.parse.urlsplit(origin).netloc != request.headers.get("host", "")


@router.post("/chat")
def chat(req: ChatRequest, request: Request):
    if _foreign_origin(request):
        return JSONResponse({"code": "forbidden_origin", "message": "This assistant only answers its own pages.",
                             "retryable": False}, status_code=403)
    s = load()
    ok, retry, which = QUOTA.check(_client(request), s.per_client_per_minute, s.per_client_per_day,
                                   s.global_per_day)
    if not ok:
        return JSONResponse({"code": "rate_limited", "message": f"Too many questions ({which}). Try again later.",
                             "retryable": True, "retry_after": retry}, status_code=429,
                            headers={"Retry-After": str(retry)})
    deps = DEPS_FACTORY(s) if DEPS_FACTORY else pipeline.default_deps(s)
    events = pipeline.run(req, s, deps)

    if req.response_mode == "complete":
        final, errors, warnings = None, [], []
        for e in events:
            if e["event"] == "done":
                final = e["data"]["answer"]
            elif e["event"] == "error":
                errors.append(e["data"])
            elif e["event"] == "warning":
                warnings.append(e["data"])
        if final is None:
            err = errors[0] if errors else {"code": "internal", "message": "no answer", "retryable": True}
            raise HTTPException(503 if err.get("retryable") else 502, err)
        return {"answer": final, "warnings": warnings}

    def stream():
        n = 0
        # Ask the browser to wait before auto-reconnecting; the page retries
        # explicitly instead, so a dropped stream never replays half an answer.
        yield "retry: 600000\n\n"
        for e in events:
            n += 1
            yield f"id: {n}\nevent: {e['event']}\ndata: {json.dumps(e['data'], separators=(',', ':'))}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
