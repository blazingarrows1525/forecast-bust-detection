"""Search and LLM backends behind two small interfaces, swappable by config.

    SearchProvider.search(query, *, max_results, topic, include_domains) -> list[dict]
    LLMProvider.generate(system, user, *, max_tokens, deadline)          -> LLMResult

Hosts are constants from settings.PROVIDERS (plus the operator's Ollama host,
scheme-checked), never from a request, so these calls are not an SSRF surface;
page fetches, which ARE driven by search results, go through urlsafe.fetch.

Failure policy, the same for every backend:

* a bounded number of attempts (``ATTEMPTS``) with exponential backoff and
  jitter, and only for errors worth retrying -- 429, 5xx, timeouts, resets;
* 400/401/403/404 are not retried: a bad key does not get better with time;
* a per-provider circuit breaker: after ``BREAK_AFTER`` consecutive failures
  the provider is skipped for ``BREAK_FOR`` seconds instead of being hammered;
* every error that reaches a caller is a ProviderFailure with a short,
  sanitised message and a ``retryable`` flag -- never a raw body, never a key.
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from fbd.genai.research.settings import ProviderSpec

ATTEMPTS = 3
BREAK_AFTER = 3
BREAK_FOR = 60.0


def _proxy_auth() -> bool:
    """True when a trusted outbound proxy injects auth headers for us.

    Set ``FBD_ASSISTANT_PROXY_AUTH=1`` when the keys live in Claude Code's
    Network secrets panel (or an equivalent gateway). The code then neither
    checks for a local key nor sends an auth header; the proxy adds the right
    header for the host, and the key never reaches this process.
    """
    import os
    return os.environ.get("FBD_ASSISTANT_PROXY_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}


class ProviderFailure(RuntimeError):
    def __init__(self, provider: str, code: str, message: str, retryable: bool):
        super().__init__(f"{provider}: {message}")
        self.provider, self.code, self.retryable = provider, code, retryable


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


# ------------------------------------------------------------------ breaker
_breaker: dict[str, tuple[int, float]] = {}
_lock = threading.Lock()


def _open(name: str) -> bool:
    with _lock:
        fails, until = _breaker.get(name, (0, 0.0))
        return fails >= BREAK_AFTER and time.monotonic() < until


def _record(name: str, ok: bool) -> None:
    with _lock:
        if ok:
            _breaker.pop(name, None)
        else:
            fails, _ = _breaker.get(name, (0, 0.0))
            _breaker[name] = (fails + 1, time.monotonic() + BREAK_FOR)


def reset_breakers() -> None:
    with _lock:
        _breaker.clear()


def _post_json(name: str, url: str, payload: dict, headers: dict, timeout: float,
               deadline: float | None = None) -> dict:
    """POST with the failure policy above. ``deadline`` is a monotonic time."""
    if _open(name):
        raise ProviderFailure(name, "circuit_open", "temporarily skipped after repeated failures", True)
    body = json.dumps(payload).encode("utf-8")
    last: ProviderFailure | None = None
    for attempt in range(ATTEMPTS):
        if deadline is not None and time.monotonic() > deadline:
            break
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        t = timeout if deadline is None else max(1.0, min(timeout, deadline - time.monotonic()))
        try:
            # Hosts are fixed https endpoints from settings.PROVIDERS, or the
            # scheme-checked Ollama host; never a URL from a request.
            with urllib.request.urlopen(req, timeout=t) as resp:  # nosec B310
                data = json.loads(resp.read().decode("utf-8"))
            _record(name, True)
            return data
        except urllib.error.HTTPError as exc:
            code = exc.code
            retryable = code == 429 or code >= 500
            msg = {401: "the API key was rejected", 403: "access denied for this key or model",
                   404: "model or endpoint not found", 400: "the request was rejected as invalid",
                   429: "rate limited by the provider"}.get(code, f"HTTP {code}")
            last = ProviderFailure(name, f"http_{code}", msg, retryable)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            last = ProviderFailure(name, "network", f"unreachable ({type(reason).__name__})", True)
        except json.JSONDecodeError:
            last = ProviderFailure(name, "bad_response", "the provider returned malformed JSON", True)
        if not last.retryable:
            break
        # Jitter for retry spacing, not for anything security-relevant.
        time.sleep(min(4.0, (0.5 * 2 ** attempt)) * (0.5 + random.random()))  # nosec B311
    _record(name, False)
    raise last or ProviderFailure(name, "deadline", "request deadline reached", True)


# ------------------------------------------------------------------ search
class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, key: str | None):
        self.key = key

    def available(self) -> tuple[bool, str]:
        if self.key:
            return True, "ready"
        if _proxy_auth():
            return True, "ready (auth injected by the Claude Code proxy)"
        return False, "TAVILY_API_KEY is not set on the server"

    def search(self, query: str, *, max_results: int = 5, topic: str = "general",
               include_domains: list[str] | None = None, deadline: float | None = None) -> list[dict]:
        if not self.key and not _proxy_auth():
            raise ProviderFailure(self.name, "no_key", "TAVILY_API_KEY is not set on the server", False)
        payload = {"query": query[:400], "search_depth": "basic", "max_results": max(1, min(max_results, 10)),
                   "topic": topic if topic in ("general", "news") else "general",
                   "include_answer": False, "include_raw_content": False}
        if include_domains:
            payload["include_domains"] = include_domains[:20]
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        data = _post_json(self.name, self.URL, payload, headers, 15.0, deadline)
        out = []
        for r in data.get("results") or []:
            if isinstance(r, dict) and isinstance(r.get("url"), str):
                out.append({"title": str(r.get("title") or "")[:300], "url": r["url"],
                            "snippet": str(r.get("content") or "")[:1200],
                            "published_at": r.get("published_date") if isinstance(r.get("published_date"), str) else None})
        return out


class MockSearch:
    """Deterministic results from a fixture table. Used by CI and the eval set."""

    name = "mock"

    def __init__(self, fixtures: dict | None = None):
        self.fixtures = fixtures or {}
        self.calls: list[str] = []

    def available(self) -> tuple[bool, str]:
        return True, "mock"

    def search(self, query: str, **_kw) -> list[dict]:
        self.calls.append(query)
        q = query.lower()
        for key, results in self.fixtures.items():
            if key in q:
                return [dict(r) for r in results]
        return []


# ------------------------------------------------------------------ LLMs
class GeminiLLM:
    """Gemini REST generateContent, JSON response mode."""

    def __init__(self, spec: ProviderSpec):
        self.spec = spec

    def available(self) -> tuple[bool, str]:
        if self.spec.key():
            return True, "ready"
        if _proxy_auth():
            return True, "ready (auth injected by the Claude Code proxy)"
        return False, f"{self.spec.key_env} is not set on the server"

    def generate(self, system: str, user: str, *, max_tokens: int = 1200, deadline: float | None = None) -> LLMResult:
        key = self.spec.key()
        if not key and not _proxy_auth():
            raise ProviderFailure(self.spec.name, "no_key", f"{self.spec.key_env} is not set on the server", False)
        model = self.spec.model()
        t0 = time.monotonic()
        headers = {"x-goog-api-key": key} if key else {}
        data = _post_json(
            self.spec.name, f"{self.spec.base_url}/models/{model}:generateContent",
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"role": "user", "parts": [{"text": user}]}],
             "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens,
                                  "responseMimeType": "application/json"}},
            headers, 40.0, deadline)
        cands = data.get("candidates") or []
        parts = ((cands[0] or {}).get("content") or {}).get("parts") or [] if cands else []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text:
            reason = (cands[0] or {}).get("finishReason") if cands else "no candidates"
            raise ProviderFailure(self.spec.name, "empty", f"no text returned ({reason})", False)
        um = data.get("usageMetadata") or {}
        return LLMResult(text, self.spec.name, model,
                         {"input_tokens": um.get("promptTokenCount"), "output_tokens": um.get("candidatesTokenCount")},
                         int((time.monotonic() - t0) * 1000))


class OpenAICompatLLM:
    """OpenRouter and Groq: OpenAI-style chat completions, JSON object mode."""

    def __init__(self, spec: ProviderSpec):
        self.spec = spec

    def available(self) -> tuple[bool, str]:
        if self.spec.key():
            return True, "ready"
        if _proxy_auth():
            return True, "ready (auth injected by the Claude Code proxy)"
        return False, f"{self.spec.key_env} is not set on the server"

    def generate(self, system: str, user: str, *, max_tokens: int = 1200, deadline: float | None = None) -> LLMResult:
        key = self.spec.key()
        if not key and not _proxy_auth():
            raise ProviderFailure(self.spec.name, "no_key", f"{self.spec.key_env} is not set on the server", False)
        model = self.spec.model()
        t0 = time.monotonic()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        data = _post_json(
            self.spec.name, f"{self.spec.base_url}/chat/completions",
            {**self.spec.request_extra, "model": model, "temperature": 0.2, "max_tokens": max_tokens,
             "response_format": {"type": "json_object"},
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            headers, 40.0, deadline)
        choices = data.get("choices") or []
        text = ((choices[0] or {}).get("message") or {}).get("content") if choices else None
        if not text:
            raise ProviderFailure(self.spec.name, "empty", "no text returned", False)
        u = data.get("usage") or {}
        return LLMResult(text, self.spec.name, data.get("model") or model,
                         {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")},
                         int((time.monotonic() - t0) * 1000))


class OllamaLLM:
    """The project's local model (D-019), in JSON mode. Nothing leaves the machine."""

    def __init__(self, spec: ProviderSpec):
        self.spec = spec

    def _host(self) -> str:
        from fbd.genai.providers.ollama import _require_http_url
        from fbd.genai.settings import LocalSettings
        return _require_http_url(LocalSettings().host.rstrip("/"))

    def available(self) -> tuple[bool, str]:
        try:
            from fbd.genai.providers.ollama import OllamaClient
            from fbd.genai.settings import LocalSettings
            return OllamaClient(LocalSettings()).availability()
        except Exception as exc:  # noqa: BLE001 - availability never raises
            return False, f"local model unavailable ({type(exc).__name__})"

    def generate(self, system: str, user: str, *, max_tokens: int = 1200, deadline: float | None = None) -> LLMResult:
        model = self.spec.model()
        t0 = time.monotonic()
        data = _post_json(self.spec.name, f"{self._host()}/api/chat",
                          {"model": model, "stream": False, "format": "json",
                           "options": {"num_predict": max_tokens, "temperature": 0.2},
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                          {}, 120.0, deadline)
        text = (data.get("message") or {}).get("content") or ""
        if not text:
            raise ProviderFailure(self.spec.name, "empty", "no text returned", False)
        return LLMResult(text, self.spec.name, model,
                         {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")},
                         int((time.monotonic() - t0) * 1000))


class MockLLM:
    """Returns whatever ``responder(system, user)`` returns. Tests and evals only."""

    def __init__(self, responder=None, name: str = "mock", fail: ProviderFailure | None = None):
        self.responder = responder or (lambda s, u: json.dumps({"answer": "No evidence was provided.",
                                                                "claims": [], "limitations": []}))
        self.spec = type("S", (), {"name": name, "model": staticmethod(lambda: "mock-1"), "tier": "local"})()
        self.fail = fail
        self.prompts: list[tuple[str, str]] = []

    def available(self) -> tuple[bool, str]:
        return True, "mock"

    def generate(self, system: str, user: str, **_kw) -> LLMResult:
        self.prompts.append((system, user))
        if self.fail:
            raise self.fail
        return LLMResult(self.responder(system, user), self.spec.name, "mock-1", {"input_tokens": 0, "output_tokens": 0}, 1)


def build_llm(spec: ProviderSpec | None):
    if spec is None:
        return None
    return {"gemini": GeminiLLM, "openai_compat": OpenAICompatLLM, "ollama": OllamaLLM}.get(spec.kind, lambda s: MockLLM())(spec)


def build_search(name: str):
    import os
    if name == "tavily":
        return TavilySearch(os.environ.get("TAVILY_API_KEY"))
    if name == "mock":
        return MockSearch()
    return None
