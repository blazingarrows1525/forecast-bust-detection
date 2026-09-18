"""Ollama backend: a local model, no account, no key, no spend.

This is the default provider. It preserves the property the rest of the system
is built around -- the demo runs with the network unplugged -- which a managed
cloud backend cannot.

The job the model actually does here is narrow: rewrite a fixed set of TreeSHAP
attributions and bulletin numbers into a sentence a duty forecaster would
accept, and route read-only tool calls. It is narration over numbers the
pipeline already computed, not open-ended reasoning. A 3B model is sufficient,
runs in ~2 GB of VRAM, and keeps the whole loop inside the forecaster's window.

Translation notes
-----------------
Ollama's native ``/api/chat`` speaks an OpenAI-ish dialect; the agent speaks
Messages API. The mapping is confined to this file:

  system prompt        -> a leading {"role": "system"} message
  tool schemas         -> {"type": "function", "function": {...}}
  assistant tool_use   -> assistant message carrying ``tool_calls``
  user tool_result     -> one {"role": "tool"} message per result
  response.tool_calls  -> ToolUseBlock list + stop_reason "tool_use"

``thinking`` and ``output_config`` are Messages-API-specific and are accepted
then ignored, so the agent needs no provider branching.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid

from fbd.genai.providers import ProviderError, Response, TextBlock, ToolUseBlock


def _post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise ProviderError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(
            f"cannot reach Ollama at {url} ({exc.reason}). "
            "Start it with `ollama serve`, or pull a model with "
            "`ollama pull llama3.2:3b`."
        ) from exc


def _to_ollama_messages(system: str | None, messages: list[dict]) -> list[dict]:
    """Messages API -> Ollama chat messages."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        # Block list. Split into text, tool_use (assistant) and tool_result (user).
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in content or []:
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            get = (lambda b, k, d=None: b.get(k, d)) if isinstance(block, dict) else (
                lambda b, k, d=None: getattr(b, k, d)
            )

            if btype == "text":
                text_parts.append(get(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "function": {
                            "name": get(block, "name", ""),
                            "arguments": get(block, "input", {}) or {},
                        }
                    }
                )
            elif btype == "tool_result":
                tool_results.append(
                    {"role": "tool", "content": str(get(block, "content", ""))}
                )

        if tool_results:
            # A Messages-API tool_result batch arrives as one user message;
            # Ollama wants one message per result.
            out.extend(tool_results)
            continue

        entry: dict = {"role": role, "content": "\n".join(text_parts)}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)

    return out


def _to_ollama_tools(tools: list[dict] | None) -> list[dict]:
    """Messages-API tool schemas -> OpenAI-style function schemas."""
    converted = []
    for tool in tools or []:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
        )
    return converted


class _Messages:
    """Implements the ``client.messages.create(...)`` surface."""

    def __init__(self, outer: "OllamaClient"):
        self._outer = outer

    def create(
        self,
        *,
        model: str,
        max_tokens: int = 1024,
        system: str | None = None,
        tools: list | None = None,
        messages: list[dict] | None = None,
        # Accepted and ignored: Messages-API-only knobs with no Ollama analogue.
        thinking: dict | None = None,
        output_config: dict | None = None,
        **_ignored,
    ) -> Response:
        payload = {
            "model": self._outer.model,
            "messages": _to_ollama_messages(system, messages or []),
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        }
        converted_tools = _to_ollama_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        raw = _post(f"{self._outer.host}/api/chat", payload, self._outer.timeout)
        message = raw.get("message") or {}

        blocks: list = []
        text = (message.get("content") or "").strip()
        if text:
            blocks.append(TextBlock(text=text))

        stop_reason = "end_turn"
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            blocks.append(
                ToolUseBlock(
                    id=call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    name=fn.get("name", ""),
                    input=args or {},
                )
            )
            stop_reason = "tool_use"

        if not blocks:
            blocks.append(TextBlock(text=""))

        return Response(
            content=blocks,
            stop_reason=stop_reason,
            usage={
                "input_tokens": raw.get("prompt_eval_count", 0),
                "output_tokens": raw.get("eval_count", 0),
                # Ollama reports nanoseconds.
                "latency_ms": round(raw.get("total_duration", 0) / 1e6, 1),
            },
            provider="local",
            model=self._outer.model,
        )


class OllamaClient:
    """Local Ollama client exposing the Messages-API surface."""

    def __init__(self, settings):
        self.host = getattr(settings, "host", "http://localhost:11434").rstrip("/")
        self.model = getattr(settings, "model", "llama3.2:3b")
        self.timeout = float(getattr(settings, "timeout", 120.0))
        self.messages = _Messages(self)

    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Model tags currently pulled locally. Empty list if unreachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - localhost
                data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:  # noqa: BLE001 - availability probe must never raise
            return []

    def availability(self) -> tuple[bool, str]:
        """Can a call actually succeed right now, and if not, why not."""
        models = self.list_models()
        if not models:
            return False, (
                f"Ollama is not reachable at {self.host}. Install it from "
                "ollama.com, then run `ollama pull llama3.2:3b`. The offline "
                "serving path is unaffected and remains the default."
            )
        if self.model not in models:
            return False, (
                f"Ollama is running but model {self.model!r} is not pulled. "
                f"Run `ollama pull {self.model}`. Available: {', '.join(models[:5])}"
            )
        return True, f"Ollama ready at {self.host} with {self.model}"
