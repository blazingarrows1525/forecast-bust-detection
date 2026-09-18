"""Tests for the pluggable LLM backends (D-019).

The risk this guards is specific: the agent, the guardrails and the numeric
grounding checks all consume one response shape. If a provider's translation
layer emits a slightly different shape, the guardrails do not fail loudly --
they silently stop seeing tool calls or text, and an ungrounded number could
reach a forecaster. So these tests assert the *contract*, not the vendor.

No network is touched. The Ollama transport is stubbed; what is under test is
the translation in both directions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd.genai import providers
from fbd.genai.providers import ProviderError, TextBlock, ToolUseBlock
from fbd.genai.providers import ollama as O
from fbd.genai.settings import GenAISettings, LocalSettings


# ------------------------------------------------------------------ defaults
def test_default_provider_is_local_and_needs_no_credentials():
    """The assistant must be runnable with no account, key or spend."""
    s = GenAISettings()
    assert s.provider == "local"
    assert s.local.host.startswith("http://localhost")
    # And the master switch is still OFF: offline serving stays the default.
    assert s.enabled is False


def test_describe_reports_provider_and_offline_capability():
    s = GenAISettings()
    d = s.describe()
    assert d["offline_capable"] is True
    assert d["offline_default"] is True
    # A local model has no cloud region to report.
    assert d["region"] is None


def test_unknown_provider_is_rejected_loudly():
    with pytest.raises(ProviderError, match="unknown GenAI provider"):
        providers.build("nonesuch", LocalSettings())


# ------------------------------------------------- request translation (ours -> Ollama)
def test_system_prompt_becomes_a_leading_system_message():
    msgs = O._to_ollama_messages("BE CAREFUL", [{"role": "user", "content": "hi"}])
    assert msgs[0] == {"role": "system", "content": "BE CAREFUL"}
    assert msgs[1]["role"] == "user"


def test_tool_schemas_convert_to_function_shape():
    converted = O._to_ollama_tools(
        [{"name": "get_bulletin", "description": "d",
          "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}]
    )
    assert converted[0]["type"] == "function"
    fn = converted[0]["function"]
    assert fn["name"] == "get_bulletin"
    # input_schema must land as `parameters`, or the model gets no argument spec.
    assert fn["parameters"]["properties"] == {"x": {"type": "string"}}


def test_tool_results_become_one_message_each():
    """A Messages-API tool_result batch is one user message; Ollama wants N."""
    msgs = O._to_ollama_messages(
        None,
        [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "{}"},
            {"type": "tool_result", "tool_use_id": "b", "content": "{}"},
        ]}],
    )
    assert [m["role"] for m in msgs] == ["tool", "tool"]


def test_assistant_tool_use_blocks_become_tool_calls():
    msgs = O._to_ollama_messages(
        None,
        [{"role": "assistant", "content": [
            ToolUseBlock(id="t1", name="get_bulletin", input={"region": "ODISHA"}),
        ]}],
    )
    call = msgs[0]["tool_calls"][0]["function"]
    assert call["name"] == "get_bulletin"
    assert call["arguments"] == {"region": "ODISHA"}


# ------------------------------------------------ response translation (Ollama -> ours)
def _client_returning(payload, monkeypatch):
    monkeypatch.setattr(O, "_post", lambda url, body, timeout: payload)
    return O.OllamaClient(LocalSettings())


def test_plain_answer_maps_to_text_block_and_end_turn(monkeypatch):
    c = _client_returning({"message": {"content": "Konkan looks risky."}}, monkeypatch)
    r = c.messages.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert r.stop_reason == "end_turn"
    assert [b.type for b in r.content] == ["text"]
    assert r.content[0].text == "Konkan looks risky."


def test_tool_call_maps_to_tool_use_block_and_stop_reason(monkeypatch):
    """If this mapping breaks, the agent silently stops calling tools."""
    c = _client_returning(
        {"message": {"content": "", "tool_calls": [
            {"function": {"name": "get_bulletin", "arguments": {"region": "ODISHA"}}}
        ]}},
        monkeypatch,
    )
    r = c.messages.create(model="m", messages=[])
    assert r.stop_reason == "tool_use"
    block = [b for b in r.content if b.type == "tool_use"][0]
    assert block.name == "get_bulletin"
    assert block.input == {"region": "ODISHA"}
    # The agent echoes tool_use_id back; a missing id would break the loop.
    assert block.id


def test_string_encoded_tool_arguments_are_parsed(monkeypatch):
    """Some builds return `arguments` as a JSON string rather than an object."""
    c = _client_returning(
        {"message": {"tool_calls": [
            {"function": {"name": "t", "arguments": json.dumps({"a": 1})}}
        ]}},
        monkeypatch,
    )
    r = c.messages.create(model="m", messages=[])
    assert [b for b in r.content if b.type == "tool_use"][0].input == {"a": 1}


def test_malformed_tool_arguments_degrade_to_empty_not_crash(monkeypatch):
    c = _client_returning(
        {"message": {"tool_calls": [{"function": {"name": "t", "arguments": "{not json"}}]}},
        monkeypatch,
    )
    r = c.messages.create(model="m", messages=[])
    assert [b for b in r.content if b.type == "tool_use"][0].input == {}


def test_empty_response_still_yields_a_content_block(monkeypatch):
    """`agent._extract_text` indexes into content; it must never be empty."""
    c = _client_returning({"message": {}}, monkeypatch)
    r = c.messages.create(model="m", messages=[])
    assert len(r.content) == 1
    assert r.content[0].type == "text"


def test_usage_is_reported_for_honest_cost_and_latency_claims(monkeypatch):
    c = _client_returning(
        {"message": {"content": "ok"}, "prompt_eval_count": 120,
         "eval_count": 30, "total_duration": 1_500_000_000},
        monkeypatch,
    )
    r = c.messages.create(model="m", messages=[])
    assert r.usage["input_tokens"] == 120
    assert r.usage["output_tokens"] == 30
    assert r.usage["latency_ms"] == 1500.0
    assert r.provider == "local"


# ------------------------------------------------------------------ availability
def test_availability_explains_an_unreachable_backend(monkeypatch):
    c = O.OllamaClient(LocalSettings())
    monkeypatch.setattr(c, "list_models", lambda: [])
    ok, reason = c.availability()
    assert ok is False
    # The message must be actionable, and must reassure that serving is unaffected.
    assert "ollama.com" in reason.lower() or "not reachable" in reason.lower()
    assert "offline serving path is unaffected" in reason


def test_availability_flags_a_missing_model_distinctly(monkeypatch):
    c = O.OllamaClient(LocalSettings())
    monkeypatch.setattr(c, "list_models", lambda: ["some-other-model:1b"])
    ok, reason = c.availability()
    assert ok is False
    assert "not pulled" in reason


def test_availability_ok_when_model_present(monkeypatch):
    s = LocalSettings()
    c = O.OllamaClient(s)
    monkeypatch.setattr(c, "list_models", lambda: [s.model])
    ok, reason = c.availability()
    assert ok is True
    assert s.model in reason


def test_transport_failure_raises_provider_error_not_urlerror(monkeypatch):
    """A request-path failure must surface an actionable message, not a stack trace."""
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(O.urllib.request, "urlopen", boom)
    c = O.OllamaClient(LocalSettings())
    with pytest.raises(ProviderError, match="cannot reach Ollama"):
        c.messages.create(model="m", messages=[])


# --------------------------------------------------- the contract the agent relies on
def test_agent_runs_end_to_end_against_a_stubbed_local_provider(monkeypatch):
    """The real integration check: agent + guardrails over the local provider."""
    from fbd.genai import agent

    c = _client_returning({"message": {"content": "No numbers here, just prose."}}, monkeypatch)
    settings = GenAISettings(enabled=True, narration=True, tool_calling=False)
    result = agent.run("Which regions look risky?", settings=settings, client=c)
    assert result.ok is True
    assert "prose" in result.text


def test_guardrails_still_block_an_invented_number_through_the_local_provider(monkeypatch):
    """Swapping the backend must not weaken numeric grounding."""
    from fbd.genai import agent

    c = _client_returning(
        {"message": {"content": "Konkan & Goa has a 91.4% chance of busting."}},
        monkeypatch,
    )
    settings = GenAISettings(enabled=True, narration=True, tool_calling=False)
    result = agent.run("Give me a probability", settings=settings, client=c)
    assert result.ok is False
    assert result.violations
