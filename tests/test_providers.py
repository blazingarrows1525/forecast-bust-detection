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


# ------------------------------------------- backend selection on the real path
# These are the tests that were missing. Everything above stubs the transport
# and injects a client, which proved the translation worked but said nothing
# about whether a real request ever reaches it. It did not: agent.run passed
# settings.bedrock to build_client, which took its backwards-compatibility
# branch and constructed the cloud backend no matter what FBD_GENAI_PROVIDER
# said. The abstraction was correct and unreachable. (D-019 addendum 2.)
def test_build_client_honours_the_configured_provider():
    from fbd.genai.client import build_client

    client = build_client(GenAISettings(enabled=True, provider="local"))
    assert isinstance(client, O.OllamaClient)


def test_agent_default_path_constructs_the_local_provider(monkeypatch):
    """No injected client: this is the path a real request takes."""
    from fbd.genai import agent

    seen = {}

    def fake_post(url, payload, timeout):
        seen["url"] = url
        seen["model"] = payload["model"]
        return {"message": {"content": "Nothing unusual in the queue."}}

    monkeypatch.setattr(O, "_post", fake_post)
    settings = GenAISettings(enabled=True, narration=True, tool_calling=False, provider="local")

    result = agent.run("anything to review?", settings=settings)

    assert result.ok is True
    assert seen["url"].startswith("http://localhost:11434"), "must reach Ollama, not AWS"
    assert seen["model"] == settings.local.model, "must send the local model tag"


def test_agent_sends_the_backend_model_not_the_cloud_one(monkeypatch):
    """A local server given a cloud model id 404s; the ids are not interchangeable."""
    from fbd.genai import agent

    sent = {}

    class _Recorder:
        class messages:
            @staticmethod
            def create(**kwargs):
                sent.update(kwargs)
                return providers.Response(content=[TextBlock(text="ok")])

    settings = GenAISettings(enabled=True, narration=True, tool_calling=False, provider="local")
    agent.run("hello", settings=settings, client=_Recorder())

    assert sent["model"] == settings.local.model_id
    assert "anthropic." not in sent["model"]
    # Messages-API-only knobs must not be claimed of a backend that ignores them.
    assert "thinking" not in sent
    assert "output_config" not in sent


def test_agent_sends_cloud_only_knobs_to_the_cloud_backend(monkeypatch):
    from fbd.genai import agent

    sent = {}

    class _Recorder:
        class messages:
            @staticmethod
            def create(**kwargs):
                sent.update(kwargs)
                return providers.Response(content=[TextBlock(text="ok")])

    settings = GenAISettings(enabled=True, narration=True, tool_calling=False, provider="bedrock")
    agent.run("hello", settings=settings, client=_Recorder())

    assert sent["model"] == settings.bedrock.model_id
    assert sent["output_config"] == {"effort": settings.bedrock.effort}


def test_an_unsupported_provider_is_reported_not_silently_downgraded():
    """`none` was advertised in the docstring but never implemented."""
    from fbd.genai.client import provider_availability

    settings = GenAISettings(enabled=True, provider="none")
    assert settings.provider_supported is False
    assert settings.describe()["provider_supported"] is False
    assert settings.describe()["offline_capable"] is False

    check = provider_availability(settings)
    assert check.ok is False
    assert "not a backend this build can construct" in check.reason


def test_ollama_host_must_be_an_http_url(monkeypatch):
    """FBD_OLLAMA_HOST reaches urlopen, which also speaks file: and ftp:."""
    for bad in ("file:///etc/passwd", "ftp://example.invalid/x", "localhost:11434"):
        with pytest.raises(ProviderError, match="must be an http"):
            O._require_http_url(bad)

    ok, reason = O.OllamaClient(LocalSettings(host="file:///etc")).availability()
    assert ok is False
    assert "http(s) URL" in reason


# --------------------------------------------- the failure the guardrails miss
# These do not touch a model. They pin down the detector for the D-019 failure
# (scripts/compare_local_models.py), so that whenever the status-grounding check
# is finally wired into the serving path, it has a spec to satisfy rather than
# being written from memory of what went wrong.
def _detector():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compare_local_models import check_status_grounding

    return check_status_grounding


# The exact sentence both llama3.2:3b and llama3.1:8b produced about a row whose
# status is OK and whose bust probability is 1.000 (D-019 addendum 3).
_OBSERVED_FABRICATION = (
    "The system declined to score Chhattisgarh at day 10 on 2021-06-10 because "
    "the atmospheric state is unlike anything in its training data, and refused "
    "days historically bust far more often than accepted ones."
)
_SCORED_ROW = {"status": "OK", "bust_probability": 1.0}
_REFUSED_ROW = {"status": "OUT_OF_DISTRIBUTION", "bust_probability": None}


def test_numeric_guardrail_cannot_see_the_observed_fabrication():
    """Why a second check is needed at all: this sentence contains no number."""
    from fbd.genai import guardrails

    report = guardrails.guard_output(_OBSERVED_FABRICATION, [1.0, 10.0, 2021.0])
    assert report.ok is True, (
        "the numeric guardrail passes this sentence -- which is the point. "
        "The fabrication is about status and direction, not arithmetic."
    )


def test_status_grounding_catches_a_refusal_claimed_against_a_scored_row():
    report = _detector()(_OBSERVED_FABRICATION, _SCORED_ROW, top_decile=0.069)
    assert report["ok"] is False
    assert any("status_inverted" in v for v in report["violations"])


def test_status_grounding_allows_a_refusal_claim_when_the_row_was_refused():
    """The inverse must pass, or the check would suppress a true statement."""
    report = _detector()(_OBSERVED_FABRICATION, _REFUSED_ROW, top_decile=0.069)
    assert report["ok"] is True


def test_status_grounding_catches_low_risk_claimed_on_a_top_decile_row():
    """The more dangerous half: it inverts what the forecaster should do."""
    text = "Konkan & Goa shows low risk of busting; no cause for concern."
    report = _detector()(text, _SCORED_ROW, top_decile=0.069)
    assert report["ok"] is False
    assert any("direction_inverted" in v for v in report["violations"])


def test_status_grounding_passes_a_correct_narration():
    text = (
        "The forecast approaches this subdivision's 90th-percentile rainfall and "
        "this lead time busts often, so it merits a second look."
    )
    assert _detector()(text, _SCORED_ROW, top_decile=0.069)["ok"] is True
