"""Guardrails for the GenAI layer.

D-015 makes two things binding, and this module is where they are enforced in
code rather than asserted in a prompt:

1. **The LLM never produces a bust probability.**  Every number in generated
   text must already appear in the grounded facts handed to the model.  A model
   that invents "roughly 80% chance" next to a disaster-management product is a
   liability, and a system prompt saying "do not invent numbers" is not a
   control -- it is a request.  This module checks the output and rejects it.

2. **The non-interference invariant (LOGIC.md 1.3) survives.**  The system is
   decision support.  Generated text may not issue directives to the public or
   to civil authorities; it advises a duty forecaster to look at something.

Guardrails run **client side, in-process**.  That is deliberate: a network
guardrail service that is unreachable fails open, and failing open is the one
behaviour this layer must never have.  Bedrock Guardrails can be layered on
top as defence in depth (see ``BEDROCK_GUARDRAIL_NOTE``), but the checks here
are the ones the invariant actually rests on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# A Bedrock Guardrail (guardrailIdentifier / guardrailVersion on the request)
# is a *second* layer, applied server side by AWS.  It is complementary, not a
# replacement: it cannot know which numbers our model computed, so it cannot
# enforce invariant 1.  Configure it for the generic content-safety categories
# and keep these checks for the domain invariants.
BEDROCK_GUARDRAIL_NOTE = (
    "Bedrock Guardrails cover generic content safety. The numeric-grounding "
    "and non-interference invariants are domain-specific and are enforced here."
)

# Any run of digits, with optional decimal part and optional trailing percent.
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(%|percent)?", re.IGNORECASE)

# Numbers that are never risk claims and are safe to see in prose.  Lead days
# run 1-10, small counts appear in phrases like "three reasons".
_SAFE_SMALL_INTEGERS = set(range(0, 11))

# Phrasing that would turn decision support into an instruction.  The system
# advises a forecaster; it does not tell anyone to act.  LOGIC.md 1.3.
_AUTHORITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bevacuat(e|ing|ion)\b",
        r"\bissue (a |an )?(red|orange|amber) (alert|warning)\b",
        r"\bwe (are )?(hereby )?(issuing|declaring)\b",
        r"\bthe public should\b",
        r"\bdo not travel\b",
        r"\bclose (the )?(schools|offices)\b",
        r"\bshut down\b",
        r"\bofficial (warning|bulletin) is\b",
    )
]

# Instructions arriving inside retrieved documents or tool output are data, not
# commands.  These are the shapes that most often smuggle one in.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |the )?(previous|prior|above) instructions",
        r"disregard (all |the )?(previous|prior|above)",
        r"you are now (a|an|in) ",
        r"system prompt\s*[:=]",
        r"<\s*/?\s*(system|instructions?)\s*>",
        r"\bnew instructions?\s*[:=]",
        r"reveal (your|the) (system )?prompt",
        r"act as (if )?(you are )?(an? )?(unrestricted|jailbroken|developer mode)",
    )
]


class GuardrailViolation(Exception):
    """Raised when generated text fails a hard invariant."""


@dataclass
class GuardrailReport:
    """Outcome of a guardrail pass.  Always inspect ``ok``."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)
    redacted_text: str | None = None

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise GuardrailViolation("; ".join(self.violations))


def _candidate_values(raw: str, is_percent: bool) -> list[float]:
    """Every reading of a numeric token that we are willing to call grounded.

    "70.6%" may legitimately restate a stored probability of 0.706, so both
    readings count as a match.
    """
    value = float(raw)
    readings = [value]
    if is_percent:
        readings.append(value / 100.0)
    else:
        readings.append(value * 100.0)
    return readings


def _is_grounded(raw: str, is_percent: bool, allowed: list[float], tol: float) -> bool:
    for reading in _candidate_values(raw, is_percent):
        for ok_value in allowed:
            if abs(reading - ok_value) <= tol:
                return True
            # Tolerate the model rounding a stored 0.7058 to 0.71 or 70.6%.
            if round(reading, 2) == round(ok_value, 2):
                return True
            if round(reading, 1) == round(ok_value, 1):
                return True
    return False


def check_numeric_grounding(
    text: str,
    allowed_numbers: list[float] | set[float],
    tol: float = 0.05,
) -> GuardrailReport:
    """Verify every number in ``text`` traces back to a grounded fact.

    ``allowed_numbers`` is whatever the caller actually computed and handed to
    the model -- probabilities, rainfall amounts, interval bounds, base rates.
    Anything else in the output is a number the model made up.
    """
    allowed = [float(v) for v in allowed_numbers]
    ungrounded: list[float] = []

    for match in _NUMBER_RE.finditer(text):
        raw, suffix = match.group(1), match.group(2)
        is_percent = suffix is not None
        value = float(raw)

        # Bare small integers (lead days, "three factors") are not risk claims.
        if not is_percent and value.is_integer() and int(value) in _SAFE_SMALL_INTEGERS:
            continue
        # Years are not risk claims either.
        if not is_percent and value.is_integer() and 1900 <= value <= 2100:
            continue

        if not _is_grounded(raw, is_percent, allowed, tol):
            ungrounded.append(value)

    if ungrounded:
        listed = ", ".join(str(v) for v in ungrounded)
        return GuardrailReport(
            ok=False,
            violations=[
                f"ungrounded numeric claim(s) in generated text: {listed}. "
                "Every number must come from the calibrated model, not the LLM."
            ],
            ungrounded_numbers=ungrounded,
        )
    return GuardrailReport(ok=True)


def check_authority(text: str) -> GuardrailReport:
    """Reject text that issues instructions instead of advising review."""
    hits = [p.pattern for p in _AUTHORITY_PATTERNS if p.search(text)]
    if hits:
        return GuardrailReport(
            ok=False,
            violations=[
                "generated text issues a directive, violating the "
                f"non-interference invariant (LOGIC.md 1.3): matched {hits}"
            ],
        )
    return GuardrailReport(ok=True)


def check_injection(text: str) -> GuardrailReport:
    """Flag instruction-shaped content arriving from retrieved data.

    Retrieved documents and tool results are data.  When one contains something
    aimed at the model, the correct behaviour is to surface it, not obey it.
    """
    hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    if hits:
        return GuardrailReport(
            ok=False,
            violations=[f"possible prompt injection in untrusted content: {hits}"],
        )
    return GuardrailReport(ok=True)


def guard_output(
    text: str,
    allowed_numbers: list[float] | set[float],
    evidence: "Evidence | None" = None,
) -> GuardrailReport:
    """Full output pass: numeric grounding, non-interference, status grounding.

    This is the function the serving path calls.  It combines every violation
    rather than short-circuiting, so an operator sees all of what went wrong.

    ``evidence`` is optional and defaults to None, which skips the status check.
    A caller with no tool results to check against keeps its previous behaviour
    rather than being handed a verdict assembled out of nothing.
    """
    violations: list[str] = []
    ungrounded: list[float] = []

    numeric = check_numeric_grounding(text, allowed_numbers)
    if not numeric.ok:
        violations.extend(numeric.violations)
        ungrounded.extend(numeric.ungrounded_numbers)

    authority = check_authority(text)
    if not authority.ok:
        violations.extend(authority.violations)

    status = check_status_grounding(text, evidence)
    if not status.ok:
        violations.extend(status.violations)

    return GuardrailReport(
        ok=not violations, violations=violations, ungrounded_numbers=ungrounded
    )


# --------------------------------------------------------------------------
# Status grounding (D-019 addendum 3)
# --------------------------------------------------------------------------
# check_numeric_grounding above cannot see the failure that actually occurred.
# Asked why confidence was low for a row whose status is OK and whose bust
# probability is 1.000, both llama3.2:3b (3 of 3 runs) and llama3.1:8b (1 of 4)
# answered:
#
#     "The system declined to score Chhattisgarh at day 10 on 2021-06-10
#      because the atmospheric state is unlike anything in its training data."
#
# Every word of that is false, and the error INVERTS the meaning: a forecaster
# reading it stands down on the worst cell on the map -- the exact failure this
# project exists to prevent, produced by our own assistant. The numeric check
# passes it, correctly: the sentence contains no invented number. The claim is
# about a row's *status* and about *direction*, and nothing was checking those.
#
# The rule here is the one numeric grounding already uses, applied to a
# different kind of fact: the model may only assert a refusal that a tool
# result actually reported. The agent loop sees every tool result, so it can
# carry the statuses alongside the numbers and check both at output time.

_REFUSAL_CLAIM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(declin\w+|refus\w+) to (score|assess|evaluate|rate)\b",
        r"\b(did not|didn't|was not|wasn't|could not|couldn't|unable to) "
        r"(be )?(score|assess|evaluat)\w*\b",
        r"\bout[- ]of[- ]distribution\b",
        r"\bunlike anything in its training\b",
        r"\boutside (its|the) training\b",
    )
]

# Asserting low risk.  Said about a row the system flagged, this is the more
# dangerous half: it tells the forecaster the opposite of the truth.
_LOW_RISK_CLAIM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\blow (bust )?(risk|probability|chance)\b",
        r"\bunlikely to bust\b",
        r"\b(no|little|minimal) (cause for concern|risk|reason for concern)\b",
        r"\bnot concerning\b",
        r"\bhistorically low\b",
        r"\bappears? (reliable|safe)\b",
    )
]


@dataclass
class Evidence:
    """What the tools actually returned, as opposed to what the model says.

    The numeric counterpart is a plain ``list[float]``.  Statuses need more
    structure, because a claim has to be attached to the region it is made
    about before it can be called true or false.
    """

    #: display name -> status string, for every row any tool returned
    region_status: dict = field(default_factory=dict)
    #: display name -> bust probability (None on a refused row)
    region_probability: dict = field(default_factory=dict)
    #: True once a documentation search has run.  A methodology question --
    #: "what happens when a region is out of distribution?" -- legitimately
    #: describes a refusal without asserting one about any particular row.
    docs_consulted: bool = False
    rows_seen: int = 0

    def observe(self, result, tool_name: str = "") -> "Evidence":
        """Record every region/status pair in one tool result."""
        if tool_name == "search_project_docs":
            self.docs_consulted = True
        self._walk(result)
        return self

    def _walk(self, node) -> None:
        if isinstance(node, dict):
            name = node.get("region") or node.get("region_id")
            if name and "status" in node:
                key = canonical_region(name)
                self.region_status[key] = str(node["status"])
                self.region_probability[key] = node.get("bust_probability")
                self.rows_seen += 1
            for value in node.values():
                self._walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                self._walk(value)

    @classmethod
    def from_row(cls, row) -> "Evidence":
        """Build evidence from one bulletin row directly.

        The narration path hands the model a row instead of calling a tool, so
        it has evidence without ever going through ``observe``.
        """
        return cls().observe(dict(row))

    def any_refused(self) -> bool:
        return any(status != "OK" for status in self.region_status.values())


@lru_cache(maxsize=1)
def _region_aliases() -> dict:
    """``alias -> canonical display name`` for all 36 IMD subdivisions.

    Read from the committed subdivision config, not from the bulletin store: a
    guardrail must not depend on a 63 MB artifact that is gitignored and absent
    in a fresh checkout.  Returns {} if the config cannot be read, and
    ``check_status_grounding`` fails closed in that case.
    """
    try:
        import json

        from fbd import config

        raw = json.loads(config.SUBDIVISION_CONFIG.read_text(encoding="utf-8"))
        subdivisions = raw.get("subdivisions", {})
    except Exception:  # noqa: BLE001 - a guardrail may not raise on the serving path
        return {}

    aliases: dict = {}
    for region_id, entry in subdivisions.items():
        display = (entry or {}).get("name") or region_id
        for alias in (display, region_id, region_id.replace("_", " ")):
            aliases[alias.lower()] = display
    return aliases


def canonical_region(name) -> str:
    """Map any spelling of a subdivision to its canonical display name."""
    text = str(name).strip()
    return _region_aliases().get(text.lower(), text)


def _named_regions(text: str) -> list:
    """Canonical subdivisions this text actually names.

    This is the distinction the first version of this check got wrong. A claim
    naming a subdivision is an assertion about that row and needs evidence; a
    sentence that names none is a description of the methodology and does not.
    """
    lowered = text.lower()
    found = []
    for alias, display in _region_aliases().items():
        if alias in lowered and display not in found:
            found.append(display)
    return found


def _is_elevated(status: str, probability) -> bool:
    """True if this is a row the system wants a forecaster to look at.

    Defers to the project's own cost-optimal threshold rather than inventing a
    number for this check.  There should be exactly one definition of
    "elevated" in the system and ``quality.escalation`` already owns it.
    """
    from fbd.quality.escalation import ReviewTier, classify_tier

    tier = classify_tier(probability, ood_flag=(status != "OK"))
    return tier in (ReviewTier.REVIEW, ReviewTier.REFUSE)


def check_status_grounding(text: str, evidence: "Evidence | None") -> GuardrailReport:
    """Check claims about *status* and *direction* against what tools returned.

    Returns OK when there is nothing to check against.  With no evidence this
    check has no opinion, and a check that manufactures an opinion out of no
    data is the failure mode it exists to prevent.
    """
    if evidence is None:
        return GuardrailReport(ok=True)

    violations: list[str] = []
    named = _named_regions(text)

    if any(p.search(text) for p in _REFUSAL_CLAIM_PATTERNS):
        for name in named:
            status = evidence.region_status.get(name)
            if status is None:
                violations.append(
                    f"status_ungrounded: the text says the system declined to "
                    f"score {name}, but no tool result in this turn returned a "
                    f"row for it. Reciting the definition of a refusal is not "
                    f"evidence that this region was refused."
                )
                break
            if status == "OK":
                p = evidence.region_probability.get(name)
                detail = f", bust_probability {p:.3f}" if isinstance(p, (int, float)) else ""
                violations.append(
                    f"status_inverted: the text says the system declined to "
                    f"score {name}, but the tool reported status OK{detail}. A "
                    f"refusal means unknown risk and a scored row means "
                    f"measured risk; the two lead a forecaster to opposite "
                    f"actions."
                )
                break
        else:
            # Names no subdivision. That is a description of what a refusal is,
            # which the assistant is meant to be able to give -- unless the
            # region list could not be loaded at all, in which case we cannot
            # tell a description from an assertion and fail closed.
            if not _region_aliases() and not evidence.any_refused():
                violations.append(
                    "status_ungrounded: the text asserts a refusal and the "
                    "subdivision list could not be loaded, so the claim cannot "
                    "be checked against what the tools returned."
                )

    if any(p.search(text) for p in _LOW_RISK_CLAIM_PATTERNS):
        for name in named:
            if name not in evidence.region_status:
                continue
            status = evidence.region_status[name]
            p = evidence.region_probability.get(name)
            if _is_elevated(status, p):
                shown = f"{p:.3f}" if isinstance(p, (int, float)) else "none (refused)"
                violations.append(
                    f"direction_inverted: the text calls {name} low risk, but "
                    f"its bust probability is {shown}, at or above the "
                    f"cost-optimal review threshold. This is the inversion that "
                    f"would make a forecaster stand down on a flagged row."
                )
                break

    return GuardrailReport(ok=not violations, violations=violations)


def sanitise_untrusted(text: str, max_chars: int = 4000) -> tuple[str, GuardrailReport]:
    """Prepare retrieved/tool content for inclusion in a prompt.

    Truncates, and reports (does not silently strip) anything instruction
    shaped, so the caller can decide whether to drop the passage entirely.
    """
    clipped = text[:max_chars]
    return clipped, check_injection(clipped)
