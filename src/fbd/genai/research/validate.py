"""Check a model's answer against the evidence it was given. Fail closed.

Order of checks, each one able to reduce what is shown:

1. Parse: the reply must be a JSON object matching ModelOutput. One repair is
   attempted (strip code fences, take the outermost object); after that the
   reply is unusable and the user gets the sources instead of prose.
2. Citations: every [S#] in the prose and every claim's source_ids must be an
   id the server issued for THIS request. Unknown ids are removed and counted;
   a claim left with no valid source is marked unsupported.
3. Numbers: every number in the answer must appear in a source it could have
   come from (guardrails.check_numeric_grounding, fed the sources' numbers).
   A date the model was given -- today's date in the prompt, a source's
   published or retrieved date in its packet header -- is grounded too.
4. Authority: no directives (guardrails.check_authority).

Any failure of 3 or 4 withholds the prose entirely: the user sees "unable to
verify" and the source list. A partially wrong answer about rainfall is worse
than no answer, which is the conclusion D-019 reached about narration.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from fbd.genai.guardrails import check_authority, check_numeric_grounding
from fbd.genai.research.schemas import ClaimEvidence, ModelOutput

CITE_RE = re.compile(r"\[(S\d{1,2})\]")
# Models also write "[S1, S2]". Split it into "[S1][S2]" before the citation
# check, so each id is checked against the packet and shown as its own link.
GROUP_RE = re.compile(r"\[(S\d{1,2}(?:\s*[,;]\s*S\d{1,2})+)\]")


def _split_groups(s: str) -> str:
    return GROUP_RE.sub(lambda m: "".join(f"[{x.strip()}]" for x in re.split(r"[,;]", m.group(1))), s)

# Dates. Models state today's date and a source's date ("issued 18 October
# 2026") as a matter of course, and the day is often not in the source text,
# so the number check used to withhold those answers. A whole date expression that
# matches a date the model was given is now taken out before the number check.
# Anything else is judged exactly as before: an invented date still withholds
# the answer, and today being the 18th does not ground "18 mm".
_MONTH_NO: dict[str, int] = {}
for _i, _m in enumerate(("January", "February", "March", "April", "May", "June", "July", "August",
                         "September", "October", "November", "December"), 1):
    for _form in {_m, _m[:3], "Sept" if _i == 9 else _m[:3]}:
        _MONTH_NO[_form] = _MONTH_NO[_form.upper()] = _i
_MONTH = "(" + "|".join(sorted(_MONTH_NO, key=len, reverse=True)) + r")(?![A-Za-z])\.?"
_ORD = r"(?:st|nd|rd|th)?"
# (pattern, groups -> candidate (year or None, month, day) readings)
_DATE_RES = (
    # 2026-10-18, also inside an ISO timestamp
    (re.compile(r"(?<![\d.])(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)"),
     lambda g: [(int(g[0]), int(g[1]), int(g[2]))]),
    # 18-10-2026, 18/10/2026, 18.10.2026: day first (IMD's form) or month first
    (re.compile(r"(?<![\d.])(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)"),
     lambda g: [(int(g[2]), int(g[1]), int(g[0])), (int(g[2]), int(g[0]), int(g[1]))]),
    # 18 October 2026, 18th Oct, 18 of October
    (re.compile(rf"(?<![\d.])(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MONTH},?(?:\s+(\d{{4}})(?!\d))?"),
     lambda g: [(int(g[2]) if g[2] else None, _MONTH_NO[g[1]], int(g[0]))]),
    # October 18, 2026; Oct 18th
    (re.compile(rf"(?<![A-Za-z]){_MONTH}\s+(\d{{1,2}}){_ORD}(?![\d.])(?:,?\s+(\d{{4}})(?!\d))?"),
     lambda g: [(int(g[2]) if g[2] else None, _MONTH_NO[g[0]], int(g[1]))]),
)


def given_dates(texts) -> set[tuple[int | None, int, int]]:
    """The calendar dates written in ``texts`` (the prompt's date, packet headers)."""
    out: set[tuple[int | None, int, int]] = set()
    for t in texts:
        for rx, read in _DATE_RES if t else ():
            for m in rx.finditer(t):
                out.update(c for c in read(m.groups()) if 1 <= c[1] <= 12 and 1 <= c[2] <= 31)
    return out


def _strip_given_dates(text: str, given: set[tuple[int | None, int, int]]) -> str:
    """Blank out each date expression that names a given date; leave the rest."""
    def drop(m: re.Match, read) -> str:
        for y, mo, d in read(m.groups()):
            if any(gm == mo and gd == d and (y is None or gy is None or gy == y) for gy, gm, gd in given):
                return " "
        return m.group(0)
    for rx, read in _DATE_RES if given else ():
        text = rx.sub(lambda m, read=read: drop(m, read), text)
    return text


@dataclass
class Checked:
    ok: bool
    output: ModelOutput | None
    claims: list[ClaimEvidence] = field(default_factory=list)
    removed_citations: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    repaired: bool = False
    cited_ids: list[str] = field(default_factory=list)


def parse(text: str) -> tuple[ModelOutput | None, bool]:
    """Strict parse, then exactly one repair attempt. Returns (output, repaired)."""
    for attempt, candidate in enumerate((text, _repair(text))):
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return ModelOutput.model_validate(data), attempt == 1
        except (json.JSONDecodeError, ValidationError):
            continue
    return None, False


def _repair(text: str) -> str | None:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = t.find("{"), t.rfind("}")
    return t[start:end + 1] if 0 <= start < end else None


def check(text: str, valid_ids: set[str], source_numbers: list[float], dates=()) -> Checked:
    """``dates``: the strings the model was given dates in (today's date, each
    source's published and retrieved times). A date matching one is grounded."""
    out, repaired = parse(text)
    if out is None:
        return Checked(False, None, violations=["unparseable: the model did not return the required JSON"])

    removed: list[str] = []

    def strip_unknown(s: str) -> str:
        def keep(m: re.Match) -> str:
            if m.group(1) in valid_ids:
                return m.group(0)
            removed.append(m.group(1))
            return ""
        return CITE_RE.sub(keep, s)

    out.answer = strip_unknown(_split_groups(out.answer))
    out.key_points = [strip_unknown(_split_groups(k)) for k in out.key_points]

    claims: list[ClaimEvidence] = []
    for i, c in enumerate(out.claims):
        if not isinstance(c, dict) or not isinstance(c.get("text"), str):
            continue
        ids = [s for s in (c.get("source_ids") or []) if isinstance(s, str)]
        good = [s for s in ids if s in valid_ids]
        removed.extend(s for s in ids if s not in valid_ids)
        claims.append(ClaimEvidence(claim_id=f"C{i + 1}", claim_text=c["text"][:500], source_ids=good,
                                    support_level="cited" if good else "unsupported",
                                    notes=None if good else "no valid source id"))

    violations: list[str] = []
    prose = "\n".join([out.answer, *out.key_points])
    # Citation markers are ids, not claims: "[S12]" must not count as a number.
    numeric = check_numeric_grounding(_strip_given_dates(CITE_RE.sub("", prose), given_dates(dates)),
                                      source_numbers)
    if not numeric.ok:
        violations.extend(numeric.violations)
    authority = check_authority(prose)
    if not authority.ok:
        violations.extend(authority.violations)
    cited = sorted(set(CITE_RE.findall(out.answer)) | {s for c in claims for s in c.source_ids},
                   key=lambda s: int(s[1:]))
    return Checked(not violations, out, claims, removed, violations, repaired, cited)
