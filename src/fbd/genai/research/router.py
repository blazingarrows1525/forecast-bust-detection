"""Deterministic routing, before any model is involved.

The route decides what may run: the project's own records, its documentation,
the web, or nothing until the user says where and when. It is decided by
explicit rules over the question and the attached context, not by asking a
model to classify itself -- a model's view of whether a question is
safety-critical is exactly the opinion this layer must not depend on.

Rules are checked in a fixed order, most consequential first:

1. safety vocabulary                  -> SAFETY_CRITICAL (official sources first)
2. forecast vs observation, with a row -> DATA_COMPARISON (computed here, not by a model)
3. a project record or project terms   -> PROJECT_EXPLANATION
4. how-do-I questions about the app    -> HOW_TO_USE_PRODUCT
5. current / latest / official markers -> CURRENT_WEB_RESEARCH
6. stable science vocabulary           -> SCIENTIFIC_BACKGROUND
7. anything else                       -> OUT_OF_SCOPE_OR_AMBIGUOUS (one question back)

Current and safety questions need a place. If none is named, the route asks for
one rather than assuming the user's location; the browser's location is never
requested.
"""
from __future__ import annotations

import re
from functools import lru_cache

from fbd.genai.research.schemas import ContextRef, Intent, RouteDecision


def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)


SAFETY = _rx(
    r"evacuat\w*", r"warnings?", r"alerts?", r"red alert", r"orange alert", r"advisor(y|ies)",
    r"cyclone", r"landslides?", r"flash floods?", r"floods? warning", r"dam release", r"cloudburst",
    r"emergency", r"imminent", r"is it safe", r"should i (travel|go|leave|stay)", r"danger(ous)?",
    r"rescue", r"disaster",
)
ACT = re.compile(r"\b(can you|could you|please|go ahead and)?\s*(send|issue|publish|broadcast|notify|"
                 r"cancel|post|email|sms|text)\b.{0,40}\b(alert|warning|message|sms|email|notice|officials?|"
                 r"public|people|residents)\b", re.IGNORECASE)
COMPARE = _rx(r"compare", r"comparison", r"versus", r"vs\.?", r"observed", r"observation",
              r"how much rain (fell|was)", r"actual(ly)? (fell|rain)", r"forecast (and|vs\.?|versus) obs\w*")
PROJECT = _rx(
    r"flag(ged)?", r"this (reading|forecast|row|cell|column|region|subdivision|number)",
    r"bust(s|ed)?", r"bust probability", r"p\(bust\)", r"review queue", r"review tier", r"refus\w+",
    r"out[- ]of[- ]distribution", r"ood", r"served (number|probability)", r"baseline",
    r"spread baseline", r"why (is|was) .* (flagged|high|low|refused)", r"uncertainty score",
    r"confidence in (the )?estimate", r"prediction interval", r"this (system|model|product|app)",
    r"the model", r"calibrat\w+", r"occlusion", r"reasons?",
)
# "What does a refusal mean?" is about the method even with a reading attached;
# "why was THIS flagged?" is about the reading.
METHOD = _rx(r"what does .{0,60} mean", r"what is (a|an|the)", r"how is .{0,60} (defined|computed|calculated|measured)",
             r"in general", r"definition", r"what happens when", r"how does the (system|model) work")
THIS = _rx(r"this", r"here", r"that (reading|forecast|row|column|region)", r"flagged")
HOW_TO = _rx(r"how do i", r"how to", r"where (is|do i|can i)", r"shortcut", r"keyboard",
             r"button", r"toggle", r"which view", r"open (the )?(map|columns|volume)",
             r"what does the .* (button|view|page) do")
CURRENT = _rx(r"today", r"tonight", r"tomorrow", r"yesterday", r"this week", r"latest", r"current(ly)?",
              r"right now", r"now", r"recent(ly)?", r"news", r"official", r"live", r"20(2[3-9]|3\d)")
SCIENCE = _rx(r"monsoon", r"ensemble", r"coriolis", r"rossby", r"forecast skill", r"nwp", r"reanalysis",
              r"era5", r"hres", r"ecmwf", r"auroc", r"brier", r"rainfall", r"precipitation",
              r"what is", r"what are", r"explain", r"why does", r"how does", r"depression", r"western disturbance")

# Places a current or safety question can be scoped to without guessing. The
# subdivisions come from the committed config; states and union territories
# are listed here because they are what people actually type.
_STATES = (
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "delhi", "jammu", "kashmir", "ladakh", "puducherry", "chandigarh",
    "andaman", "nicobar", "lakshadweep", "mumbai", "kolkata", "chennai", "bengaluru", "bangalore",
    "hyderabad", "guwahati", "bhubaneswar", "pune", "ahmedabad", "jaipur", "lucknow", "patna",
    "india", "konkan", "vidarbha", "marathwada", "saurashtra", "kutch", "rayalaseema",
)


@lru_cache(maxsize=1)
def _places() -> tuple[str, ...]:
    names = set(_STATES)
    try:
        from fbd.genai.guardrails import _region_aliases
        names.update(a for a in _region_aliases() if len(a) > 3)
    except Exception:  # noqa: BLE001 - routing must not fail on a missing config
        pass
    return tuple(sorted(names, key=len, reverse=True))


def find_place(text: str) -> str | None:
    lowered = text.lower()
    for name in _places():
        if re.search(r"\b" + re.escape(name) + r"\b", lowered):
            return name
    return None


def route(message: str, context_refs: list[ContextRef] | None = None,
          web_mode: str = "auto") -> RouteDecision:
    refs = context_refs or []
    has_ref = bool(refs)
    place = find_place(message)
    web_allowed = web_mode != "off"

    def decide(intent: Intent, rationale: str, *, web=False, internal=False, safety="none",
               time="timeless", tools=(), clarification=None) -> RouteDecision:
        if web_mode == "on" and intent in (Intent.SCIENTIFIC_BACKGROUND, Intent.PROJECT_EXPLANATION):
            web = True
        return RouteDecision(
            intent=intent, needs_web_search=web and web_allowed, needs_internal_data=internal,
            safety_level=safety, required_geo_scope=place, time_scope=time,
            allowed_tools=list(tools) + (["web_search", "fetch_page"] if web and web_allowed else []),
            rationale=rationale, clarification=clarification)

    # Asking the assistant to DO something -- send, issue, publish, notify --
    # is answered plainly: it is read-only and has no such tool (LOGIC.md 1.3).
    if ACT.search(message):
        return decide(Intent.OUT_OF_SCOPE_OR_AMBIGUOUS, "asks the assistant to take an action",
                      safety="elevated",
                      clarification="I cannot send, issue, publish or cancel anything: this assistant is "
                                    "read-only and has no messaging or alerting tool. Warnings are issued by "
                                    "the India Meteorological Department and State Disaster Management "
                                    "Authorities. I can explain a reading or find their published advisories.")

    if SAFETY.search(message):
        if not place:
            return decide(Intent.SAFETY_CRITICAL, "safety vocabulary; no place named",
                          web=True, safety="critical", time="current",
                          clarification="Which state, district or city is this about? Advisories are "
                                        "issued for specific areas, and this assistant will not guess yours.")
        return decide(Intent.SAFETY_CRITICAL, "safety vocabulary with a named place",
                      web=True, safety="critical", time="current")

    if has_ref and COMPARE.search(message):
        return decide(Intent.DATA_COMPARISON, "forecast/observation comparison on an attached record",
                      internal=True, time="historical", tools=["get_bulletin"])

    about_method = bool(METHOD.search(message)) and not THIS.search(message)
    if has_ref and not about_method:
        return decide(Intent.PROJECT_EXPLANATION, "attached project record",
                      internal=True, time="historical", tools=["get_bulletin", "search_project_docs"])
    if has_ref or PROJECT.search(message):
        return decide(Intent.PROJECT_EXPLANATION,
                      "question about the method" if has_ref else "project vocabulary",
                      internal=True, time="timeless", tools=["search_project_docs"])

    if HOW_TO.search(message):
        return decide(Intent.HOW_TO_USE_PRODUCT, "question about using the product",
                      internal=True, tools=["search_project_docs"])

    if CURRENT.search(message):
        if not place:
            return decide(Intent.CURRENT_WEB_RESEARCH, "asks for current information; no place named",
                          web=True, safety="elevated", time="current",
                          clarification="Which place should I look up? Name a state, district or city.")
        return decide(Intent.CURRENT_WEB_RESEARCH, "asks for current information", web=True,
                      safety="elevated", time="current")

    if SCIENCE.search(message):
        return decide(Intent.SCIENTIFIC_BACKGROUND, "stable science vocabulary",
                      internal=True, web=(web_mode == "on"), tools=["search_project_docs"])

    return decide(Intent.OUT_OF_SCOPE_OR_AMBIGUOUS, "no rule matched",
                  clarification="I can explain a reading in this product, compare its forecast with "
                                "what was observed, explain the method, or, if web search is enabled, "
                                "look up current public information for a named place. Which do you need?")
