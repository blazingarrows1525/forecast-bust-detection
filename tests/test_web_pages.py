"""Invariants that apply to every page in ``web/``.

File-level assertions, for the same reason the shader ones are: a page cannot
assert on itself, and these are properties that fail silently. A hardcoded
number stays plausible forever; an external font keeps working right up until
the machine is offline, which is the one moment the claim matters.

Needs no bulletin store, so it runs in a fresh checkout and in CI.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd import config  # noqa: E402

WEB = config.ROOT / "web"
PAGES = sorted(WEB.glob("*.html"))


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_there_are_pages_to_check():
    """Guards the glob: an empty list would make every test below vacuous."""
    assert PAGES, f"no pages found in {WEB}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_references_no_external_origin(page: Path):
    """The property the whole offline claim rests on.

    CI enforces this too; having it here means it fails in a local test run
    rather than after a push.
    """
    external = re.findall(
        r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)',
        page.read_text(encoding="utf-8"),
    )
    assert not external, f"{page.name} references {external}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_uses_a_system_font_stack(page: Path):
    """A webfont is an external origin with extra steps."""
    html = page.read_text(encoding="utf-8")
    assert "@font-face" not in html, f"{page.name} declares a webfont"
    assert "fonts.googleapis" not in html and "fonts.gstatic" not in html


# --------------------------------------------------------------------------
# The landing page states findings, so it has a stricter rule than the rest.
# --------------------------------------------------------------------------
LANDING = "landing.html"

#: Numbers from the flagship case. Every one of these must be fetched, not
#: remembered: the archive is the source of truth, and a page that hardcodes
#: them keeps asserting the old value after the data moves, with nothing to
#: reveal it.
CASE_NUMBERS = ["70.6", "0.706", "11.2", "0.112", "115.6", "45.6", "33.3", "47.5"]


@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
@pytest.mark.parametrize("number", CASE_NUMBERS)
def test_landing_page_does_not_hardcode_the_case_numbers(number: str):
    html = _read(LANDING)
    assert number not in html, (
        f"landing.html contains the literal {number!r}. Every figure from the "
        "demo case must come from /api/convergence, so the page reports what "
        "the archive says rather than what it said when the page was written."
    )


@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
def test_landing_page_derives_its_claim_rather_than_asserting_it():
    """The claim text is computed from the rows, so it cannot go stale.

    Writing "the model stayed higher at every lead" into the markup would be a
    sentence that stays confident after it stops being true.
    """
    html = _read(LANDING)
    assert 'id="claim"' in html
    assert "r.bust_probability > r.baseline_probability" in html, (
        "the 'stayed higher' claim must be derived by comparing the rows"
    )


@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
def test_landing_page_breaks_the_line_at_a_refusal():
    """A refused lead has no probability; the chart must not bridge it.

    Drawing through the gap would assert a value the model explicitly declined
    to give, which is the §2.1 invariant in chart form.
    """
    html = _read(LANDING)
    assert "segments.push(run)" in html, "model line must be drawn in segments"
    assert "url(#hatch)" in html, "refused leads must be marked, not left blank"


@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
def test_landing_page_states_the_ensemble_margin_is_not_established():
    """D-022: the margin over a real ensemble contains zero. Say so.

    This is the one claim the project is most tempted to overstate, and a
    landing page is where overstatement usually happens.
    """
    html = _read(LANDING)
    assert "+0.025" in html and "0.058" in html, "the interval must be shown"
    assert "is not" in html or "not</strong>" in html


@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
def test_landing_page_respects_reduced_motion():
    html = _read(LANDING)
    assert "prefers-reduced-motion" in html, (
        "scroll-driven drawing must be switchable off; without this the page "
        "animates at people who asked it not to"
    )
