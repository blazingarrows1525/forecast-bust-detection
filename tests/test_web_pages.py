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


# --------------------------------------------------------------------------
# The risk ramp, under colour-vision deficiency
# --------------------------------------------------------------------------
# This is a computable safety property, so it is computed rather than reviewed.
#
# The ramp this replaced ran dark-green -> green -> amber -> orange -> red.
# Under simulated protanopia its "<5%" and ">35%" bands sat at CIELAB dE 10.0:
# the safest and the most dangerous band, effectively the same colour, for
# roughly 1% of men. That is the inversion this whole project exists to
# prevent, arriving through the palette.
#
# Anyone reaching for a traffic-light ramp again will trip this test.

#: Roughly one just-noticeable difference is dE 2.3. Adjacent bands need to be
#: clearly separable; the extremes must never be near each other.
MIN_ADJACENT_DE = 14.0
MIN_EXTREME_DE = 60.0

_CVD = {
    # Brettel/Vienot LMS transforms.
    "deuteranopia": [[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]],
    "protanopia": [[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]],
    "tritanopia": [[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]],
    "normal": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
}


def _ramp_from_css() -> list:
    """The five risk stops, read out of index.html so the test tracks the page."""
    css = _read("index.html")
    root = re.search(r":root\{(.*?)\}", css, re.S).group(1)
    stops = []
    for i in range(5):
        m = re.search(rf"--c{i}\s*:\s*(#[0-9a-fA-F]{{6}})", root)
        assert m, f"--c{i} not found in :root"
        stops.append(m.group(1))
    return stops


def _simulate(hex_colour: str, kind: str):
    import numpy as np

    def srgb2lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def lin2srgb(c):
        return np.where(c <= 0.0031308, c * 12.92,
                        1.055 * np.clip(c, 0, 1) ** (1 / 2.4) - 0.055)

    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    h = hex_colour.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255
    lin = srgb2lin(rgb)
    out = np.linalg.inv(rgb2lms) @ (np.array(_CVD[kind]) @ (rgb2lms @ lin))
    seen = np.clip(lin2srgb(out), 0, 1)

    # CIELAB, D65.
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = m @ srgb2lin(seen) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, xyz ** (1 / 3), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


@pytest.mark.parametrize("kind", sorted(_CVD))
def test_risk_ramp_stays_separable_under_colour_vision_deficiency(kind: str):
    import numpy as np

    ramp = _ramp_from_css()
    lab = [_simulate(c, kind) for c in ramp]

    for i in range(len(lab) - 1):
        d = float(np.linalg.norm(lab[i] - lab[i + 1]))
        assert d >= MIN_ADJACENT_DE, (
            f"{kind}: risk bands {i} and {i + 1} ({ramp[i]}, {ramp[i + 1]}) are "
            f"dE {d:.1f} apart, below {MIN_ADJACENT_DE}. Adjacent risk levels "
            f"must be distinguishable."
        )

    extremes = float(np.linalg.norm(lab[0] - lab[-1]))
    assert extremes >= MIN_EXTREME_DE, (
        f"{kind}: the lowest and highest risk bands are dE {extremes:.1f} apart. "
        "These two must never approach each other -- confusing them inverts the "
        "meaning of the map."
    )


def test_refusal_is_a_pattern_not_a_hue():
    """§2.1 in the palette.

    Any hue puts refusal on the same visual axis as risk, which invites reading
    it as a point on the ramp. The previous purple fill was also *darker* than
    the top risk band, so a region that busts 23.4% of the time looked calmer
    than one at 35%.
    """
    html = _read("index.html")
    assert 'OOD_FILL = "url(#oodHatch)"' in html
    assert re.search(r"--ood\s*:", html) is None, (
        "refusal must not be a colour token; it is a hatch pattern"
    )
    assert "oodHatch" in html and "<pattern" in html


def test_dashboard_legend_states_that_a_refusal_is_not_low_risk():
    html = _read("index.html")
    assert "23.4" in html and "3.4" in html, (
        "the legend must say what a refusal means operationally, not just "
        "that the system declined"
    )


def test_dashboard_is_keyboard_operable():
    """45 minutes to a deadline; a dropdown per region is the wrong interaction."""
    html = _read("index.html")
    for key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Escape"):
        assert f'"{key}"' in html, f"no handler for {key}"
    assert 'id="help"' in html, "shortcuts need a discoverable overlay"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_page_is_branded_with_the_competition_id(page: Path):
    """This is a personal project; the product surface carries no entry number."""
    html = page.read_text(encoding="utf-8")
    assert "SIH" not in html, f"{page.name} still carries SIH branding"
