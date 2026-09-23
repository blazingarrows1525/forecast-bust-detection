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


def _root_stops(page: str) -> list:
    css = _read(page)
    root = re.search(r":root\{(.*?)\}", css, re.S).group(1)
    return [re.search(rf"--c{i}\s*:\s*(#[0-9a-fA-F]{{6}})", root).group(1).lower()
            for i in range(5)]


def test_column_view_uses_the_dashboard_ramp():
    """One risk encoding across the product.

    command.html kept the green -> red ramp after the dashboard was fixed:
    protanopia dE 10.0 between its safest and most dangerous bands. The CVD
    test above covers index.html; equality carries that guarantee here.
    """
    assert _root_stops("command.html") == _root_stops("index.html")
    html = _read("command.html")
    for old in ("#1a4d2e", "#3d7c47", "#c9a227", "#e07b39", "#c0392b", "#7d5bbe"):
        assert old not in html.lower(), f"old ramp / refusal colour {old} still present"


def test_column_view_refusal_is_a_stripe_not_a_hue():
    html = _read("command.html")
    assert re.search(r"--ood\s*:", html) is None, "refusal must not be a colour token"
    assert "function refusalTexture()" in html and "map: refusalTexture()" in html
    assert "wireframe: isOOD" not in html, "a wireframe reads as less there than a solid low column"
    assert "23.4%" in html and "3.4%" in html


def test_column_view_is_seen_from_the_south():
    """It defaulted to the north side: correct handedness, India upside down."""
    html = _read("command.html")
    assert "theta: Math.PI/2," in html and "orbit.theta=Math.PI/2;" in html
    assert "theta: -Math.PI/2" not in html and "orbit.theta=-Math.PI/2" not in html


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


# --------------------------------------------------------------------------
# Override UI
# --------------------------------------------------------------------------
# POST /api/override is the one write in the whole product, and the reason
# field is the entire point of it: overrides are immutable and stored with a
# user so a decision can be reconstructed after an event. A prefilled or
# optional reason would quietly destroy that.
def test_override_reason_is_required_and_never_prefilled():
    html = _read("index.html")
    m = re.search(r'<textarea id="ovrReason"[^>]*>', html)
    assert m, "no reason field"
    tag = m.group(0)
    assert "required" in tag and 'minlength="3"' in tag
    # An empty element: anything between the tags would be a default answer.
    assert '<textarea id="ovrReason"' in html
    assert re.search(r'<textarea id="ovrReason"[^>]*>\s*</textarea>', html), (
        "the reason field must start empty; a prefilled reason is not a reason"
    )
    assert "reason.length < 3" in html, "client-side guard missing"


def test_dismissing_a_refused_row_warns_that_it_is_not_low_risk():
    """The one genuinely dangerous action in this UI.

    "Dismiss" on a refused row says "nothing to see here" about the highest-risk
    class of row on the map -- refused region-days busted 23.4% of the time
    against 3.4% for scored ones. The forecaster is still the authority and
    this does not block them; it states what is being dismissed.
    """
    html = _read("index.html")
    assert 'action === "dismiss" && r.status !== "OK"' in html, (
        "the warning must fire on dismiss-of-a-refusal specifically, not on "
        "every override -- a warning that always fires is ignored"
    )
    warn = html[html.index('action === "dismiss" && r.status !== "OK"'):]
    warn = warn[:warn.index("} else {")]
    assert "23.4" in warn and "3.4" in warn
    assert "elevated" in warn


def test_override_ui_states_that_nothing_public_changes():
    """LOGIC.md 1.3: the system never issues or suppresses a warning."""
    # Whitespace-normalised: the assertion is about what the page says, not
    # about where the source happens to wrap.
    html = re.sub(r"\s+", " ", _read("index.html"))
    assert "does not change any public warning" in html
    assert "Immutable, and stored with your name" in html


# --------------------------------------------------------------------------
# Satellite imagery (D-023)
# --------------------------------------------------------------------------
# The only external origin in the product. Two properties have to hold, and
# both are the kind that rot quietly: it must stay opt-in, and it must stay
# date-matched rather than "live".
IMAGERY_ORIGIN = "https://gibs.earthdata.nasa.gov"


def test_imagery_is_off_by_default():
    """The offline property is the default, not a mode you can select.

    CI asserts this too. Duplicated deliberately: this is the single line that
    decides whether "runs with the network unplugged" is true of a fresh
    checkout.
    """
    html = _read("index.html")
    assert "enabled: false" in html, "the imagery layer must default to off"
    assert "IMAGERY.enabled = on" in html, "it must only turn on via the toggle"


def test_imagery_is_confined_to_the_dashboard():
    """No other page may reach the network at all."""
    for page in PAGES:
        if page.name == "index.html":
            continue
        assert IMAGERY_ORIGIN not in page.read_text(encoding="utf-8"), (
            f"{page.name} must stay fully offline"
        )


def test_imagery_date_comes_from_the_row_not_from_the_clock():
    """The distinction the whole feature turns on.

    Showing today's satellite over a 2022 risk field would imply the two
    describe the same moment. The tile date is the VALID DATE of the lead on
    screen, read off the row rather than computed, so there is one definition
    of what "Day N" means.
    """
    html = _read("index.html")
    assert "function validDateForLead" in html
    assert "r.valid_date" in html
    body = html[html.index("function validDateForLead"):]
    body = body[:body.index("function imageryNote")]
    for forbidden in ("new Date()", "Date.now()", "toISOString"):
        assert forbidden not in body, (
            f"the tile date must not come from the clock ({forbidden})"
        )


def test_imagery_says_it_is_not_current_conditions():
    html = re.sub(r"\s+", " ", _read("index.html"))
    assert "not current conditions" in html.lower()
    assert "gaps between orbital passes" in html, (
        "unexplained black bands on a risk map invite the worst reading"
    )


def test_imagery_credits_nasa_and_degrades_when_unreachable():
    html = _read("index.html")
    assert "NASA EOSDIS GIBS" in html, "GIBS requires attribution"
    assert "attributionControl:true" in html
    assert 'on("tileerror"' in html, "offline is the expected case, not an error"
    assert "setImagery(false)" in html, "a broken basemap must switch itself off"
