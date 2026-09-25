"""The README and the decision log state the S1 verdict the data gives.

Written because the previous ENS margin lived in five places (README, two
decision entries, the landing page, a logic file) and nothing checked they
agreed with each other or with the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402

SETTLEMENT = config.ARTIFACTS / "ens_settlement.json"
pytestmark = pytest.mark.skipif(not SETTLEMENT.exists(), reason="S1 not settled yet")


def _interval() -> str:
    p = json.loads(SETTLEMENT.read_text())["primary"]
    return f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"


@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_settled_interval(doc):
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert _interval() in text, f"{doc} does not state the settled interval {_interval()}"


def test_readme_no_longer_leads_with_the_40_date_margin_as_current():
    text = (config.ROOT / "README.md").read_text(encoding="utf-8")
    p = json.loads(SETTLEMENT.read_text())["primary"]
    assert f"{p['n_init_dates']} init dates" in text or f"{p['n_init_dates']} dates" in text


BACKTEST = config.ARTIFACTS / "backtest.json"


@pytest.mark.skipif(not BACKTEST.exists(), reason="S1b not run yet")
@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_backtest_interval(doc):
    p = json.loads(BACKTEST.read_text())["primary"]
    s = f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert s in text, f"{doc} does not state the S1b interval {s}"


MLP = config.ARTIFACTS / "candidates" / "mlp.json"


@pytest.mark.skipif(not MLP.exists(), reason="S3a not scored yet")
@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_mlp_interval(doc):
    p = json.loads(MLP.read_text(encoding="utf-8"))["primary"]
    s = f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert s in text, f"{doc} does not state the S3a interval {s}"


CONFIRM = config.ARTIFACTS / "confirm_2018.json"


@pytest.mark.skipif(not CONFIRM.exists(), reason="S3a-C not run yet")
@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_2018_confirmation(doc):
    p = json.loads(CONFIRM.read_text(encoding="utf-8"))["primary"]
    s = f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert s in text, f"{doc} does not state the 2018 confirmation {s}"


COMBINATION = config.ARTIFACTS / "combination.json"


@pytest.mark.skipif(not COMBINATION.exists(), reason="S1c not run yet")
@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_combination(doc):
    p = json.loads(COMBINATION.read_text(encoding="utf-8"))["primary"]
    s = f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert s in text, f"{doc} does not state the S1c interval {s}"
