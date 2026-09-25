"""S3 promotion rule: pure, run in CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import promotion as PM  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402


def test_the_slate_is_declared_in_full():
    assert PM.SLATE == ("mlp", "temporal", "spatial")
    assert set(PM.DISPLAY) == set(PM.SLATE)


def test_bonferroni_over_the_whole_slate():
    assert PM.ALPHA_FAMILY == 0.05
    assert PM.alpha_per_candidate() == pytest.approx(0.05 / 3)


def test_years():
    assert PM.YEARS == (2019, 2020, 2021, 2022)
    assert PM.ENS_YEARS == (2019, 2020, 2021)


@pytest.mark.parametrize("lo, hi, v", [(0.001, 0.02, "promoted"),
                                       (-0.02, -0.001, "incumbent_better"),
                                       (-0.01, 0.01, "indistinguishable"),
                                       (0.0, 0.01, "indistinguishable")])
def test_verdict(lo, hi, v):
    assert PM.verdict(lo, hi) == v
    text = PM.verdict_text(v, "mlp")
    assert "MLP" in text and "2019–2022" in text


def test_check_candidate_refuses_undeclared_and_unregistered():
    reg = {"mlp_params_sha256": "x"}
    PM.check_candidate("mlp", reg)
    with pytest.raises(R.RegistrationError, match="not in the declared slate"):
        PM.check_candidate("lstm", reg)
    with pytest.raises(R.RegistrationError, match="no registered parameter hash"):
        PM.check_candidate("temporal", reg)


def test_mlp_params_are_frozen_and_hashable():
    p = PR.MLP_PARAMS
    assert p["hidden"] == [128, 64] and p["dropout"] == 0.2 and len(p["seeds"]) == 5
    json.dumps(p)  # serialisable, so the hash is well defined
    assert PR.params_sha256(p) == PR.params_sha256(dict(p))
    assert PR.params_sha256(dict(p, dropout=0.3)) != PR.params_sha256(p)


def test_year_statement_names_the_loser():
    assert R.year_statement(2019, -0.05, -0.01, whom="the MLP") == \
        "ENS spread outranks the MLP in 2019"
    assert R.year_statement(2019, -0.05, -0.01) == "ENS spread outranks the model in 2019"
