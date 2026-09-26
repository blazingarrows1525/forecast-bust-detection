"""S1b guards: dependency-free, run in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = Path(__file__).resolve().parents[1] / "docs" / "PREREGISTRATION_S1B.md"


def test_check_complete_names_the_year():
    R.check_complete(122, 122, year=2019)
    with pytest.raises(R.RegistrationError, match="for 2019 on disk"):
        R.check_complete(21, 122, year=2019)


def test_check_complete_keeps_the_s1_wording_by_default():
    with pytest.raises(R.RegistrationError, match="for 2022 on disk"):
        R.check_complete(41, 122)


def test_check_params_refuses_changed_hyperparameters():
    R.check_params({"params_sha256": PR.params_sha256()}, PR.params_sha256())
    with pytest.raises(R.RegistrationError, match="hyperparameters"):
        R.check_params({"params_sha256": "0" * 64}, PR.params_sha256())


@pytest.mark.parametrize("lo, hi, v", [(0.01, 0.05, "model_better"),
                                       (-0.05, -0.01, "ens_better"),
                                       (-0.01, 0.04, "indistinguishable")])
def test_backtest_verdict_text_covers_every_outcome(lo, hi, v):
    assert R.verdict(lo, hi) == v
    assert "2019–2021" in R.BACKTEST_VERDICT_TEXT[v]


def test_a_year_the_ensemble_wins_is_stated_plainly():
    assert R.year_statement(2020, -0.05, -0.01) == "ENS spread outranks the model in 2020"
    assert R.year_statement(2020, -0.05, 0.01) is None
    assert R.year_statement(2020, -0.05, -0.01, who="the lagged proxy") == \
        "the lagged proxy outranks the model in 2020"


@pytest.mark.skipif(not PREREG.exists(), reason="S1b not registered yet")
def test_the_s1b_registration_has_every_key():
    reg = R.parse_registration(PREREG.read_text(encoding="utf-8"))
    for key in ("dataset_sha256", "model_sha256", "params_sha256", "seed", "n_boot",
                "n_boot_per_year", "primary_years", "lead_days", "alpha", "mode",
                *(f"required_ens_dates_{y}" for y in (2019, 2020, 2021, 2022))):
        assert key in reg, key
    assert reg["params_sha256"] == PR.params_sha256()
