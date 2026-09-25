"""S1 machinery: provenance, the registration, the verdict, and the statistics.

The provenance, registration and verdict parts are dependency-free and run in
CI. The statistics need scikit-learn and skip where it is not installed.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd.evaluate import provenance as P  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   cwd=repo, check=True, capture_output=True)


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"forecast" * 1000)
    assert P.sha256_file(f) == hashlib.sha256(b"forecast" * 1000).hexdigest()


def test_is_committed_tracks_state(tmp_path):
    _git(tmp_path, "init", "-q")
    f = tmp_path / "PREREG.md"
    f.write_text("plan")
    assert not P.is_committed(f, repo=tmp_path), "untracked is not committed"
    _git(tmp_path, "add", "PREREG.md")
    _git(tmp_path, "commit", "-q", "-m", "register")
    assert P.is_committed(f, repo=tmp_path)
    f.write_text("plan, edited after seeing data")
    assert not P.is_committed(f, repo=tmp_path), "an edit must be visible"


# ---------------------------------------------------------- registration
from fbd.evaluate import registration as R  # noqa: E402

REG = """# Registration

text

```registration
model_sha256: aaa
dataset_sha256: bbb
seed: 20260919
n_boot: 10000
required_ens_dates_2022: 122
```
"""


def test_parse_registration_block():
    reg = R.parse_registration(REG)
    assert reg["seed"] == "20260919" and reg["required_ens_dates_2022"] == "122"


def test_parse_registration_requires_the_block():
    with pytest.raises(R.RegistrationError):
        R.parse_registration("# no block here")


def test_verify_hashes_names_the_mismatch(tmp_path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"m")
    reg = {"model_sha256": hashlib.sha256(b"m").hexdigest()}
    assert R.verify_hashes(reg, {"model_sha256": f}) == []
    f.write_bytes(b"retrained")
    assert "model_sha256" in R.verify_hashes(reg, {"model_sha256": f})[0]


def test_incomplete_season_is_refused():
    R.check_complete(122, 122)
    with pytest.raises(R.RegistrationError):
        R.check_complete(121, 122)


@pytest.mark.parametrize("lo,hi,want", [
    (0.001, 0.05, "model_better"),
    (-0.05, -0.001, "ens_better"),
    (-0.01, 0.04, "indistinguishable"),
    (0.0, 0.04, "indistinguishable"),      # touching zero is not excluding it
])
def test_verdict_mapping(lo, hi, want):
    assert R.verdict(lo, hi) == want
    assert R.VERDICT_TEXT[want]


def test_guard_refuses_an_uncommitted_registration(tmp_path):
    _git(tmp_path, "init", "-q")
    prereg = tmp_path / "PREREGISTRATION_S1.md"
    prereg.write_text(REG)
    with pytest.raises(R.RegistrationError, match="committed"):
        R.guard(prereg, {}, repo=tmp_path)


def test_guard_refuses_changed_inputs(tmp_path):
    _git(tmp_path, "init", "-q")
    model = tmp_path / "model.joblib"
    model.write_bytes(b"m")
    prereg = tmp_path / "PREREGISTRATION_S1.md"
    prereg.write_text(REG.replace("aaa", hashlib.sha256(b"other").hexdigest()))
    _git(tmp_path, "add", prereg.name)
    _git(tmp_path, "commit", "-q", "-m", "register")
    with pytest.raises(R.RegistrationError, match="model_sha256"):
        R.guard(prereg, {"model_sha256": model}, repo=tmp_path)


# ------------------------------------------------------------- statistics
def _synthetic(n_dates=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for s in range(20):
            for lead in (3, 4, 5, 6, 7):
                y = float(rng.random() < 0.05)
                rows.append({"init_date": pd.Timestamp("2022-06-01") + pd.Timedelta(days=d),
                             "subdivision_id": f"S{s}", "lead_day": lead, "bust": y,
                             "ens_spread": rng.gamma(2, 1) + 1.5 * y,
                             "ens_mean": 5.0, "p_model": 0.02 + 0.5 * y * rng.random()})
    df = pd.DataFrame(rows)
    df["ens_spread_rel"] = df.ens_spread / (df.ens_mean + 1.0)
    return df


def test_primary_is_deterministic_and_finds_a_clear_edge():
    pytest.importorskip("sklearn")
    from fbd.evaluate import settle as S

    df = _synthetic()
    y = df.bust.to_numpy(float)
    clusters = df.init_date.dt.strftime("%Y-%m-%d").to_numpy()
    name, comp = S.choose_comparator(y, df.ens_spread.to_numpy(float),
                                     df.ens_spread_rel.to_numpy(float))
    a = S.primary(y, df.p_model.to_numpy(float), comp, clusters, n_boot=300, seed=7)
    b = S.primary(y, df.p_model.to_numpy(float), comp, clusters, n_boot=300, seed=7)
    assert a == b, "same seed, same result"
    assert a["verdict"] in R.VERDICT_TEXT
    assert a["n_init_dates"] == 60


def test_comparator_is_the_stronger_ens_variant():
    pytest.importorskip("sklearn")
    from fbd.evaluate import settle as S

    y = np.array([0, 0, 1, 1], float)
    strong, weak = np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.9, 0.1, 0.2, 0.8])
    assert S.choose_comparator(y, strong, weak)[0] == "raw"
    assert S.choose_comparator(y, weak, strong)[0] == "relative"


def _fit_2021(n_dates=40):
    df = _synthetic(n_dates=n_dates, seed=3)
    df["init_date"] = df.init_date - pd.Timedelta(days=365)      # 2021 season
    return df


def test_secondary_calibrated_fits_on_2021_only():
    pytest.importorskip("sklearn")
    from fbd.evaluate import settle as S

    test = _synthetic()
    fit = pd.concat([_fit_2021(), _fit_2021().assign(
        init_date=lambda d: d.init_date - pd.Timedelta(days=365))])   # 2021 + 2020
    out = S.secondary_calibrated(test, fit, test.p_model.to_numpy(float),
                                 test.init_date.dt.strftime("%Y-%m-%d").to_numpy(),
                                 n_boot=200, seed=7)
    assert out["fit_rows_2021"] == len(_fit_2021()), "2020 rows must not enter the fit"
    for k in ("auroc", "brier", "bss", "cost_per_1000"):
        assert {"point", "lo", "hi", "excludes_zero"} <= set(out[k])


def test_secondary_combined_reports_both_comparisons():
    pytest.importorskip("sklearn")
    from fbd.evaluate import settle as S

    test, fit21 = _synthetic(), _fit_2021()
    out = S.secondary_combined(test, fit21, test.p_model.to_numpy(float),
                               fit21.p_model.to_numpy(float),
                               test.init_date.dt.strftime("%Y-%m-%d").to_numpy(),
                               n_boot=200, seed=7)
    assert set(out["coefficients"]) == {"logit_p_model", "log1p_ens_spread", "intercept"}
    assert "combined_minus_ens" in out and "combined_minus_model" in out
    assert "in-sample for calibration" in out["caveat"]
