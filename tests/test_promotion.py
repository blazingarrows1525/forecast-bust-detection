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


def test_temporal_params_are_frozen_and_hashable():
    p = PR.TEMPORAL_PARAMS
    assert p["window"] == 14 and p["verify_lag"] == 2 and p["verify_leads"] == [1, 3, 5]
    assert len(p["weather"]) == 17 and p["gru_hidden"] == 32
    assert p["hidden"] == PR.MLP_PARAMS["hidden"] and p["seeds"] == PR.MLP_PARAMS["seeds"]
    json.dumps(p)
    assert PR.params_sha256(p) == PR.params_sha256(dict(p))
    assert PR.params_sha256(dict(p, window=7)) != PR.params_sha256(p)


def test_addendum_adds_and_never_overrides():
    base = {"slate": "mlp,temporal,spatial", "mlp_params_sha256": "a"}
    got = R.merge_addendum(base, {"temporal_params_sha256": "b"}, "temporal")
    assert got == {**base, "temporal_params_sha256": "b"}
    with pytest.raises(R.RegistrationError, match="never overrides"):
        R.merge_addendum(base, {"mlp_params_sha256": "z"}, "temporal")


def test_build_candidate_gives_the_dataset_only_to_those_that_ask():
    class Plain:
        def __init__(self):
            self.got = None

    class Reader:
        needs_dataset = True

        def __init__(self, dataset=None):
            self.got = dataset

    ds = object()
    assert PM.build_candidate(Plain, ds).got is None
    assert PM.build_candidate(Reader, ds).got is ds


def test_year_statement_names_the_loser():
    assert R.year_statement(2019, -0.05, -0.01, whom="the MLP") == \
        "ENS spread outranks the MLP in 2019"
    assert R.year_statement(2019, -0.05, -0.01) == "ENS spread outranks the model in 2019"


def test_the_confirmation_year_is_outside_the_s3_years():
    assert PM.CONFIRM_YEAR == 2018 and PM.CONFIRM_YEAR not in PM.YEARS


@pytest.mark.parametrize("v", ["model_better", "ens_better", "indistinguishable"])
def test_confirmation_text_covers_every_verdict(v):
    assert "2018" in PM.CONFIRM_TEXT[v] and "MLP" in PM.CONFIRM_TEXT[v]


def test_spatial_params_are_frozen_and_hashable():
    p = PR.SPATIAL_PARAMS
    assert p["window_size"] == 13 and len(p["window_channels"]) == 10
    assert len(p["map_channels"]) == 9 and p["branch_width"] == 32
    assert p["hidden"] == PR.MLP_PARAMS["hidden"] and p["seeds"] == PR.MLP_PARAMS["seeds"]
    assert p["dropout"] == PR.MLP_PARAMS["dropout"] and p["lr"] == PR.MLP_PARAMS["lr"]
    json.dumps(p)
    assert PR.params_sha256(p) == PR.params_sha256(dict(p))
    for k, v in (("window_size", 11), ("device", "cpu"), ("conv_padding", 0),
                 ("variants", {"window_only": ["window"]})):
        assert PR.params_sha256(dict(p, **{k: v})) != PR.params_sha256(p)


def test_spatial_primary_and_variants_are_exactly_the_registered_three():
    p = PR.SPATIAL_PARAMS
    assert p["branches"] == ["window", "map"]
    assert p["variants"] == {"window_only": ["window"], "map_only": ["map"]}


def _promote():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import promote
    return promote


def test_each_addendum_carries_its_own_pins():
    pm = _promote()
    path_b, pins_b = pm.ADDENDA["temporal"]
    assert path_b.name == "PREREGISTRATION_S3B.md" and pins_b == {"mlp_json_sha256": pm.MLP_JSON}
    path_c, pins_c = pm.ADDENDA["spatial"]
    assert path_c.name == "PREREGISTRATION_S3C.md"
    assert pins_c == {"mlp_json_sha256": pm.MLP_JSON, "spatial_inputs_sha256": pm.S3C_INPUTS}
    assert pm.MANIFESTS == {"spatial": pm.S3C_INPUTS}
    assert set(pm.CANDIDATES) == set(PM.SLATE)


def test_a_changed_gridded_input_is_refused(tmp_path, monkeypatch):
    pm = _promote()
    f = tmp_path / "hres.nc"
    f.write_bytes(b"grid")
    man = tmp_path / "s3c_inputs.json"
    from fbd.model import grids as G
    G.write_manifest(man, files=[f], root=tmp_path)
    f.write_bytes(b"regridded")
    real = G.verify_manifest
    monkeypatch.setattr(pm.G, "verify_manifest", lambda path: real(path, root=tmp_path))
    monkeypatch.setattr(pm.R, "guard", lambda path, *a, **k: (
        {"spatial_params_sha256": "x"} if path.name == "PREREGISTRATION_S3C.md" else {}))
    monkeypatch.setitem(pm.MANIFESTS, "spatial", man)
    with pytest.raises(R.RegistrationError, match="pinned gridded inputs changed"):
        pm.guards("spatial")
