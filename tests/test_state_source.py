"""B2: the analysed-state source switch and the scoring script's guards."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_the_default_source_is_era5_everywhere():
    e5 = pytest.importorskip("fbd.features.era5")
    import build_dataset as BD

    assert set(e5.SOURCES) == {"era5", "analysis_t0"}
    assert e5.SOURCES["era5"][0] == e5.ERA5_DIR
    for fn in (e5.national_indices, e5.national_daily, e5.subdivision_fields, e5._open):
        assert inspect.signature(fn).parameters["source"].default == "era5"
    assert inspect.signature(BD.build_fold).parameters["state_source"].default == "era5"
    assert inspect.signature(BD.attach_state_features).parameters["source"].default == "era5"


def test_an_unknown_or_missing_source_is_refused(tmp_path, monkeypatch):
    e5 = pytest.importorskip("fbd.features.era5")
    with pytest.raises(ValueError, match="unknown state source"):
        e5._open("2d", [2016], source="gfs")
    monkeypatch.setitem(e5.SOURCES, "analysis_t0", (tmp_path, "analysis_t0"))
    with pytest.raises(FileNotFoundError, match="fetch_analysis_t0.py"):
        e5._open("2d", [2016], source="analysis_t0")


def test_a_candidate_fold_never_overwrites_a_pinned_one(monkeypatch):
    import score_b2 as SB
    from fbd.evaluate import folds as F

    target = F.FOLD_DIR / f"fold_2019_strict_{SB.SOURCE}.parquet"
    monkeypatch.setattr(F, "fold_path", lambda y, m: target)
    with pytest.raises(RuntimeError, match="pinned fold"):
        SB.candidate_fold(2019, F.FOLD_DIR)


def test_the_pinned_inputs_are_the_14_seasons_and_the_static_fields():
    import score_b2 as SB

    files = SB.input_files()
    assert len(files) == 15 and len(set(files)) == 15
    assert files[-1].name == "era5_static.nc"
    assert all("analysis_t0_" in f.name for f in files[:-1])
