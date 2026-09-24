"""S1b folds and hyperparameters: pure, so CI checks them without data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.model import params as PR  # noqa: E402


def test_the_2022_fold_is_the_published_split():
    f = F.FOLDS[2022]
    assert f.train == config.TRAIN_YEARS
    assert f.val == config.VAL_YEARS
    assert (f.test,) == config.TEST_YEARS


def test_folds_are_an_expanding_window():
    years = sorted(F.FOLDS)
    assert years == [2019, 2020, 2021, 2022]
    for a, b in zip(years, years[1:]):
        assert F.FOLDS[b].train == F.FOLDS[a].train + F.FOLDS[a].val


def test_every_fold_trains_then_calibrates_then_tests():
    for f in F.FOLDS.values():
        assert max(f.train) < min(f.val) <= max(f.val) < f.test
        assert not set(f.train) & set(f.val)


def test_primary_years_exclude_the_year_s1_has_seen():
    assert F.PRIMARY_YEARS == (2019, 2020, 2021)
    assert 2022 not in F.PRIMARY_YEARS


def test_no_fold_file_can_overwrite_the_frozen_dataset():
    frozen = (config.PROCESSED / "dataset.parquet").resolve()
    for year in F.FOLDS:
        for mode in F.MODES:
            for p in (F.fold_path(year, mode), F.model_path(year, mode)):
                assert p.resolve() != frozen
                assert p.parent == F.FOLD_DIR


def test_legacy_mode_fits_regimes_on_every_date_and_strict_on_training_years():
    f = F.FOLDS[2019]
    assert F.regime_fit_years(f, "legacy") is None
    assert F.regime_fit_years(f, "strict") == (2016, 2017)


def test_split_labels_reproduce_the_published_split():
    years = np.arange(2016, 2023)
    got = F.split_labels(years, config.TRAIN_YEARS, config.VAL_YEARS, config.TEST_YEARS)
    old = np.where(np.isin(years, config.TEST_YEARS), "test",
                   np.where(np.isin(years, config.VAL_YEARS), "val", "train"))
    assert got.tolist() == old.tolist()


def test_split_labels_exclude_years_after_the_test_year():
    f = F.FOLDS[2019]
    got = F.split_labels(np.arange(2016, 2023), f.train, f.val, (f.test,))
    assert got.tolist() == ["train", "train", "val", "test",
                            "excluded", "excluded", "excluded"]


def test_params_hash_is_stable_and_sensitive():
    assert PR.params_sha256() == PR.params_sha256(dict(PR.DEFAULT_PARAMS))
    changed = dict(PR.DEFAULT_PARAMS, max_depth=PR.DEFAULT_PARAMS["max_depth"] + 1)
    assert PR.params_sha256(changed) != PR.params_sha256()


def test_params_exclude_the_data_dependent_class_weight():
    assert "scale_pos_weight" not in PR.DEFAULT_PARAMS
