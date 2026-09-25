"""The S3 promotion rule, registered in docs/PREREGISTRATION_S3.md. Pure.

Three candidate families are declared before any exists. Each is judged
against the same incumbent -- the S1b XGBoost fold models -- on the mean
within-year AUROC margin over 2019-2022, at a Bonferroni-corrected level so
that scoring them weeks apart cannot inflate the chance of a false promotion.
"""
from __future__ import annotations

from fbd.evaluate.registration import RegistrationError

SLATE = ("mlp", "temporal", "spatial")
DISPLAY = {"mlp": "MLP", "temporal": "temporal model", "spatial": "spatial model"}
ALPHA_FAMILY = 0.05
YEARS = (2019, 2020, 2021, 2022)
ENS_YEARS = (2019, 2020, 2021)


def alpha_per_candidate() -> float:
    return ALPHA_FAMILY / len(SLATE)


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "promoted"
    if hi < 0:
        return "incumbent_better"
    return "indistinguishable"


def verdict_text(v: str, name: str) -> str:
    who = f"the {DISPLAY[name]}"
    return {
        "promoted": f"{who} outranks the XGBoost incumbent across 2019–2022",
        "incumbent_better": f"the XGBoost incumbent outranks {who} across 2019–2022",
        "indistinguishable": f"{who} is not distinguishable from the XGBoost incumbent "
                             "across 2019–2022",
    }[v]


def check_candidate(name: str, reg: dict) -> None:
    if name not in SLATE:
        raise RegistrationError(f"{name!r} is not in the declared slate {SLATE}; "
                                "a new candidate needs a new registration")
    if f"{name}_params_sha256" not in reg:
        raise RegistrationError(f"{name!r} has no registered parameter hash; commit its "
                                "addendum before scoring it")
