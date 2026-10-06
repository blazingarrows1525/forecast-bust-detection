"""The B5 serving-decision rule (docs/PREREGISTRATION_B5.md). Pure.

Spatial + ENS against the served XGBoost + ENS (D-030), both through S1c's
combiner fitted on each fold's validation year.
"""
from __future__ import annotations

YEARS = (2019, 2020, 2021, 2022)
ENS_YEARS = (2019, 2020, 2021)   # S1c's years for anything against ENS alone


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "spatial_better"
    if hi < 0:
        return "served_better"
    return "indistinguishable"


TEXT = {
    "spatial_better": "spatial + ENS outranks the served XGBoost + ENS combination "
                      "across 2019–2022",
    "served_better": "the served XGBoost + ENS combination outranks spatial + ENS "
                     "across 2019–2022",
    "indistinguishable": "spatial + ENS is not distinguishable from the served XGBoost + "
                         "ENS combination across 2019–2022",
}

#: what each verdict means for the product, as committed in the design (§4)
CONSEQUENCE = {
    "spatial_better": "eligible to be served after a separate, designed store change; "
                      "the served number is unchanged until then",
    "served_better": "the served product stays XGBoost + ENS",
    "indistinguishable": "the served product stays XGBoost + ENS",
}
