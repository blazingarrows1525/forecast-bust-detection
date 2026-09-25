"""The S1 and S1b registrations, read by machine, so neither analysis can drift from its own.

docs/PREREGISTRATION_S1.md carries a fenced ```registration block of
``key: value`` lines. settle_ens.py refuses to run unless that file is
committed, every hashed input still matches, and the season is complete.
Dependency-free on purpose: these guards are tested in CI.
"""
from __future__ import annotations

from pathlib import Path

from fbd.evaluate import provenance as P

FENCE = "```registration"


class RegistrationError(RuntimeError):
    """The registered analysis cannot run as registered."""


def parse_registration(text: str) -> dict:
    if FENCE not in text:
        raise RegistrationError("no ```registration block found")
    body = text.split(FENCE, 1)[1].split("```", 1)[0]
    out = {}
    for line in body.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def verify_hashes(reg: dict, files: dict) -> list:
    problems = []
    for key, path in files.items():
        want = reg.get(key)
        got = P.sha256_file(path)
        if want != got:
            problems.append(f"{key}: registered {want}, file {Path(path).name} is {got}")
    return problems


def check_complete(have: int, required: int, year: int = 2022) -> None:
    if have < required:
        raise RegistrationError(
            f"{have} ENS init dates for {year} on disk, {required} registered. "
            f"Re-run scripts/fetch_ens.py --year {year}; a partial season is never "
            "reported as the full one.")


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "model_better"
    if hi < 0:
        return "ens_better"
    return "indistinguishable"


#: "ENS spread", not "raw ENS spread": the registered comparator is whichever
#: of raw and relative spread ranks busts better; the output names which.
VERDICT_TEXT = {
    "model_better": "the model outranks ENS spread over the full held-out season",
    "ens_better": "ENS spread outranks the model over the full held-out season",
    "indistinguishable": "the model is not distinguishable from ENS spread over the "
                         "full held-out season",
}


def guard(prereg: Path, files: dict, repo: Path) -> dict:
    """Every precondition, in order. Returns the parsed registration."""
    if not P.is_committed(prereg, repo=repo):
        raise RegistrationError(
            f"{Path(prereg).name} is not committed as-is. The analysis runs only "
            "against a registration that git can show existed first.")
    reg = parse_registration(Path(prereg).read_text(encoding="utf-8"))
    problems = verify_hashes(reg, files)
    if problems:
        raise RegistrationError("frozen inputs changed:\n  " + "\n  ".join(problems))
    return reg


# ------------------------------------------------------------------ S1b
BACKTEST_VERDICT_TEXT = {
    "model_better": "the edge replicates in 2019–2021",
    "ens_better": "ENS spread outranks the model in 2019–2021",
    "indistinguishable": "the model is not distinguishable from ENS spread in 2019–2021",
}


def check_params(reg: dict, sha: str) -> None:
    if reg.get("params_sha256") != sha:
        raise RegistrationError(
            f"hyperparameters changed: registered {reg.get('params_sha256')}, "
            f"the code has {sha}")


def year_statement(year: int, lo: float, hi: float, who: str = "ENS spread",
                   whom: str = "the model"):
    """The registered per-year rule: a year the comparator wins is said plainly."""
    return f"{who} outranks {whom} in {year}" if hi < 0 else None
