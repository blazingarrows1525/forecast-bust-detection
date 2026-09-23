"""The S1 registration, read by machine, so the analysis cannot drift from it.

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


def check_complete(have: int, required: int) -> None:
    if have < required:
        raise RegistrationError(
            f"{have} ENS init dates for 2022 on disk, {required} registered. "
            "Re-run scripts/fetch_ens.py --year 2022; a partial season is never "
            "reported as the full one.")


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "model_better"
    if hi < 0:
        return "ens_better"
    return "indistinguishable"


VERDICT_TEXT = {
    "model_better": "the model outranks raw ENS spread over the full held-out season",
    "ens_better": "raw ENS spread outranks the model over the full held-out season",
    "indistinguishable": "the model is not distinguishable from raw ENS spread over the "
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
