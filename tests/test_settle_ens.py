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
