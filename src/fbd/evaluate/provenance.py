"""Hashes and git state, so a registered analysis can prove what it ran on."""
from __future__ import annotations

import hashlib
import subprocess  # nosec B404 - fixed argv, no shell
from pathlib import Path

from fbd import config


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_committed(path, repo: Path = config.ROOT) -> bool:
    """True if ``path`` is tracked by git and has no uncommitted changes."""
    path = Path(path)
    tracked = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=repo, capture_output=True, text=True)
    if tracked.returncode != 0:
        return False
    dirty = subprocess.run(  # nosec B603 B607
        ["git", "status", "--porcelain", "--", str(path)],
        cwd=repo, capture_output=True, text=True)
    return dirty.returncode == 0 and dirty.stdout.strip() == ""
