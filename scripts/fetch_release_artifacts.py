"""Download the large precomputed artifacts that ship as GitHub Release assets.

Only `bulletins.sqlite` (~60 MB) is distributed this way -- it is too large to
keep in git history but is what the offline dashboard serves. It is fully
regenerable from the committed `dataset.parquet` + `bust_model.joblib` via
`scripts/generate_bulletins.py`, so this downloader is a convenience, not a
dependency.

The download is checksum-verified: a truncated or tampered asset is rejected
rather than silently served to the dashboard. See DATA.md for the policy.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "blazingarrows1525/forecast-bust-detection"
TAG = "v0.1.0"

# (release asset name, destination path relative to repo root, expected sha256)
ASSETS = [
    (
        "bulletins.sqlite",
        "data/artifacts/bulletins.sqlite",
        "5cbb61350c2ba63dbcf2dc00bc1d13b9a3d2f9a70e66d8317121cf996e6f241c",
    ),
]

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(name: str, dest_rel: str, expected: str) -> bool:
    dest = ROOT / dest_rel
    if dest.exists() and _sha256(dest) == expected:
        print(f"  [skip] {dest_rel} already present and verified")
        return True

    url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get]  {url}")
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed github.com URL
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] download error: {exc}")
        print(f"         you can regenerate this file instead with:\n"
              f"         PYTHONPATH=src python scripts/generate_bulletins.py")
        return False

    got = _sha256(dest)
    if got != expected:
        dest.unlink(missing_ok=True)
        print(f"  [FAIL] checksum mismatch for {name}\n"
              f"         expected {expected}\n         got      {got}\n"
              f"         file deleted; do not trust a mismatched asset.")
        return False

    print(f"  [ok]   {dest_rel}  ({dest.stat().st_size/1e6:.1f} MB, checksum verified)")
    return True


def main() -> int:
    print(f"Fetching release assets from {REPO} @ {TAG}")
    ok = all(fetch_one(*a) for a in ASSETS)
    if ok:
        print("\nAll assets present and verified. The offline dashboard can now run:")
        print("  docker compose up -d   # or")
        print("  PYTHONPATH=src python -m uvicorn fbd.api.app:app --port 8912")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
