"""Every file the API reads at serve time is COPYd into the image. Pure, run in CI.

The image is built from an explicit COPY list, not the whole repo, so a file the
API reads can be in git, work from a checkout, and still be absent from every
container. `voxel_grid.json` was: the Volume view said "Voxel grid missing" on
every deployment while the local dashboard drew it.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "fbd" / "api" / "app.py"
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"

# config attribute -> repo-relative directory
DIRS = {"INTERIM": "data/interim", "ARTIFACTS": "data/artifacts"}


def _served_files() -> list[str]:
    """Repo-relative paths of the files app.py opens under config.INTERIM / ARTIFACTS."""
    src = APP.read_text(encoding="utf-8")
    out = set()
    for attr, name in re.findall(r'config\.(INTERIM|ARTIFACTS)\s*/\s*"([^"]+)"', src):
        out.add(f"{DIRS[attr]}/{name}")
    return sorted(out)


def _copy_sources() -> list[str]:
    srcs = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "COPY" or any(p.startswith("--from") for p in parts):
            continue
        args = [p for p in parts[1:] if not p.startswith("--")]
        srcs += args[:-1]
    return srcs


def _copied(path: str, sources: list[str]) -> bool:
    return any(path == s or (s.endswith("/") and path.startswith(s)) for s in sources)


def _ignored(path: str) -> bool:
    for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        pat = line.strip()
        if not pat or pat.startswith("#") or pat.startswith("!"):
            continue
        if fnmatch.fnmatch(path, pat) or path.startswith(pat.rstrip("/") + "/"):
            return True
    return False


def test_the_scan_finds_the_served_files():
    files = _served_files()
    for f in ("data/interim/regions.geojson", "data/interim/voxel_grid.json",
              "data/artifacts/bulletins.sqlite"):
        assert f in files, f


def test_every_served_file_is_copied_into_the_image():
    sources = _copy_sources()
    missing = [f for f in _served_files() if not _copied(f, sources)]
    assert not missing, f"read by the API but not COPYd by the Dockerfile: {missing}"


def test_no_served_file_is_dockerignored():
    ignored = [f for f in _served_files() if _ignored(f)]
    assert not ignored, f"read by the API but excluded by .dockerignore: {ignored}"


def _publish_paths() -> list[str]:
    text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    block = text.split("paths:", 1)[1].split("workflow_dispatch:", 1)[0]
    return re.findall(r'^\s*-\s*"([^"]+)"', block, flags=re.M)


def test_every_copied_path_triggers_a_publish():
    # A COPYd file outside publish.yml's path filter can change without
    # rebuilding the image, leaving :latest stale.
    pats = _publish_paths()

    def covered(src: str) -> bool:
        return any(src == p or (p.endswith("/**") and src.startswith(p[:-2])) for p in pats)

    missing = [s for s in _copy_sources() if not covered(s)]
    assert not missing, f"COPYd by the Dockerfile but not in publish.yml paths: {missing}"


def test_committed_interim_files_the_api_reads_exist():
    # data/artifacts/bulletins.sqlite is a Release asset, not in git (DATA.md).
    for f in _served_files():
        if f.startswith("data/interim/"):
            assert (ROOT / f).exists(), f
