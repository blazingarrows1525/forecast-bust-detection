"""C5: requirements-serve.lock stays in step with requirements-serve.txt. Pure, run in CI."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "requirements-serve.txt"
LOCK = ROOT / "requirements-serve.lock"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.split("[")[0].strip()).lower()


def _version(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _spec() -> dict:
    out = {}
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)\s*(==|>=)\s*([0-9][^\s;]*)", line)
        if m:
            out[_norm(m.group(1))] = (m.group(2), m.group(3))
    return out


def _lock() -> dict:
    """{package: (version, n_hashes)}"""
    out, current = {}, None
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line)
        if m:
            current = _norm(m.group(1))
            out[current] = [m.group(2), 0]
        elif current and "--hash=sha256:" in line:
            out[current][1] += 1
    return {k: tuple(v) for k, v in out.items()}


def test_every_locked_package_carries_hashes():
    lock = _lock()
    assert len(lock) >= 20
    assert all(n >= 1 for _v, n in lock.values()), [k for k, (_v, n) in lock.items() if n < 1]


def test_every_pin_in_the_spec_is_locked_at_a_matching_version():
    spec, lock = _spec(), _lock()
    assert spec, "no pins parsed from requirements-serve.txt"
    for name, (op, version) in spec.items():
        assert name in lock, f"{name} is in requirements-serve.txt but not in the lock"
        got = lock[name][0]
        if op == "==":
            assert got == version, f"{name}: spec {version}, lock {got}; re-run pip-compile"
        else:
            assert _version(got) >= _version(version), f"{name}: lock {got} < {version}"


def test_the_image_installs_from_the_lock_with_hashes_required():
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements-serve.lock" in docker
    assert "--require-hashes" in docker and "-r requirements-serve.lock" in docker
