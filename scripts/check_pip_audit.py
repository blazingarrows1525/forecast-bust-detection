"""Fail the build on any known vulnerability in the serving dependencies (C3).

    pip-audit -r requirements-serve.txt --format json -o pip-audit.json || true
    python scripts/check_pip_audit.py pip-audit.json

pip-audit's own exit code cannot tell "found a vulnerability" from "could not
run", and its JSON carries no severity. So the gate is strict and explicit:
every finding fails the build unless ``config/pip_audit_allowlist.json``
names it with a reason and an expiry date. An expired exception fails too,
so nothing is waived forever. A missing or unreadable report fails: an audit
that did not run is not a clean audit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "config" / "pip_audit_allowlist.json"


def findings(report: dict) -> list:
    """(package, version, vulnerability id, aliases) for every reported vulnerability."""
    out = []
    for dep in report.get("dependencies", []):
        for v in dep.get("vulns", []):
            out.append((dep["name"], dep.get("version", "?"), v["id"],
                        tuple(v.get("aliases", []))))
    return out


def evaluate(report: dict, allowlist: list, today: dt.date) -> list:
    """The problems that fail the build; empty means pass."""
    problems = []
    live = {}
    for e in allowlist:
        for key in ("id", "package", "reason", "expires"):
            if not e.get(key):
                problems.append(f"allowlist entry {e!r} lacks {key!r}")
        try:
            expires = dt.date.fromisoformat(str(e.get("expires")))
        except ValueError:
            problems.append(f"allowlist entry {e.get('id')!r}: bad expiry {e.get('expires')!r}")
            continue
        if expires < today:
            problems.append(f"allowlist entry {e['id']} for {e.get('package')} expired on "
                            f"{expires}; fix the dependency or renew with a new reason")
        else:
            live[(str(e.get("package", "")).lower(), e["id"])] = e
    for name, version, vid, aliases in findings(report):
        ids = {vid, *aliases}
        if not any((name.lower(), i) in live for i in ids):
            problems.append(f"{name} {version}: {vid} (aliases {', '.join(aliases) or 'none'})")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    args = ap.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"::error::pip-audit report unreadable ({exc}); the audit did not run")
        return 1
    if not report.get("dependencies"):
        print("::error::pip-audit report lists no dependencies; the audit did not run")
        return 1
    allowlist = json.loads(args.allowlist.read_text(encoding="utf-8"))
    problems = evaluate(report, allowlist, dt.date.today())
    for p in problems:
        print(f"::error::{p}")
    n = len(report["dependencies"])
    print(f"{'FAIL' if problems else 'OK'}: {n} serving dependencies audited, "
          f"{len(findings(report))} finding(s), {len(problems)} blocking")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
