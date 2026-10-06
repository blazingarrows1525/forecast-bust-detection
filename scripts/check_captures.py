"""Fail if any UI capture shows overflow, a console error, an exception or a failed request (C2).

    python scripts/capture_screenshots.py --label ci
    python scripts/check_captures.py artifacts/frontend/ci/report.json

No pixel diff: software-rendered WebGL varies between runners. These are the
checks that do not: every page, at every viewport, lays out without
horizontal scroll, logs no error, throws nothing and loads every request.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED = 20  # 4 routes x 5 viewports


def problems(report: list, expected: int = EXPECTED) -> list:
    out = []
    if len(report) != expected:
        out.append(f"expected {expected} captures, got {len(report)}")
    for r in report:
        where = f"{r.get('route')} @ {r.get('viewport')}"
        if r.get("overflowX"):
            out.append(f"{where}: horizontal overflow (scrollWidth {r.get('scrollWidth')})")
        for key in ("console", "exceptions", "failed"):
            for item in r.get(key) or []:
                out.append(f"{where}: {key}: {str(item)[:200]}")
        if not r.get("title"):
            out.append(f"{where}: page has no title (did it load?)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--expected", type=int, default=EXPECTED)
    args = ap.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"::error::capture report unreadable ({exc}); the captures did not run")
        return 1
    found = problems(report, args.expected)
    for p in found:
        print(f"::error::{p}")
    print(f"{'FAIL' if found else 'OK'}: {len(report)} captures, {len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
