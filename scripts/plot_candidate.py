"""S3: one candidate's margin over the XGBoost incumbent, per year and on average.

    PYTHONPATH=src python scripts/plot_candidate.py --candidate mlp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import promotion as PM  # noqa: E402

INK, MUTED, ACCENT = "#1f2933", "#8b98a5", "#2a6fb0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    name = ap.parse_args().candidate
    s = json.loads((config.ARTIFACTS / "candidates" / f"{name}.json").read_text(encoding="utf-8"))
    label = PM.DISPLAY[name]
    p, per, ens = s["primary"], s["per_year"], s["secondary"]["ens"]
    rows = [(f"PRIMARY  mean 2019–2022, {label} − XGBoost ({1 - p['alpha']:.2%})", p, ACCENT, True)]
    rows += [(f"{y}: {label} − XGBoost", per[y], INK, False) for y in sorted(per, key=int)]
    rows.append((f"secondary: mean 2019–2021, {label} − ENS spread", ens["mean"], MUTED, False))

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.4))
    left = min(r[1]["lo"] for r in rows)
    for i, (text, iv, colour, bold) in enumerate(rows):
        y = len(rows) - i
        ax.plot([iv["lo"], iv["hi"]], [y, y], color=colour, lw=3 if bold else 1.8)
        ax.plot(iv["point"], y, "o", color=colour, ms=7 if bold else 5)
        ax.text(left - 0.005, y, text, ha="right", va="center", fontsize=9, color=colour,
                fontweight="bold" if bold else "normal")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([])
    ax.set_xlabel("ΔAUROC, cluster-bootstrap interval over init dates")
    ax.set_title(f"S3: {p['text']}", fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / f"candidate_{name}"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {out}.png/.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
