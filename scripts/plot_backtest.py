"""S1b: each test year's margin over ENS spread and the registered mean, zero marked.

    PYTHONPATH=src python scripts/plot_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402

INK, MUTED, ACCENT = "#1f2933", "#8b98a5", "#2a6fb0"


def main() -> int:
    s = json.loads((config.ARTIFACTS / "backtest.json").read_text())
    p, per, proxy = s["primary"], s["per_year"], s["secondary"]["proxy"]
    rows = [("PRIMARY  mean 2019–2021, model − ENS", p, ACCENT, True)]
    for y in sorted(per, key=int):
        seen = y == "2022"
        rows.append((f"{y}: model − ENS" + ("  (seen in S1)" if seen else ""),
                     per[y], MUTED if seen else INK, False))
    rows.append(("secondary: mean 2019–2021, model − lagged proxy", proxy["mean"], MUTED, False))

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.4))
    left = min(r[1]["lo"] for r in rows)
    for i, (label, iv, colour, bold) in enumerate(rows):
        y = len(rows) - i
        ax.plot([iv["lo"], iv["hi"]], [y, y], color=colour, lw=3 if bold else 1.8)
        ax.plot(iv["point"], y, "o", color=colour, ms=7 if bold else 5)
        ax.text(left - 0.005, y, label, ha="right", va="center", fontsize=9,
                color=colour, fontweight="bold" if bold else "normal")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([])
    ax.set_xlabel("ΔAUROC, 95% cluster-bootstrap interval over init dates")
    ax.set_title(f"S1b: {p['text']}", fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "backtest"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {out}.png/.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
