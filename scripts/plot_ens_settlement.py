"""Every S1 interval on one axis, zero marked. Reads ens_settlement.json.

    PYTHONPATH=src python scripts/plot_ens_settlement.py
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
    s = json.loads((config.ARTIFACTS / "ens_settlement.json").read_text())
    p, sec, ex = s["primary"], s["secondary"], s["exploratory"]
    rows = [
        (f"PRIMARY  model − {p['comparator']} ENS spread", p, ACCENT, True),
        ("(a) model − ENS calibrated on 2021", sec["a_calibrated_2021"]["auroc"], INK, False),
        ("(b) model+ENS − ENS", sec["b_model_plus_ens"]["combined_minus_ens"], INK, False),
        ("(b) model+ENS − model", sec["b_model_plus_ens"]["combined_minus_model"], INK, False),
        ("(c) model − ENS, pooled calibration", sec["c_pooled_calibration"]["auroc"], INK, False),
    ] + [(f"exploratory: Day {k}", v, MUTED, False)
         for k, v in sorted(ex["by_lead"].items(), key=lambda kv: int(kv[0]))] + [
        (f"exploratory: {k.replace('_', ' ')} dates", v, MUTED, False)
        for k, v in ex["original_vs_new_dates"].items()]

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(rows) + 1.4))
    for i, (label, iv, colour, bold) in enumerate(rows):
        y = len(rows) - i
        ax.plot([iv["lo"], iv["hi"]], [y, y], color=colour, lw=3 if bold else 1.8)
        ax.plot(iv["point"], y, "o", color=colour, ms=7 if bold else 5)
        ax.text(-0.005 + min(r[1]["lo"] for r in rows), y, label, ha="right",
                va="center", fontsize=9, color=colour,
                fontweight="bold" if bold else "normal")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([])
    ax.set_xlabel("ΔAUROC, 95% cluster-bootstrap interval over init dates")
    ax.set_title(f"S1: {p['text']} ({p['n_init_dates']} init dates)", fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "ens_settlement"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {out}.png/.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
