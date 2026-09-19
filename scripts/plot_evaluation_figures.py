"""Figures for the two claims this project is judged on.

The repo had no figures at all -- not one PNG or SVG -- while asserting in prose
that the model is calibrated and that its refusals are earned. Both claims are
measured (``results.json`` carries the ECE, LOGIC.md 11.2 carries the refusal
rates), and neither had ever been drawn. A reliability diagram is the only
honest way to show a calibration claim: a single ECE number can hide a curve
that is badly wrong at the top of the range, which is exactly where a forecaster
is looking.

    PYTHONPATH=src python scripts/plot_evaluation_figures.py

Two rules this script follows, because getting either wrong would make the
figures worse than no figures at all:

1. **Test year only (2022).** The store also holds 2021, which is the
   VAL split -- the isotonic calibrator was *fitted* on it (config.VAL_YEARS).
   Plotting a calibration curve on the data the calibrator was fitted to would
   produce a beautiful diagonal that means nothing. 2022 is touched only at
   final evaluation.
2. **The project's own binning.** Bins and ECE come from
   ``fbd.evaluate.metrics``, which uses equal-count rather than equal-width
   bins -- with a 4% base rate, equal-width bins leave the upper bins nearly
   empty and produce a dramatic-looking diagram about almost no data. Using a
   second definition here would put a number on the figure that disagrees with
   the number in results.json.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on CI or on a server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd import config  # noqa: E402
from fbd.evaluate import metrics  # noqa: E402

DB = config.ARTIFACTS / "bulletins.sqlite"
OUT_DIR = config.ROOT / "docs" / "figures"

TEST_YEAR = config.TEST_YEARS[0]
BAND = (3, 7)  # the decision band: the leads a forecaster can still act on

INK = "#1a1a1a"
MODEL_C = "#0b6fa4"
BASE_C = "#b0752a"
REFUSE_C = "#a8322d"
OK_C = "#3f7d4f"
GRID = "#d8d8d8"


def _load() -> dict:
    if not DB.exists():
        sys.exit(f"bulletin store missing: {DB}\nRun: python scripts/fetch_release_artifacts.py")
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    scored = con.execute(
        "SELECT bust_probability, baseline_probability, actual_bust FROM bulletins "
        "WHERE init_date LIKE ? AND lead_day BETWEEN ? AND ? "
        "  AND bust_probability IS NOT NULL AND actual_bust IS NOT NULL",
        (f"{TEST_YEAR}%", *BAND),
    ).fetchall()

    # The refusal comparison uses every lead: refusals are a property of the
    # atmospheric state, not of how far ahead the forecast was made.
    refusal = con.execute(
        "SELECT status, COUNT(*) AS n, AVG(actual_bust) AS rate FROM bulletins "
        "WHERE init_date LIKE ? AND actual_bust IS NOT NULL GROUP BY status",
        (f"{TEST_YEAR}%",),
    ).fetchall()
    con.close()

    if not scored:
        sys.exit(f"no scored {TEST_YEAR} rows in the store; nothing to plot")

    return {
        "y": np.array([r["actual_bust"] for r in scored], dtype=float),
        "p_model": np.array([r["bust_probability"] for r in scored], dtype=float),
        "p_base": np.array(
            [r["baseline_probability"] if r["baseline_probability"] is not None else np.nan
             for r in scored], dtype=float),
        "refusal": {r["status"]: (r["n"], r["rate"]) for r in refusal},
    }


def plot_reliability(d: dict) -> Path:
    y, p_model, p_base = d["y"], d["p_model"], d["p_base"]
    base_rate = float(y.mean())

    rc = metrics.reliability_curve(y, p_model, n_bins=10)
    ece = metrics.expected_calibration_error(y, p_model, n_bins=10)
    brier = metrics.brier(y, p_model)
    bss = metrics.brier_skill_score(y, p_model)

    ok = ~np.isnan(p_base)
    rc_base = metrics.reliability_curve(y[ok], p_base[ok], n_bins=10) if ok.any() else None

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(7.2, 7.6), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    top = max(rc.mean_predicted.max(), rc.observed_frequency.max()) * 1.12
    ax.plot([0, top], [0, top], "--", color="#888", lw=1.2, zorder=1,
            label="perfect calibration")
    ax.axhline(base_rate, color=GRID, lw=1, zorder=1)
    ax.text(top * 0.015, base_rate, f"climatological base rate {base_rate:.3f}",
            va="bottom", ha="left", fontsize=8, color="#777")

    if rc_base is not None:
        ax.plot(rc_base.mean_predicted, rc_base.observed_frequency, "s--",
                color=BASE_C, ms=5, lw=1.4, alpha=0.85, zorder=2,
                label="climatology baseline")

    ax.plot(rc.mean_predicted, rc.observed_frequency, "o-", color=MODEL_C,
            ms=7, lw=2.2, zorder=3, label="XGBoost + isotonic")

    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_ylabel("observed bust frequency", fontsize=10, color=INK)
    ax.set_title(
        f"Calibration on the held-out test year ({TEST_YEAR}), "
        f"decision band day {BAND[0]}–{BAND[1]}",
        fontsize=12, color=INK, pad=12,
    )
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # The top bin is where a forecaster is actually looking, and it is where a
    # single ECE figure hides the most.  Label the gap rather than let a low
    # ECE imply the curve sits on the diagonal everywhere.
    hi = rc.iloc[-1]
    gap = float(hi.mean_predicted - hi.observed_frequency)
    if abs(gap) > 0.01:
        ax.annotate(
            f"highest bin: predicts {hi.mean_predicted:.3f},\n"
            f"observes {hi.observed_frequency:.3f}\n"
            f"({'over' if gap > 0 else 'under'}confident by {abs(gap):.3f})",
            xy=(hi.mean_predicted, hi.observed_frequency),
            xytext=(hi.mean_predicted * 0.44, hi.observed_frequency * 1.30),
            fontsize=8.5, color=REFUSE_C, ha="left",
            arrowprops=dict(arrowstyle="->", color=REFUSE_C, lw=1),
        )

    ax.text(
        0.985, 0.04,
        f"n = {len(y):,}\nbase rate = {base_rate:.4f}\n"
        f"Brier = {brier:.4f}\nBSS = {bss:+.3f}\nECE = {ece:.4f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        color=INK, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID),
    )

    # Equal-WIDTH bins here, deliberately unlike the panel above.  The markers
    # above each carry the same number of rows by construction, so a histogram
    # over those bins is flat and says nothing -- the first version of this
    # figure had exactly that, a log axis spanning 1.988e3 to 1.989e3.  This
    # one shows how far into the tail the model goes, and on how little data.
    axh.hist(p_model, bins=np.linspace(0, top, 46), color=MODEL_C, alpha=0.8)
    axh.set_yscale("log")
    axh.set_xlabel("predicted bust probability", fontsize=10, color=INK)
    axh.set_ylabel("rows\n(log)", fontsize=9, color=INK)
    axh.grid(True, axis="y", color=GRID, lw=0.6)
    axh.set_axisbelow(True)

    fig.text(
        0.5, 0.008,
        "Equal-count bins (fbd.evaluate.metrics). 2021 is excluded: the isotonic "
        "calibrator was fitted on it.",
        ha="center", fontsize=8, color="#666",
    )

    for a in (ax, axh):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    return _save(fig, "reliability")


def plot_earned_refusal(d: dict) -> Path | None:
    refusal = d["refusal"]
    if "OUT_OF_DISTRIBUTION" not in refusal or "OK" not in refusal:
        print("  [skip] store has no refused rows in the test year")
        return None

    n_ok, rate_ok = refusal["OK"]
    n_ood, rate_ood = refusal["OUT_OF_DISTRIBUTION"]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    bars = ax.bar(
        ["scored\n(status OK)", "refused\n(out of distribution)"],
        [rate_ok, rate_ood],
        color=[OK_C, REFUSE_C], width=0.55,
    )
    for bar, rate, n in zip(bars, (rate_ok, rate_ood), (n_ok, n_ood)):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + rate_ood * 0.025,
                f"{rate:.1%}", ha="center", fontsize=14, color=INK, weight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, rate_ood * 0.035,
                f"n = {n:,}", ha="center", fontsize=9, color="white")

    ax.set_ylim(0, rate_ood * 1.25)
    ax.set_ylabel("share of forecasts that actually busted", fontsize=10, color=INK)
    ax.set_title(
        f"A refusal is not a low-risk result ({TEST_YEAR} test year)",
        fontsize=12, color=INK, pad=12,
    )
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(
        0.5, -0.17,
        f"Days the model declines to score bust {rate_ood / rate_ok:.1f}× more often "
        f"than the days it accepts.\nCollapsing them into “not flagged” would hide "
        f"the highest-risk days in the product.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555",
    )
    return _save(fig, "earned_refusal")


def _save(fig, stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {png.relative_to(config.ROOT)} (+ .svg)")
    return png


def main() -> int:
    d = _load()
    print(f"test year {TEST_YEAR}, decision band day {BAND[0]}-{BAND[1]}: "
          f"{len(d['y']):,} scored rows")
    plot_reliability(d)
    plot_earned_refusal(d)

    # Printed so the figure's numbers can be checked against results.json by
    # eye rather than taken on trust.
    print("\ncross-check against data/artifacts/results.json, "
          "row '4 XGBoost + isotonic':")
    print(f"  ECE   {metrics.expected_calibration_error(d['y'], d['p_model']):.4f}")
    print(f"  Brier {metrics.brier(d['y'], d['p_model']):.4f}")
    print(f"  BSS   {metrics.brier_skill_score(d['y'], d['p_model']):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
