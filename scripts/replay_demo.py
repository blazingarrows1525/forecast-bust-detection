"""Terminal replay of a real bust, as a narrated causal chain.

This exists as the *backup* demo: the browser UI is the primary, but the hour-33
rule assumes the venue fails you, and a script that runs from a terminal against
a local SQLite file cannot fail in the same ways a browser can.

It follows the beat structure in the strategy document: normal state -> hazard
appears -> model detects -> explanation -> truth reveal -> baseline comparison
-> and finally the honest full-year numbers, so nobody can call it a cherry-pick.

Default case: Assam & Meghalaya, 17 June 2022 -- the June 2022 Assam-Meghalaya
floods.  Chosen from the held-out test year, not hand-picked from training.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402

DB = config.ARTIFACTS / "bulletins.sqlite"

BOLD, DIM, RED, GRN, YEL, CYN, RST = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
)


def rule(ch: str = "=") -> None:
    print(ch * 78)


def beat(n: str, title: str) -> None:
    print()
    rule()
    print(f"{BOLD}[{n}] {title}{RST}")
    rule()


def bar(p: float, width: int = 40) -> str:
    if p != p:  # NaN
        return DIM + "unavailable" + RST
    fill = int(round(p * width))
    col = GRN if p < 0.10 else (YEL if p < 0.30 else RED)
    return f"{col}{'#'*fill}{DIM}{'.'*(width-fill)}{RST} {p:5.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="ASSAM_MEGHALAYA")
    ap.add_argument("--valid-date", default="2022-06-17")
    ap.add_argument("--event", default="June 2022 Assam & Meghalaya floods")
    args = ap.parse_args()

    if not DB.exists():
        print(f"missing {DB}; run scripts/generate_bulletins.py")
        return 1
    con = sqlite3.connect(DB)
    d = pd.read_sql(
        "SELECT * FROM bulletins WHERE region_id=? AND valid_date=? ORDER BY lead_day",
        con, params=(args.region, args.valid_date),
    )
    if d.empty:
        print(f"no bulletins for {args.region} valid {args.valid_date}")
        return 1
    region = d.region.iloc[0]
    obs = d.observed_rain_mm.dropna().iloc[0]

    beat("0:00", "NORMAL STATE — what the forecaster sees today")
    print(f"Region : {BOLD}{region}{RST}")
    print(f"Target : {args.valid_date}   ({args.event})")
    print(f"\n{DIM}The official bulletin gives a rainfall number for each lead day.")
    print(f"Every one of them looks equally confident.{RST}\n")
    for _, r in d.sort_values("lead_day", ascending=False).iterrows():
        print(f"   Day {int(r.lead_day):2d}  forecast {r.forecast_rain_mm:6.1f} mm/day")

    beat("1:00", "OUR LAYER — bust probability attached to the same bulletin")
    print(f"{DIM}Read bottom-up: this is the forecast getting closer in time.{RST}\n")
    print(f"   {'':7s}{'our model':<48s}{'ensemble spread baseline'}")
    for _, r in d.sort_values("lead_day", ascending=False).iterrows():
        p = r.bust_probability if pd.notna(r.bust_probability) else float("nan")
        b = r.baseline_probability
        tag = f"{DIM}[refused: OOD]{RST}" if r.status != "OK" else ""
        print(f"   Day {int(r.lead_day):2d}  {bar(p)}   {b:5.1%}  {tag}")

    band = d[(d.lead_day >= 3) & (d.lead_day <= 7) & d.bust_probability.notna()]
    if not band.empty:
        pk = band.loc[band.bust_probability.idxmax()]
        beat("2:00", f"EXPLANATION — why, at Day {int(pk.lead_day)}")
        print(f"Issued {pk.init_date}, valid {pk.valid_date}\n")
        print(f"   bust probability     {BOLD}{pk.bust_probability:.1%}{RST}")
        print(f"   80% interval         {pk.pi_low:.1%} – {pk.pi_high:.1%}")
        print(f"   confidence in that   {pk.confidence_in_estimate:.1%}")
        print(f"\n   {BOLD}Dominant factors{RST}")
        for f in json.loads(pk.dominant_factors):
            print(f"     * {f}")
        if pk.regime_json:
            reg = json.loads(pk.regime_json)
            entropy = reg.pop("entropy", None)  # not a regime, a property of the vector
            print(f"\n   {BOLD}Regime (soft probabilities){RST}")
            for k, v in sorted(reg.items(), key=lambda kv: -kv[1])[:4]:
                print(f"     {k:22s} {v:.2f}")
            if entropy is not None:
                print(f"     {DIM}regime uncertainty (entropy) {entropy:.2f} — "
                      f"higher means the regime itself is ambiguous{RST}")

    beat("2:30", "TRUTH REVEAL — what actually happened")
    fc = band.forecast_rain_mm.mean() if not band.empty else float("nan")
    print(f"   forecast (Day 3-7 mean) : {fc:6.1f} mm/day")
    print(f"   observed (IMD gauges)   : {BOLD}{obs:6.1f} mm/day{RST}")
    print(f"   error                   : {RED}{obs-fc:+6.1f} mm/day{RST}")
    nb = int(d.actual_bust.fillna(0).sum())
    print(f"\n   {RED}{BOLD}The forecast busted on {nb} of {len(d)} lead days.{RST}")

    beat("3:00", "BASELINE COMPARISON — what the standard method said")
    if not band.empty:
        print(f"   our model, Day 3-7 peak     : {band.bust_probability.max():.1%}")
        print(f"   ensemble spread, same rows  : {band.baseline_probability.max():.1%}")
        trend = d.sort_values("lead_day", ascending=False).baseline_probability
        print(
            f"\n{DIM}   As the event approached, spread FELL "
            f"({trend.iloc[0]:.1%} -> {trend.iloc[-1]:.1%}): the models were converging,\n"
            f"   which reads as growing confidence. Our model went the other way.{RST}"
        )

    beat("3:30", "NOT A CHERRY-PICK — the whole held-out year")
    res_path = config.ARTIFACTS / "results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        print(f"   Held-out year {config.TEST_YEARS[0]}, Day 3-7 decision band, "
              f"{res['n_test']:,} region-days\n")
        print(f"   {'predictor':<36s}{'AUROC':>8s}{'Brier':>9s}{'value':>8s}")
        for r in sorted(res["decision_band"], key=lambda x: -x["auroc"]):
            print(f"   {r['model']:<36s}{r['auroc']:8.3f}{r['brier']:9.4f}{r['value']:8.3f}")

    beat("4:00", "WE BROKE IT ON PURPOSE")
    for name, path in (("input dropout", "stress_dropout.csv"),
                       ("synthetic bust injection", "stress_injection.csv")):
        p = config.ARTIFACTS / path
        if p.exists():
            print(f"\n   {BOLD}{name}{RST}")
            print("   " + pd.read_csv(p).round(3).to_string(index=False).replace("\n", "\n   "))
    n_ood = con.execute(
        "SELECT COUNT(*) FROM bulletins WHERE status='OUT_OF_DISTRIBUTION'"
    ).fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM bulletins").fetchone()[0]
    print(
        f"\n   {BOLD}Out-of-distribution refusals{RST}: {n_ood:,} of {total:,} "
        f"({n_ood/total:.2%}) region-days where the system declines to give a\n"
        f"   number at all. When it does not know, it says so."
    )
    print()
    rule()
    print(f"{BOLD}The system never issues or suppresses a warning. "
          f"A human forecaster decides.{RST}")
    rule()
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
