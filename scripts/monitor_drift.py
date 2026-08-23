"""Run drift monitoring and write an operator report.

Default comparison is train (2016-2020) vs the held-out year, which is the
honest self-test: it should reproduce the calibration drift README.md already
discloses rather than reporting a clean bill of health.

    PYTHONPATH=src python scripts/monitor_drift.py
    PYTHONPATH=src python scripts/monitor_drift.py --current-year 2021
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.mlops import drift  # noqa: E402
from fbd.model import train as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-year", type=int, default=config.TEST_YEARS[0])
    ap.add_argument("--decision-band-only", action="store_true")
    args = ap.parse_args()

    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    ds["init_date"] = pd.to_datetime(ds.init_date)
    ds["year"] = ds.init_date.dt.year

    model = T.BustModel.load(config.ARTIFACTS / "bust_model.joblib")
    features = list(model.features)

    reference = ds[ds.year.isin(config.TRAIN_YEARS)].copy()
    current = ds[ds.year == args.current_year].copy()
    if args.decision_band_only:
        current = current[current.lead_day.isin(config.DECISION_BAND)]
        reference = reference[reference.lead_day.isin(config.DECISION_BAND)]

    if current.empty:
        print(f"no rows for year {args.current_year}")
        return 1

    # Score both windows so prediction drift and calibration are measurable.
    for frame in (reference, current):
        frame["proba"] = model.predict_proba(frame)

    print(f"reference: {len(reference):,} rows ({list(config.TRAIN_YEARS)})")
    print(f"current  : {len(current):,} rows ({args.current_year})")

    report = drift.assess(
        reference=reference,
        current=current,
        features=features,
        proba_col="proba",
        label_col="bust",
    )

    print("\n" + "=" * 78)
    print(f"DRIFT VERDICT: {report.verdict.value}")
    print("=" * 78)
    for reason in report.reasons:
        print(f"  - {reason}")

    if report.calibration_ratio is not None:
        print(
            f"\ncalibration: observed {report.observed_rate:.3%} vs predicted "
            f"{report.predicted_rate:.3%}  (ratio {report.calibration_ratio:.3f})"
        )
    if report.prediction_psi is not None:
        print(f"prediction PSI: {report.prediction_psi:.4f}")

    print("\ntop drifting features by PSI:")
    for feature, psi in report.top_drifting(10):
        band = "major" if psi > drift.PSI_RETRAIN else ("moderate" if psi > drift.PSI_WATCH else "-")
        print(f"  {feature:28s} {psi:7.4f}  {band}")

    out = config.ARTIFACTS / "drift_report.json"
    payload = report.as_dict() | {
        "reference_years": list(config.TRAIN_YEARS),
        "current_year": args.current_year,
        "decision_band_only": args.decision_band_only,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")

    # Exit code is the monitoring contract: a scheduler or CI job can gate on
    # it without parsing stdout. RETRAIN is a failure; WATCH is not.
    return 2 if report.verdict is drift.Verdict.RETRAIN else 0


if __name__ == "__main__":
    sys.exit(main())
