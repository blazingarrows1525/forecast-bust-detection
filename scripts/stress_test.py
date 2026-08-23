"""Break it on purpose: robustness evidence, not a victory lap.

Three tests, all mandated by LOGIC.md sec 7.1 / sec 8.3 / sec 16:

1. **Synthetic bust injection.**  Scale a real forecast by a known factor so the
   realised error is forced across the bust threshold, then ask whether the
   detector fires.  Produces a sensitivity curve -- *how large must the error be
   before we catch it?* -- rather than a single pass/fail.

2. **30% input dropout.**  Blank 30% of feature values at random and confirm the
   system degrades gracefully instead of collapsing or crashing.

3. **Calibration under dropout.**  A model that stays sharp but loses
   calibration under degraded inputs is more dangerous than one that admits
   uncertainty, so ECE is re-checked, not just AUROC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.ood import detector as OOD  # noqa: E402
from fbd.labels import bust as L  # noqa: E402


def injection_test(model, te: pd.DataFrame) -> pd.DataFrame:
    """Scale forecast rainfall by a factor; does predicted bust risk respond?

    Only rows that did NOT originally bust are perturbed, so any rise in flagged
    rate is attributable to the injected error rather than to pre-existing risk.
    We recompute the features that genuinely depend on the forecast amount --
    leaving the others untouched would let the model 'notice' an inconsistency
    that would not exist in a real corrupted forecast.
    """
    base = te[te.bust == 0].copy()
    rows = []
    for factor in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]:
        d = base.copy()
        d["fcst_rain_mm"] = d.fcst_rain_mm * factor
        # Features that are functions of the forecast amount must move with it.
        if "clim_fcst_mean" in d:
            d["fcst_anomaly"] = d.fcst_rain_mm - d.clim_fcst_mean
        if "clim_obs_p90" in d:
            d["fcst_rel_to_p90"] = d.fcst_rain_mm / (d.clim_obs_p90 + 1.0)
        if "lagged_mean" in d:
            d["lagged_mean"] = d.lagged_mean * factor
            d["lagged_spread"] = d.lagged_spread * factor
            d["lagged_range"] = d.lagged_range * factor
            d["lagged_spread_rel"] = d.lagged_spread / (d.lagged_mean + 1.0)

        p = model.predict_proba(d)
        # What the error *would* be against the unchanged observation.
        new_abs_err = (d.fcst_rain_mm - d.obs_rain_mm).abs()
        would_bust = (
            (new_abs_err >= d.effective_threshold)
            & (L.rain_category(d.fcst_rain_mm) != L.rain_category(d.obs_rain_mm))
            & (np.maximum(d.fcst_rain_mm, d.obs_rain_mm) >= L.MODERATE_THRESHOLD)
        )
        rows.append(
            {
                "scale_factor": factor,
                "median_injected_error_mm": float(
                    (new_abs_err - base.abs_error).median()
                ),
                "frac_now_truly_bust": float(would_bust.mean()),
                "mean_predicted_prob": float(p.mean()),
                "flagged_rate_at_0.10": float((p >= 0.10).mean()),
                "flagged_rate_at_0.20": float((p >= 0.20).mean()),
                "detection_rate_on_injected": float((p[would_bust] >= 0.10).mean())
                if would_bust.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def dropout_test(model, te: pd.DataFrame, fractions=(0.0, 0.1, 0.3, 0.5)) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for frac in fractions:
        d = te.copy()
        if frac > 0:
            X = d[model.features].to_numpy(dtype=float)
            mask = rng.random(X.shape) < frac
            X[mask] = np.nan
            d[model.features] = X
        p = model.predict_proba(d)
        r = M.evaluate(d.bust, p, label=f"dropout {frac:.0%}")
        r["dropout"] = frac
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    tr, va, te = T.split_frames(ds)
    tr, te = tr.dropna(subset=["bust"]), te.dropna(subset=["bust"])
    model = T.BustModel.load(config.ARTIFACTS / "bust_model.joblib")

    print("=" * 78)
    print("TEST 1 — SYNTHETIC BUST INJECTION (held-out year, originally non-bust rows)")
    print("=" * 78)
    inj = injection_test(model, te)
    print(inj.round(4).to_string(index=False))
    print(
        "\nReading: as the injected error grows, both the fraction that truly\n"
        "becomes a bust and the model's flagged rate should rise together.\n"
        "'detection_rate_on_injected' is recall on the rows the injection\n"
        "actually pushed over the bust threshold."
    )

    print("\n" + "=" * 78)
    print("TEST 2 — INPUT DROPOUT (graceful degradation)")
    print("=" * 78)
    dro = dropout_test(model, te)
    print(
        dro[["dropout", "auroc", "brier", "ece", "pod", "far", "cost_per_1000", "value"]]
        .round(4).to_string(index=False)
    )
    d30 = dro[dro.dropout == 0.3].iloc[0]
    d0 = dro[dro.dropout == 0.0].iloc[0]
    print(
        f"\nAt 30% dropout: AUROC {d0.auroc:.3f} -> {d30.auroc:.3f} "
        f"({100*(d30.auroc-d0.auroc)/d0.auroc:+.1f}%), ECE {d0.ece:.4f} -> {d30.ece:.4f}."
    )

    print("\n" + "=" * 78)
    print("TEST 3 — OOD REFUSAL IS EARNED")
    print("=" * 78)
    ood = OOD.MahalanobisOOD().fit(tr, model.features)
    p = model.predict_proba(te)
    flag = ood.is_ood(te)
    print(f"threshold={ood.threshold_:.2f}  refused {flag.sum()}/{len(te)} ({flag.mean():.2%})")
    print(OOD.validate(te, p, flag).round(4).to_string(index=False))
    print(
        "\nRefusal is only defensible if refused rows are genuinely harder.\n"
        "If bust_rate and mean_abs_error were similar across the two groups,\n"
        "the detector would be refusing at random and should be removed."
    )

    out = config.ARTIFACTS
    inj.to_csv(out / "stress_injection.csv", index=False)
    dro.to_csv(out / "stress_dropout.csv", index=False)
    print(f"\nsaved -> {out/'stress_injection.csv'}, {out/'stress_dropout.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
