"""SHAP values -> plain-language meteorological reasons.

The problem statement demands "key meteorological reasons for low confidence".
That is a hard requirement, not a nice-to-have, and it is the main reason the
model is a tree ensemble rather than a neural network (LOGIC.md sec 6).

A raw SHAP number is not a reason.  "lagged_spread = 8.3, SHAP +0.04" means
nothing to a duty forecaster.  Each feature therefore carries a template that
turns its value, its percentile within the relevant comparison group, and the
sign of its SHAP contribution into a sentence a meteorologist would accept --
"successive model runs disagree strongly about rainfall here (top 4% for this
subdivision)".

Only factors that *raise* bust probability are reported by default: the panel
answers "why is confidence low here", so a list mixing in reassuring factors
would bury the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# feature -> (short label, template_high, template_low).
# {v} = raw value, {pct} = percentile phrase.
#
# Two variants because a SHAP contribution can be positive for *either*
# direction of a feature.  A single fixed wording produces self-contradictory
# sentences like "westerly flow is strong (-0.9 m/s, near normal)" or
# "disagreement is growing (bottom 10%)".  The low variant is used when the
# value sits in the bottom half of the subdivision's distribution (or is
# negative, for standardised-anomaly features).  `None` means the feature reads
# the same either way.
TEMPLATES: dict[str, tuple[str, str, str | None]] = {
    "lagged_spread": (
        "run-to-run disagreement",
        "successive model runs disagree about rainfall here ({v:.1f} mm spread, {pct})",
        "successive model runs agree unusually closely ({v:.1f} mm spread, {pct})",
    ),
    "lagged_spread_rel": (
        "relative disagreement",
        "run-to-run disagreement is large relative to the rainfall amount ({pct})",
        "run-to-run disagreement is small relative to the rainfall amount ({pct})",
    ),
    "lagged_range": (
        "forecast range",
        "the range across recent runs spans {v:.1f} mm ({pct})",
        None,
    ),
    "spread_growth": (
        "change in disagreement",
        "disagreement between runs is growing with lead time ({pct})",
        "disagreement between runs is shrinking with lead time ({pct})",
    ),
    "jumpiness": (
        "forecast instability",
        "the forecast for this day changed by {v:.1f} mm between consecutive runs ({pct})",
        "the forecast for this day barely changed between runs ({v:.1f} mm, {pct})",
    ),
    "fcst_rain_mm": (
        "forecast amount",
        "a high rainfall amount is forecast ({v:.1f} mm/day, {pct})",
        "little rainfall is forecast ({v:.1f} mm/day, {pct})",
    ),
    "lagged_mean": (
        "recent-run mean",
        "recent runs average a high amount here ({v:.1f} mm/day, {pct})",
        "recent runs average little rain here ({v:.1f} mm/day, {pct})",
    ),
    "fcst_anomaly": (
        "rainfall anomaly",
        "the forecast is {v:+.1f} mm/day above this subdivision's seasonal normal ({pct})",
        "the forecast is {v:+.1f} mm/day below this subdivision's seasonal normal ({pct})",
    ),
    "fcst_rel_to_p90": (
        "extreme relative to climatology",
        "the forecast approaches this subdivision's 90th-percentile rainfall ({pct})",
        "the forecast is far below this subdivision's 90th-percentile rainfall ({pct})",
    ),
    "fcst_prev_run": (
        "previous run",
        "the previous model run also forecast a high amount ({v:.1f} mm/day, {pct})",
        "the previous model run forecast little rain ({v:.1f} mm/day, {pct})",
    ),
    "clim_bust_rate": (
        "historically error-prone",
        "this subdivision and lead time historically bust often ({v:.1%} of days)",
        None,
    ),
    "lead_day": ("lead time", "the forecast is {v:.0f} days ahead", None),
    "moisture_flux_850": (
        "low-level moisture flux",
        "850 hPa moisture transport into the region is strong ({pct})",
        "850 hPa moisture transport into the region is weak ({pct})",
    ),
    "wind_shear": (
        "wind shear",
        "vertical wind shear between 200 and 850 hPa is large ({pct})",
        "vertical wind shear between 200 and 850 hPa is small ({pct})",
    ),
    "tcwv": (
        "atmospheric moisture",
        "the atmospheric column is unusually moist ({v:.0f} kg/m2, {pct})",
        "the atmospheric column is unusually dry ({v:.0f} kg/m2, {pct})",
    ),
    "mslp": (
        "surface pressure",
        "surface pressure over the region is high ({pct})",
        "surface pressure over the region is low ({pct})",
    ),
    "z500": (
        "mid-level flow",
        "the 500 hPa height field is anomalously high ({pct})",
        "the 500 hPa height field is anomalously low, indicating a trough ({pct})",
    ),
    "u850": (
        "low-level zonal flow",
        "850 hPa westerly flow is strong ({v:.1f} m/s, {pct})",
        "850 hPa flow is easterly or weak ({v:.1f} m/s, {pct})",
    ),
    "v850": (
        "low-level meridional flow",
        "850 hPa southerly flow is strong ({v:.1f} m/s, {pct})",
        "850 hPa flow is northerly or weak ({v:.1f} m/s, {pct})",
    ),
    "somali_jet_z": (
        "Somali jet",
        "the Somali jet is stronger than normal ({v:+.1f} sd)",
        "the Somali jet is weaker than normal ({v:+.1f} sd)",
    ),
    "bob_vorticity_max_z": (
        "Bay of Bengal vorticity",
        "cyclonic vorticity over the Bay of Bengal is elevated ({v:+.1f} sd)",
        "cyclonic vorticity over the Bay of Bengal is suppressed ({v:+.1f} sd)",
    ),
    "bob_mslp_min_z": (
        "Bay of Bengal pressure",
        "pressure over the Bay of Bengal is higher than normal ({v:+.1f} sd)",
        "a low-pressure area is present over the Bay of Bengal ({v:+.1f} sd)",
    ),
    "monsoon_trough_mslp_z": (
        "monsoon trough",
        "the monsoon trough is anomalously weak ({v:+.1f} sd)",
        "the monsoon trough is anomalously deep ({v:+.1f} sd)",
    ),
    "nw_z500_z": (
        "northwest India mid-levels",
        "500 hPa heights over northwest India are anomalously high ({v:+.1f} sd)",
        "a mid-level trough is present over northwest India ({v:+.1f} sd)",
    ),
    "india_shear_z": (
        "shear over India",
        "wind shear over India is stronger than normal ({v:+.1f} sd)",
        "wind shear over India is weaker than normal ({v:+.1f} sd)",
    ),
    "india_tcwv_z": (
        "moisture over India",
        "column moisture over India is above normal ({v:+.1f} sd)",
        "column moisture over India is below normal ({v:+.1f} sd)",
    ),
    "india_q850_z": (
        "low-level humidity over India",
        "850 hPa humidity over India is above normal ({v:+.1f} sd)",
        "850 hPa humidity over India is below normal ({v:+.1f} sd)",
    ),
    "mcz_q850_z": (
        "monsoon core-zone humidity",
        "humidity over the monsoon core zone is above normal ({v:+.1f} sd)",
        "humidity over the monsoon core zone is below normal ({v:+.1f} sd)",
    ),
    "regime_entropy": (
        "regime uncertainty",
        "the regime classifier is itself uncertain which regime applies "
        "(normalised entropy {v:.2f}) -- ambiguous regimes are less predictable",
        "the regime is unusually well defined (entropy {v:.2f})",
    ),
    "regime_monsoon_depression": (
        "monsoon depression",
        "conditions resemble a monsoon depression (p={v:.2f})", None),
    "regime_active_monsoon": (
        "active monsoon", "the monsoon is in an active phase (p={v:.2f})", None),
    "regime_break_monsoon": (
        "break monsoon", "the monsoon is in a break phase (p={v:.2f})", None),
    "regime_western_disturbance": (
        "western disturbance", "a western disturbance regime is indicated (p={v:.2f})", None),
    "regime_orographic": (
        "orographic forcing", "flow is being forced over terrain (p={v:.2f})", None),
    "regime_coastal": (
        "coastal convergence", "onshore coastal flow is indicated (p={v:.2f})", None),
}


# Concept family per feature.  The reason panel takes at most one factor from
# each family, because three restatements of "a lot of rain is forecast" is not
# an explanation -- it is the same explanation three times.  A forecaster wants
# to know: how extreme, how much do the runs disagree, and what is the flow doing.
FAMILY: dict[str, str] = {
    "fcst_rain_mm": "amount", "fcst_anomaly": "amount", "fcst_rel_to_p90": "amount",
    "lagged_mean": "amount",
    "lagged_spread": "disagreement", "lagged_spread_rel": "disagreement",
    "lagged_range": "disagreement", "spread_growth": "disagreement",
    "jumpiness": "disagreement", "fcst_prev_run": "disagreement",
    "clim_bust_rate": "history",
    "lead_day": "history",
    "moisture_flux_850": "moisture", "tcwv": "moisture", "india_tcwv_z": "moisture",
    "mcz_q850_z": "moisture", "india_q850_z": "moisture",
    "wind_shear": "dynamics", "india_shear_z": "dynamics", "z500": "dynamics",
    "mslp": "dynamics", "u850": "dynamics", "v850": "dynamics",
    "somali_jet_z": "dynamics", "monsoon_trough_mslp_z": "dynamics",
    "bob_vorticity_max_z": "dynamics", "bob_mslp_min_z": "dynamics",
    "nw_z500_z": "dynamics",
    "regime_entropy": "regime",
    "regime_monsoon_depression": "regime", "regime_active_monsoon": "regime",
    "regime_break_monsoon": "regime", "regime_western_disturbance": "regime",
    "regime_orographic": "regime", "regime_coastal": "regime",
}


def _percentile_phrase(pct: float) -> str:
    if pct >= 0.99:
        return "top 1% for this subdivision"
    if pct >= 0.95:
        return "top 5% for this subdivision"
    if pct >= 0.90:
        return "top 10% for this subdivision"
    if pct >= 0.75:
        return "upper quartile for this subdivision"
    if pct <= 0.01:
        return "bottom 1% for this subdivision"
    if pct <= 0.10:
        return "bottom 10% for this subdivision"
    if pct <= 0.25:
        return "lower quartile for this subdivision"
    return "near normal for this subdivision"


class ReasonExplainer:
    """Wraps a fitted BustModel and turns predictions into reason strings."""

    # Number of quantile knots stored per (subdivision, feature).
    N_KNOTS = 101

    def __init__(self, model, reference: pd.DataFrame):
        self.model = model
        self.features = list(model.features)
        self.reference_ = reference
        self._explainer = None
        # Percentile lookup is per subdivision, so "strong moisture flux" means
        # strong *for Konkan*, not strong compared with Rajasthan.
        #
        # Precomputed as a quantile grid rather than rescanning the reference
        # frame per row: the naive version is O(rows x features x |reference|)
        # and takes hours on 80k rows against a 200k-row reference.  A 101-knot
        # grid is ~1 MB total and turns each lookup into a searchsorted.
        self._knots: dict[tuple[str, str], np.ndarray] = {}
        self._build_quantile_grid()

    def _build_quantile_grid(self) -> None:
        cols = [f for f in self.features if f in TEMPLATES and f in self.reference_]
        if not cols:
            return
        qs = np.linspace(0.0, 1.0, self.N_KNOTS)
        for sid, grp in self.reference_.groupby("subdivision_id", sort=False):
            for f in cols:
                v = grp[f].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size < 30:
                    continue
                self._knots[(sid, f)] = np.quantile(v, qs)

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP contributions, one row per input.

        Uses XGBoost's native ``pred_contribs`` rather than the ``shap``
        package.  It is the same exact TreeSHAP algorithm, but it avoids a real
        incompatibility: shap 0.49 fails to parse XGBoost 3.x's ``base_score``
        field (serialised as ``'[5E-1]'``) and raises ValueError.  Going native
        also drops a heavy dependency from the serving path.

        The returned matrix has an extra trailing bias column, which is dropped.
        """
        import xgboost as xgb

        X = df[self.features].to_numpy(dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=list(self.features))
        contribs = self.model.booster.get_booster().predict(dm, pred_contribs=True)
        contribs = np.asarray(contribs)
        if contribs.ndim == 3:  # multi-class layout (n, n_class, n_feat+1)
            contribs = contribs[:, 1, :]
        return contribs[:, :-1]  # drop bias term

    def _percentile(self, subdivision_id: str, feature: str, value: float) -> float:
        knots = self._knots.get((subdivision_id, feature))
        if knots is None or not np.isfinite(value):
            return float("nan")
        return float(np.searchsorted(knots, value, side="right") / (len(knots) - 1))

    def reasons_for_row(
        self, row: pd.Series, shap_row: np.ndarray, k: int = 3
    ) -> list[str]:
        """Top-k factors that *raise* bust probability, as sentences."""
        order = np.argsort(-shap_row)  # most positive contribution first
        out: list[str] = []
        used_families: set[str] = set()
        for i in order:
            if len(out) >= k:
                break
            if shap_row[i] <= 0:
                break
            feat = self.features[i]
            if feat not in TEMPLATES:
                continue
            # One factor per concept family, so the panel gives a forecaster
            # three *different* kinds of reason rather than one restated.
            fam = FAMILY.get(feat, feat)
            if fam in used_families:
                continue
            _, tmpl_high, tmpl_low = TEMPLATES[feat]
            value = row.get(feat, np.nan)
            if not np.isfinite(value):
                continue
            pct = self._percentile(row.get("subdivision_id", ""), feat, value)
            # Pick the wording that matches the direction the value actually
            # sits in.  Standardised-anomaly features (_z suffix) are signed
            # about zero; everything else is judged against its own percentile.
            is_low = (value < 0) if feat.endswith("_z") else (
                np.isfinite(pct) and pct < 0.5
            )
            template = tmpl_low if (is_low and tmpl_low is not None) else tmpl_high
            phrase = _percentile_phrase(pct) if np.isfinite(pct) else "unusual"
            try:
                out.append(template.format(v=value, pct=phrase))
            except (KeyError, ValueError):
                continue
            used_families.add(fam)
        if not out:
            out.append("no single dominant factor -- risk comes from the combination of conditions")
        return out

    def explain(self, df: pd.DataFrame, k: int = 3) -> list[list[str]]:
        sv = self.shap_values(df)
        return [
            self.reasons_for_row(df.iloc[i], sv[i], k=k) for i in range(len(df))
        ]

    def global_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature -- what the model actually relies on."""
        sv = self.shap_values(df)
        imp = np.abs(sv).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.features, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
