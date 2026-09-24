"""Monsoon regime classifier producing **soft** probabilities.

The problem statement names six situations: active monsoon, break monsoon,
monsoon depression, western disturbance, orographic rainfall, coastal rainfall.
Two honest observations shape the design:

1. Four of those (active, break, depression, western disturbance) are genuine
   *daily circulation regimes* -- properties of the large-scale flow.
2. Two of them (orographic, coastal) are not daily regimes at all; they are
   *place* characteristics that become active when the flow impinges on terrain
   or a coastline.  We therefore score them as flow-on-geography interactions:
   a subdivision is in an "orographic rainfall regime" when strong low-level
   flow meets its terrain, which is both faithful to the physics and to what the
   ministry means by the phrase.

Method: **weak supervision**, which LOGIC.md sec 5.2 explicitly permits for the
MVP.  Each regime gets a physically motivated score built from standardised
ERA5 indices; the scores are turned into probabilities with a softmax.  Nothing
here is trained on bust labels, so the regime vector cannot leak the target.

Deliberately soft, never a hard label (LOGIC.md sec 5.1 / sec 5.3): regime
*uncertainty* is itself a predictor of low predictability, so we also emit the
entropy of the regime vector and feed that to the bust model, where it can raise
bust probability rather than being averaged away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from fbd import config
from fbd.features import standardise as S
from fbd.regions import masks

REGIMES = list(config.REGIMES)
# Softmax temperature.  Higher = softer.  1.0 keeps the vector genuinely soft:
# the classifier should rarely claim >0.9 on any regime, because real days are
# usually mixtures.
TEMPERATURE = 0.8
# Weight on the *static* orographic/coastal character of a subdivision, in the
# same z-score units as the circulation terms.  2.5 makes a fully orographic
# subdivision (Himachal) clearly orographic on an average day without letting it
# swamp a genuine depression signal.
PLACE_WEIGHT = 2.5
#: Local fields standardised within each subdivision before scoring.
LOCAL_FIELDS = ["moisture_flux_850", "wind_shear", "tcwv", "u850", "v850", "z500", "mslp"]


def static_attributes() -> pd.DataFrame:
    """Per-subdivision orography and coastal character from ERA5 static fields."""
    path = config.WB2_RAW / "era5" / "era5_static.nc"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run scripts/fetch_era5.py")
    ds = xr.open_dataset(path).rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.sortby("lat").sortby("lon")

    orog = (ds.geopotential_at_surface / 9.80665).transpose("lat", "lon")
    lsm = ds.land_sea_mask.transpose("lat", "lon")

    lats, lons = orog.lat.values, orog.lon.values
    w = masks.overlap_weights(lats, lons)
    sub_ids = sorted(w.subdivision_id.unique())
    W = masks.weights_to_matrix(w, len(lats), len(lons), sub_ids)

    elev, _ = masks.area_mean(orog.values[None], W, max_nan_fraction=0.99)
    land, _ = masks.area_mean(lsm.values[None], W, max_nan_fraction=0.99)

    # Terrain roughness: area-weighted variance of elevation, which is what makes
    # a subdivision orographically active rather than merely high.
    elev2, _ = masks.area_mean((orog.values**2)[None], W, max_nan_fraction=0.99)
    rough = np.sqrt(np.maximum(elev2[0] - elev[0] ** 2, 0.0))

    out = pd.DataFrame(
        {
            "subdivision_id": sub_ids,
            "elevation_m": elev[0],
            "terrain_roughness_m": rough,
            "land_fraction": land[0],
        }
    )
    # A subdivision whose grid cells are not fully land sits on a coast.
    out["coastal_index"] = np.clip(1.0 - out.land_fraction, 0.0, 1.0)
    out["orographic_index"] = out.terrain_roughness_m / out.terrain_roughness_m.max()
    return out


def regime_scores(nat: pd.DataFrame, loc: pd.DataFrame, static: pd.DataFrame,
                  fit_years=None) -> pd.DataFrame:
    """Physically motivated score per regime, per (subdivision, date).

    ``fit_years=None`` standardises over every date, as the published pipeline
    did (test years included -- the leak S1b measures); a fold passes its
    training years.
    """
    df = loc.merge(nat, on="date", how="inner").merge(
        static, on="subdivision_id", how="left"
    )

    # Standardise local fields within subdivision, so "strong flow" means strong
    # *for that place* -- 10 m/s is routine in Konkan and extreme in Vidarbha.
    z = S.standardise_within(df, LOCAL_FIELDS, "subdivision_id",
                             S.fit_mask(df.date, fit_years))
    for c in LOCAL_FIELDS:
        df[f"{c}_zl"] = z[f"{c}_zl"]

    lat_norm = df.subdivision_id.map(_subdivision_latitude()).fillna(20.0)

    s = pd.DataFrame(index=df.index)
    # Active monsoon: strong Somali jet, moist core zone, deep trough.
    s["active_monsoon"] = (
        0.5 * df.somali_jet_z + 0.5 * df.mcz_q850_z - 0.3 * df.monsoon_trough_mslp_z
    )
    # Break monsoon: the mirror image, plus a locally dry column.
    s["break_monsoon"] = (
        -0.5 * df.somali_jet_z - 0.5 * df.mcz_q850_z + 0.3 * df.monsoon_trough_mslp_z
        - 0.3 * df.tcwv_zl
    )
    # Monsoon depression: cyclonic vorticity and low pressure over the Bay.
    s["monsoon_depression"] = (
        0.6 * df.bob_vorticity_max_z - 0.6 * df.bob_mslp_min_z
    )
    # Western disturbance: negative z500 anomaly over NW India, and only
    # meaningful for northern subdivisions.
    north_weight = np.clip((lat_norm - 20.0) / 12.0, 0.0, 1.0)
    s["western_disturbance"] = north_weight * (-1.0 * df.nw_z500_z)
    # Orographic and coastal need a **static baseline term** as well as a
    # flow-modulated term.  The flow terms are standardised *within* subdivision,
    # so their time-mean is zero everywhere; multiplying them by a static index
    # scales the variance but not the mean, and Himachal would end up with the
    # same average orographic probability as West Rajasthan.  PLACE_WEIGHT
    # restores the intrinsic character: rough terrain is orographically active
    # whenever there is flow at all, and a coastline is always a coastline.
    s["orographic"] = PLACE_WEIGHT * df.orographic_index + df.orographic_index * (
        0.6 * df.moisture_flux_850_zl + 0.4 * df.u850_zl
    )
    s["coastal"] = PLACE_WEIGHT * df.coastal_index + df.coastal_index * (
        0.6 * df.tcwv_zl + 0.4 * df.moisture_flux_850_zl
    )

    s = s.fillna(0.0)
    out = df[["subdivision_id", "date"]].copy()
    for r in REGIMES:
        out[f"score_{r}"] = s[r].values
    return out


def softmax_probabilities(scores: pd.DataFrame, temperature: float = TEMPERATURE) -> pd.DataFrame:
    cols = [f"score_{r}" for r in REGIMES]
    x = scores[cols].to_numpy(dtype=float) / temperature
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)

    out = scores[["subdivision_id", "date"]].copy()
    for i, r in enumerate(REGIMES):
        out[f"regime_{r}"] = p[:, i]
    out["regime_top"] = [REGIMES[i] for i in p.argmax(axis=1)]
    out["regime_top_prob"] = p.max(axis=1)
    # Regime disagreement.  Normalised entropy in [0, 1]; high means the
    # classifier itself does not know, which LOGIC.md sec 5.3 requires we treat
    # as evidence of low predictability rather than smoothing away.
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(len(REGIMES))
    out["regime_entropy"] = ent
    return out


_SUB_LAT_CACHE: dict[str, float] | None = None


def _subdivision_latitude() -> dict[str, float]:
    global _SUB_LAT_CACHE
    if _SUB_LAT_CACHE is None:
        import geopandas as gpd

        subs = gpd.read_file(config.SUBDIVISION_GPKG, layer="subdivisions")
        _SUB_LAT_CACHE = dict(
            zip(subs.subdivision_id, subs.geometry.representative_point().y)
        )
    return _SUB_LAT_CACHE


def classify(nat: pd.DataFrame, loc: pd.DataFrame, fit_years=None) -> pd.DataFrame:
    """End-to-end: indices -> scores -> soft regime probabilities."""
    static = static_attributes()
    return softmax_probabilities(regime_scores(nat, loc, static, fit_years=fit_years))
