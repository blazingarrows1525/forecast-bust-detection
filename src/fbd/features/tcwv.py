"""Total column water vapour from specific humidity on pressure levels (B2). Pure numpy.

The WB2 ``hres_t0`` archive (the operational HRES initial state) has no total
column water vapour, and 11 of the 30 analysis-derived inputs depend on it
(``tcwv``, ``india_tcwv`` and its anomaly and tendencies, and every regime
probability through the classifier). This derives it from what the archive
does carry: specific humidity on 13 levels (50-1000 hPa) and surface pressure.

    TCWV = (1 / g) * integral of q dp, from the top level down to the surface

The integral is the trapezoid rule on the levels above the ground. The
segment that the surface cuts is integrated to the surface, with q at the
surface interpolated linearly in pressure between the bracketing levels;
below the lowest level (surface pressure above 1000 hPa), q is held at its
1000 hPa value. Above the top level (50 hPa) q is a few parts per million and
is ignored: about 0.002 kg m-2 against a monsoon column of 40-70.

An approximation, validated before use (docs/PREREGISTRATION_B2.md): 13
levels resolve the moist boundary layer only coarsely, so the derived column
is checked against ERA5's own TCWV on the same grid.
"""
from __future__ import annotations

import numpy as np

G = 9.80665  # m s-2, the value ECMWF uses


def column_water_vapour(q, levels_hpa, ps_pa, axis: int = 1) -> np.ndarray:
    """kg m-2. ``q`` in kg/kg with its level axis at ``axis``; ``ps_pa`` in Pa,
    shaped like ``q`` without that axis; ``levels_hpa`` in any order."""
    q = np.moveaxis(np.asarray(q, dtype=np.float64), axis, -1)
    p = np.asarray(levels_hpa, dtype=np.float64) * 100.0
    order = np.argsort(p)
    p, q = p[order], q[..., order]
    ps = np.asarray(ps_pa, dtype=np.float64)
    if ps.shape != q.shape[:-1]:
        raise ValueError(f"surface pressure {ps.shape} does not match q {q.shape[:-1]}")
    if np.any(ps <= p[0]):
        raise ValueError("surface pressure at or above the top level")
    n = len(p)

    # q at the surface: linear in p between the levels that bracket it
    j = np.clip((p[None, :] < ps.reshape(-1, 1)).sum(axis=1).reshape(ps.shape), 1, n - 1)
    p_lo, p_hi = p[j - 1], p[j]
    q_lo = np.take_along_axis(q, (j - 1)[..., None], -1)[..., 0]
    q_hi = np.take_along_axis(q, j[..., None], -1)[..., 0]
    w = np.clip((ps - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    q_s = q_lo + w * (q_hi - q_lo)

    total = np.zeros(ps.shape)
    for k in range(n - 1):
        top, bottom = p[k], p[k + 1]
        above = ps > top                       # the segment starts above the ground
        whole = bottom <= ps                   # and ends above it too
        q_end = np.where(whole, q[..., k + 1], q_s)
        p_end = np.minimum(bottom, ps)
        seg = 0.5 * (q[..., k] + q_end) * (p_end - top)
        total += np.where(above, seg, 0.0)
    below = ps > p[-1]                         # surface below the lowest level
    total += np.where(below, q[..., -1] * (ps - p[-1]), 0.0)
    return total / G
