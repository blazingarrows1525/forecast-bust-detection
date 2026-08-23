"""WeatherBench 2 access helpers (anonymous GCS, lazy Zarr).

IMPORTANT PERFORMANCE FACT, measured rather than assumed
--------------------------------------------------------
LOGIC.md sec 3 says "subset to the India bounding box BEFORE computing anything"
and implies that doing so keeps the transfer small.  That is only half true.
The WB2 Zarr stores are chunked as::

    total_precipitation_24hr  chunks=(1, 8, 240, 121)   # time, lead, lon, lat
    geopotential              chunks=(1, 8, 13, 240, 121)

i.e. **every chunk spans the entire globe**.  Slicing to India therefore saves
memory and local disk, but transfers exactly the same number of bytes as taking
the globe.  Measured compressed chunk sizes at 1.5 deg:

    hres/total_precipitation_24hr   0.41 MB per chunk
    hres/geopotential               7.71 MB per chunk
    ifs_ens/total_precipitation_24hr 20.5 MB per chunk  (50 members inside the chunk)
    ifs_ens/geopotential            91.5 MB per chunk

Consequences that drive the whole data design (see DECISIONS.md D-002/D-003):
  * deterministic HRES precipitation for JJAS 2016-2022 is ~2.1 GB -> affordable;
  * HRES 3-D flow fields at every lead would be ~47 GB per variable -> rejected;
  * the full IFS ENS at every init would be ~105 GB -> rejected, subsampled instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from fbd import config

_FS = None


def _fs():
    global _FS
    if _FS is None:
        import gcsfs

        _FS = gcsfs.GCSFileSystem(token="anon")
    return _FS


def open_store(relpath: str, chunks: dict | None = None) -> xr.Dataset:
    """Lazily open a WeatherBench 2 Zarr store from the public bucket."""
    import gcsfs

    mapper = gcsfs.GCSMap(f"{config.WB2_BUCKET}/{relpath}", gcs=_fs())
    return xr.open_zarr(
        mapper, consolidated=True, chunks=chunks, decode_timedelta=True
    )


def india_slice(ds: xr.Dataset, pad: float = 0.0) -> xr.Dataset:
    """Subset to the India bounding box, tolerant of latitude ordering.

    Does not reduce bytes transferred (see module docstring) but keeps arrays
    small in memory and on disk.
    """
    b = config.INDIA_BBOX
    lat = ds["latitude"].values
    lat_slice = (
        slice(b["lat_min"] - pad, b["lat_max"] + pad)
        if lat[0] < lat[-1]
        else slice(b["lat_max"] + pad, b["lat_min"] - pad)
    )
    return ds.sel(
        latitude=lat_slice,
        longitude=slice(b["lon_min"] - pad, b["lon_max"] + pad),
    )


def season_init_times(years, hour: int = 0) -> pd.DatetimeIndex:
    """JJAS initialisation datetimes at the given UTC hour."""
    stamps = []
    for y in years:
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        days = days[days.month.isin(config.SEASON_MONTHS)]
        stamps.append(days + pd.Timedelta(hours=hour))
    return pd.DatetimeIndex(np.concatenate([s.values for s in stamps]))


def lead_timedeltas(lead_days=None) -> np.ndarray:
    """prediction_timedelta values for the requested lead days.

    Convention (locked in DECISIONS.md D-004): lead day ``L`` uses
    ``prediction_timedelta = L * 24 h``.  WB2's 24 h accumulation valid at time
    ``T`` covers ``[T-24h, T]``, so with a 00 UTC initialisation lead day ``L``
    describes the 00Z-00Z calendar day ``init_date + (L-1)``.  Day 1 is therefore
    the day of issue, exactly matching IMD's Day-1..Day-10 bulletin convention,
    and Day 10 lands on the 240 h archive limit.
    """
    lead_days = lead_days or config.LEAD_DAYS
    return np.array(
        [np.timedelta64(24 * int(l), "h") for l in lead_days], dtype="timedelta64[ns]"
    )


def valid_date_for(init_time: pd.Timestamp, lead_day: int) -> pd.Timestamp:
    """Calendar day (00Z-00Z) described by (init_time, lead_day)."""
    return pd.Timestamp(init_time).normalize() + pd.Timedelta(days=lead_day - 1)
