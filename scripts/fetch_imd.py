"""Download IMD 0.25 degree gauge-based gridded daily rainfall (NetCDF, one file per year).

The IMD server exposes the archive through an HTML form that POSTs a year to
RF25.php.  There is no documented REST API, so we reproduce the form POST.
The server is slow (~25 s for a 25 MB year) and occasionally refuses
connections, hence the generous timeout and retry loop.

IMD gauge data outranks ERA5 for rainfall over Indian land (LOGIC.md sec 3.1) --
it is built from ~6,955 stations, whereas ERA5 precipitation is model output.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402


def fetch_year(year: int, dest: Path, retries: int = 3, timeout: int = 420) -> bool:
    out = dest / f"RF25_ind{year}_rfp25.nc"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"  [skip] {year} already present ({out.stat().st_size/1e6:.1f} MB)")
        return True

    data = urllib.parse.urlencode({"RF25": str(year)}).encode()
    req = urllib.request.Request(
        config.IMD_RF25_POST_URL,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": config.IMD_RF25_REFERER,
        },
    )
    for attempt in range(1, retries + 1):
        try:
            socket.setdefaulttimeout(timeout)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            # NetCDF classic magic number.  Anything else means the server
            # returned an HTML error page -- do not silently cache garbage.
            if not body.startswith(b"CDF"):
                raise ValueError(
                    f"not a NetCDF file (got {body[:40]!r}, {len(body)} bytes)"
                )
            out.write_bytes(body)
            print(
                f"  [ok]   {year}: {len(body)/1e6:.1f} MB in {time.time()-t0:.0f}s"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - report and retry
            print(f"  [warn] {year} attempt {attempt}/{retries}: "
                  f"{type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(5 * attempt)
    print(f"  [FAIL] {year}: giving up")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(config.ALL_YEARS))
    args = ap.parse_args()

    print(f"IMD RF25 -> {config.IMD_RAW}")
    ok = [fetch_year(y, config.IMD_RAW) for y in args.years]
    n_ok = sum(ok)
    print(f"\n{n_ok}/{len(ok)} years available")
    return 0 if n_ok == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
