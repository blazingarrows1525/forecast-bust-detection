# Offline serving image.
#
# Serving needs only the precomputed SQLite bulletins, the precomputed geo
# assets and the FastAPI app -- no raw NWP data, no geo stack, no model, no
# network, no GPU.
#
# Optimisations (D-020), each with a measurable reason:
#   * multi-stage: build tools and pip caches never reach the runtime layer.
#   * no geopandas/pyogrio/pyproj/shapely: the two geo endpoints read
#     precomputed JSON, so ~44 MB of wheels (measured) plus their bundled
#     GDAL/GEOS/PROJ native code leave the image.
#   * venv copied wholesale: one cache-friendly layer instead of pip metadata
#     scattered through site-packages.
#   * non-root runtime user: a container that cannot write to its own code.
#   * dependency layer before source: editing a .py does not re-run pip.

# ---------------------------------------------------------------- build stage
FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Dependencies first so source edits do not invalidate the pip layer.
COPY requirements-serve.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements-serve.txt \
 && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /opt/venv -name '*.pyc' -delete

# -------------------------------------------------------------- runtime stage
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

# Unprivileged runtime user. Nothing in the serving path writes to disk except
# the SQLite override log, so the app owns its data dir and nothing else.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 fbd

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=fbd:fbd src/ /app/src/
COPY --chown=fbd:fbd web/ /app/web/
COPY --chown=fbd:fbd config/ /app/config/
COPY --chown=fbd:fbd scripts/replay_demo.py /app/scripts/

# Precomputed artifacts: bulletins, metrics, evidence tables, and the geo
# assets that replace the GeoPackage + geo stack.
COPY --chown=fbd:fbd data/artifacts/ /app/data/artifacts/
COPY --chown=fbd:fbd data/interim/regions.geojson /app/data/interim/
COPY --chown=fbd:fbd data/interim/centroids.json /app/data/interim/

# fbd.config creates the full data tree at import time (raw/, processed/,
# raw/imd, raw/shapes, raw/wb2 -- none of which are COPYd, because serving
# needs none of them).  --chown above only covers the directories COPY itself
# created, so /app/data and the absent subdirectories would be root-owned and
# the first import as `fbd` would die with PermissionError before the app ever
# answered /api/health.  Create them here, owned by the runtime user.
RUN mkdir -p /app/data/raw/imd /app/data/raw/shapes /app/data/raw/wb2 \
             /app/data/interim /app/data/processed /app/data/artifacts \
 && chown -R fbd:fbd /app/data

USER fbd

EXPOSE 8912

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8912/api/health',timeout=4)"

CMD ["python", "-m", "uvicorn", "fbd.api.app:app", "--host", "0.0.0.0", "--port", "8912"]
