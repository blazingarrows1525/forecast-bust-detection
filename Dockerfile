# Offline demo image.  Serving needs only the precomputed SQLite bulletins and
# the subdivision geometry -- no raw NWP data, no network, no GPU.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Only the serving subset of the requirements: the heavy ingestion stack
# (zarr/gcsfs/dask/xgboost) is not needed to serve precomputed bulletins.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ /app/src/
COPY web/ /app/web/
COPY config/ /app/config/
COPY scripts/replay_demo.py /app/scripts/
# Precomputed artifacts: bulletins, model metrics, evidence tables.
COPY data/artifacts/ /app/data/artifacts/
COPY data/interim/imd_subdivisions.gpkg /app/data/interim/

EXPOSE 8912
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8912/api/health',timeout=4)"

CMD ["python", "-m", "uvicorn", "fbd.api.app:app", "--host", "0.0.0.0", "--port", "8912"]
