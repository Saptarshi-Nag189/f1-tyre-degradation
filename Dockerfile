# Serving image for the F1 tyre degradation model.
#
# Serving only. Collection and training stay outside: they need the FIA timing
# API, a warm 500-requests-per-hour cache and roughly ten minutes, none of
# which belong in a container that answers strategy questions in milliseconds.
#
#   python scripts/run_train.py          # produces the artefacts, on the host
#   docker build -t f1-tyre-degradation .
#   docker run --rm -p 8000:8000 f1-tyre-degradation
#
# The trained artefacts are baked into the image rather than mounted, so a
# running container is a complete, reproducible answer to "which model is
# this?". They are 5.4 MB. If they are absent the build fails with a message
# saying to train first, rather than producing an image that starts and then
# 503s for ever.

FROM python:3.12-slim AS base

# PYTHONDONTWRITEBYTECODE: the filesystem is read-only in normal operation.
# PYTHONUNBUFFERED: without it, logs sit in a pipe buffer and a crashed
# container appears to have said nothing at all.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, in their own layer. Application code changes on every
# commit; this layer changes when requirements-service.txt does, which is
# rarely, so the wheel downloads are not repeated on each build.
COPY requirements-service.txt ./
RUN pip install --requirement requirements-service.txt

# Application code and configuration.
COPY src/ ./src/
COPY scripts/run_service.py ./scripts/
COPY config/ ./config/

# Trained artefacts. COPY fails the build if these are missing, which is the
# intent; the RUN below turns that into an explanation.
COPY artifacts/models/ ./artifacts/models/
COPY data/processed/ ./data/processed/

RUN test -f artifacts/models/xgb_deg_rate.joblib \
      && test -f data/processed/stints_target.parquet \
      && test -f data/processed/laps_clean.parquet \
    || (echo "Trained artefacts missing. Run scripts/run_train.py first." \
        && exit 1)

# An unprivileged user. Nothing here writes to disk in normal operation: the
# model is read at start-up and every request is pure computation.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 service \
    && chown -R service:service /app
USER service

EXPOSE 8000

# /ready rather than /health: the model takes a couple of seconds to load, and
# an orchestrator should hold traffic back until it has. start-period covers
# that window so a slow start is not counted as a failure.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"

# exec form, so the process gets PID 1 and receives SIGTERM directly rather
# than through a shell that would swallow it.
CMD ["python", "scripts/run_service.py", "--host", "0.0.0.0", "--port", "8000"]
