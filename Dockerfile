# JARVIS Backend — Root Dockerfile (for Render deployment)
# ────────────────────────────────────────────────────────────
# Build context: repo root (.)
# The backend/ directory contains all application code.
#
# For local builds:  docker build -t jarvis-backend -f Dockerfile .
# For local dev:     docker build -t jarvis-backend ./backend
# For Render:        auto-detected via render.yaml
#
# Code is placed at /app/backend/ to preserve the "backend." import prefix.

# ── Builder Stage ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Production Stage ──────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN addgroup --system jarvis && adduser --system --ingroup jarvis jarvis

# Copy Python packages from builder
COPY --from=builder /root/.local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin

# ── Copy application code ─────────────────────────────────────
# Copy backend/ contents into /app/backend/ so that imports like
# "backend.main:app" and "from backend.config.settings import settings" resolve.
COPY --chown=jarvis:jarvis backend/ /app/backend/

# Copy start script to /app/ root
COPY --chown=jarvis:jarvis backend/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create log directory
RUN mkdir -p /app/logs && chown -R jarvis:jarvis /app/logs

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; port=os.environ.get('PORT','8000'); urllib.request.urlopen(f'http://localhost:{port}/api/admin/health')" || exit 1

EXPOSE 8000

USER jarvis

# Start via script — handles Render's dynamic $PORT
CMD ["/app/start.sh"]
