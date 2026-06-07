# JARVIS Backend — Dockerfile
# Multi-stage build for optimized production image
#
# Compatible with:
#   - Render (Docker runtime via render.yaml Blueprint)
#   - docker build -t jarvis-backend ./backend
#   - docker compose up (via docker-compose.yml)
#
# IMPORTANT: The build context MUST be the backend/ directory.
# Code is placed at /app/backend/ to preserve the "backend." import prefix.

# ── Builder Stage ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
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
# Copy entire backend/ contents into /app/backend/ so that
# "backend.main:app" and "from backend.config.settings import settings"
# resolve correctly (Python finds the "backend" package at /app/backend/).
COPY --chown=jarvis:jarvis . /app/backend/

# Copy start script to /app/ root (alongside the backend/ package)
COPY --chown=jarvis:jarvis start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create log directory
RUN mkdir -p /app/logs && chown -R jarvis:jarvis /app/logs

# ── Health check ──────────────────────────────────────────────
# Render uses this to verify the service is running.
# $PORT is resolved at runtime via shell.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; port=os.environ.get('PORT','8000'); urllib.request.urlopen(f'http://localhost:{port}/api/admin/health')" || exit 1

EXPOSE 8000

USER jarvis

# Start via script — handles Render's dynamic $PORT
CMD ["/app/start.sh"]
