# JARVIS Backend — Dockerfile for Render & Local Deployment
# ────────────────────────────────────────────────────────────
# Build context: repo root (.)
# The repo contains the complete backend application at root level.
#
# This image:
#  - Places code at /app/ so "app.main:app" resolves directly
#  - Creates writable data/ and models/ directories at build time
#  - Runs as a dedicated non-root user (jarvis) for security
#  - Supports Render's dynamic $PORT via start.sh

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

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system jarvis && adduser --system --ingroup jarvis jarvis

# Copy Python packages from builder
COPY --from=builder /root/.local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin

# ── Copy application code ─────────────────────────────────────
# All files at repo root are copied directly to /app/
# so that "app.main:app" and "from app.core.config import settings" resolve.
COPY --chown=jarvis:jarvis . .

# Make start script executable
RUN chmod +x start.sh

# Create writable runtime directories with proper ownership.
# These are needed by start.sh (mkdir -p data models) which runs as jarvis user.
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R jarvis:jarvis /app/data /app/models /app/logs

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; port=os.environ.get('PORT','8000'); urllib.request.urlopen(f'http://localhost:{port}/api/admin/health')" || exit 1

EXPOSE 8000

USER jarvis

# Start via script — handles Render's dynamic $PORT, ensures data dirs
CMD ["/app/start.sh"]
