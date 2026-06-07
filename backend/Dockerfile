# JARVIS Backend — Dockerfile
# Multi-stage build for optimized production image

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

# Copy Python packages from builder (install for system Python)
COPY --from=builder /root/.local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin

# Copy application code
COPY --chown=jarvis:jarvis . .

# Create log directory
RUN mkdir -p /app/logs && chown -R jarvis:jarvis /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/api/admin/health').read())" || exit 1

EXPOSE 8000

USER jarvis

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
