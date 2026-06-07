#!/bin/bash
# JARVIS Backend — Render/Docker Entrypoint
# ────────────────────────────────────────────────────────────
# Handles dynamic port binding and startup validation.
#
# Render injects $PORT as an environment variable.
# If $PORT is set, we override the config default.
# ────────────────────────────────────────────────────────────

set -e

echo "━━━ JARVIS Backend Startup ━━━"
echo "Environment: ${ENVIRONMENT:-development}"
echo "Python:      $(python --version 2>&1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Port Configuration ─────────────────────────────────────
# Render sets $PORT dynamically. Use it if available, otherwise
# fall back to the settings default (8000).
if [ -n "$PORT" ]; then
    echo "[start] Using Render-assigned PORT=$PORT"
    export PORT="$PORT"
else
    echo "[start] Using default port 8000"
    export PORT=8000
fi

# ── Environment Validation ─────────────────────────────────
if [ -z "$MONGODB_URI" ]; then
    echo ""
    echo "⚠  WARNING: MONGODB_URI is not set!"
    echo "   The backend requires MongoDB Atlas to function."
    echo "   Set it in Render Dashboard → Environment Variables."
    echo ""
fi

if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "change-me-to-a-random-secret" ]; then
    echo "⚠  WARNING: SECRET_KEY is not set or is using the default value."
    echo "   Generate a strong random key and set it in Render Dashboard."
    echo ""
fi

if [ -z "$JWT_SECRET_KEY" ] || [ "$JWT_SECRET_KEY" = "change-me-to-a-random-jwt-secret" ]; then
    echo "⚠  WARNING: JWT_SECRET_KEY is not set or is using the default value."
    echo "   Generate a strong random key and set it in Render Dashboard."
    echo ""
fi

if [ -z "$DEEPSEEK_API_KEY" ] && [ -z "$CODEX_API_KEY" ] && [ -z "$MINIMAX_API_KEY" ]; then
    echo "⚠  WARNING: No LLM API keys are configured!"
    echo "   At least one provider key is needed (DeepSeek recommended)."
    echo ""
fi

echo "━━━ Starting Uvicorn ━━━"
echo "Listening on: 0.0.0.0:$PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Start Server ───────────────────────────────────────────
exec uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "${WORKERS:-2}" \
    --log-level "${LOG_LEVEL:-info}" \
    --no-access-log  # We have our own request logging middleware
