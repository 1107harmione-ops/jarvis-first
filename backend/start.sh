#!/bin/bash
# Jarvis Voice Assistant — Start Script
set -e

cd "$(dirname "$0")"

# Ensure runtime directories exist (already created in Docker build with chown,
# but this handles case where a volume mount might clear them)
mkdir -p data models

# Remove --reload in production for better performance and stability
RELOAD_FLAG=""
if [ "${ENVIRONMENT}" = "development" ] || [ "${DEBUG}" = "true" ]; then
    RELOAD_FLAG="--reload"
fi

# Run the server
exec uvicorn app.main:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8000} \
    ${RELOAD_FLAG}
