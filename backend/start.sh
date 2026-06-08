#!/bin/bash
# Jarvis Voice Assistant — Start Script
set -e

cd "$(dirname "$0")"

# Ensure directories exist
mkdir -p data models

# Run the server
uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --reload
