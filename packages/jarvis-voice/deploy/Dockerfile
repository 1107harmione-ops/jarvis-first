FROM python:3.11-slim

ARG PIPER_VERSION=2023.11
ARG TARGETARCH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS binary
RUN if [ -z "$TARGETARCH" ]; then \
        ARCH=amd64; \
    else \
        ARCH=${TARGETARCH}; \
    fi && \
    wget -q "https://github.com/rhasspy/piper/releases/download/v${PIPER_VERSION}/piper_linux_${ARCH}.tar.gz" -O /tmp/piper.tar.gz && \
    tar -xzf /tmp/piper.tar.gz -C /usr/local/bin/ --strip-components=1 && \
    rm /tmp/piper.tar.gz

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY jarvis_voice/ jarvis_voice/
COPY voices/ voices/

# Create model directories
RUN mkdir -p /app/models /app/voices

# Expose voice server port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "jarvis_voice.server:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"]
