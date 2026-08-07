# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder — install Python dependencies into an isolated prefix.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build-time system deps (removed automatically by not carrying stage).
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a relocatable prefix for a slim runtime copy.
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal image running as a non-root user.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Copy only the installed packages from the builder stage.
COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY requirements.txt ./

# Run as the unprivileged 'nobody' user to satisfy enterprise Kubernetes
# security contexts and SAP AI Core cloud policies.
USER nobody

EXPOSE 8080

# Bind to 0.0.0.0 and honor the PORT env var (SAP AI Core networking).
CMD ["sh", "-c", "uvicorn src.deployment.app:app --host 0.0.0.0 --port ${PORT}"]
