# ===========================================================
# STAGE 1 — BUILDER
# ===========================================================
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build dependencies for psycopg2, faiss-cpu, pydantic, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependencies first to leverage cache
COPY requirements.txt .

# Prebuild wheels for all packages
RUN python -m pip install --upgrade pip wheel \
 && pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt


# ===========================================================
# STAGE 2 — RUNTIME
# ===========================================================
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_HOST=unix:///var/run/docker.sock

# Runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project source
COPY . .

# Install wheels built in STAGE 1
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir -r requirements.txt

# Add a non-root user
RUN useradd -m appuser
USER appuser

# Optional healthcheck (always OK)
HEALTHCHECK CMD python -c "import sys; sys.exit(0)"

CMD ["python", "run_rag_virtual_rename.py"]
