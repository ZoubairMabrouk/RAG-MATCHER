# =========
# Stage 1: builder (compile wheels so the final image stays slim)
# =========
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps to build wheels for packages like psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only copy requirement files first to leverage Docker layer caching
COPY requirements.txt ./

# Build wheels for all deps
RUN python -m pip install --upgrade pip wheel \
 && pip wheel --wheel-dir=/wheels -r requirements.txt

# =========
# Stage 2: runtime
# =========
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Allows Testcontainers to find the Docker daemon via the default socket
    DOCKER_HOST=unix:///var/run/docker.sock

# Runtime libs for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy app source
COPY . /app

# Copy prebuilt wheels and install
COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip \
 && pip install --no-index --find-links=/wheels -r requirements.txt \
 && pip install pytest pytest-cov

# Optional: drop privileges
RUN useradd -m appuser
USER appuser

# Default command runs tests; override as needed (e.g., `docker run ... pytest -k yourtest`)
CMD ["pytest", "-v", "--maxfail=1", "--disable-warnings"]
