FROM python:3.12-slim

# ---- System deps (minimal + stable) ----
# gcc/libpq-dev kept only if you ever need to build psycopg2 from source;
# psycopg2-binary normally doesn't need them, but harmless to keep for now.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Copy requirements first (layer caching) ----
COPY requirements.txt .

# ---- Install Python deps ----
# --extra-index-url for CPU-only torch is declared inside requirements.txt itself,
# so a plain pip install picks it up.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy project ----
# Make sure a .dockerignore excludes: .git, __pycache__, *.pyc, .venv, data/,
# any local .env file, pgdata/, ollama_models/ — none of these belong in the image.
COPY . .

# -------------------------------
# Default command
# Overridden by docker-compose's `command:` for the app service —
# this only applies if the image is run standalone via `docker run`.
# -------------------------------
CMD ["python", "./scripts/test_rag_generation.py"]