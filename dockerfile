FROM python:3.12-slim

# ---- System deps (minimal + stable) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Copy requirements ----
COPY requirements.txt .

# ---- Install Python deps ----
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy project ----
COPY . .
# -------------------------------
# Default command
# (à adapter selon ton projet: main.py, uvicorn, streamlit, flask…)
# -------------------------------
CMD ["python", "./scripts/test_rag_generation.py"]
