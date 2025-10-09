FROM python:3.11-slim

# System deps you might need later (BLAS, git, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app

# Healthcheck-friendly start
EXPOSE 8102
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8102"]

