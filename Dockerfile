# Production Dockerfile for Jeff's VideoScan Cloud Run Service
FROM python:3.13-slim-bookworm

# Install FFmpeg and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY metadata_catalog.md /app/metadata_catalog.md
COPY spec.md /app/spec.md

# Create non-root user and temp directories
RUN useradd -u 1000 -m appuser && \
    mkdir -p /tmp/videoprocessing && \
    chown -R appuser:appuser /app /tmp/videoprocessing

USER appuser

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TEMP_DIR=/tmp/videoprocessing

EXPOSE 8080

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
