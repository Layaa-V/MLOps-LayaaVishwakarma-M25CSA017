# Backend Dockerfile v5 — Two-Model A/B
# Pre-downloads both model weights safely so container starts instantly.

FROM python:3.10-slim AS model-downloader
WORKDIR /model-cache

RUN pip install --no-cache-dir --progress-bar off transformers torch huggingface_hub

# THE FIX: Use snapshot_download instead of from_pretrained.
# This fetches the files directly without executing them, preventing the ValueError
# and preventing Out-Of-Memory crashes during the build!
RUN python -c "from huggingface_hub import snapshot_download; \
print('Downloading raw CLIP...'); \
snapshot_download('openai/clip-vit-base-patch32'); \
print('Downloading FashionCLIP...'); \
snapshot_download('patrickjohncyh/fashion-clip'); \
print('Both models cached successfully.')"

# Stage 2: Production image
FROM python:3.10-slim AS app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy both models' cached weights
COPY --from=model-downloader /root/.cache/huggingface /root/.cache/huggingface

COPY app.py .
COPY embeddings_a.pt .
COPY embeddings_b.pt .
COPY metadata.json .
COPY eval_queries.json .
COPY images/ ./images/
COPY configs/ ./configs/

EXPOSE 8000
CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:8000", "app:app"]