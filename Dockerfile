# Base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY Q1/ ./Q1/
COPY Q2/ ./Q2/

# Create directories for data and weights
RUN mkdir -p data weights

CMD ["bash"]
