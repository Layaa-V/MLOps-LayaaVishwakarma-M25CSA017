FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 3. Setup workspace
WORKDIR /workspace

# 4. Install essential system tools for debugging and downloading
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 9. Default command to run the training script
CMD ["python", "train.py"]
