FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ffmpeg build-essential \
    python3.11 python3.11-dev python3.11-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CPU-only) and torchvision
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install openmim, mmcv, and mmaction2
RUN pip install openmim && \
    mim install mmcv==2.0.1 && \
    mim install mmaction2==1.1.0

COPY . .

# Install project requirements
RUN pip install -r requirements.txt

# Install custom NMS package
RUN pip install ./tad/models/utils/post_processing/nms

# Default command: train on Charades dataset
CMD ["python", "./tools/train.py", "./configs/charades_i3d_rgb.py"]
