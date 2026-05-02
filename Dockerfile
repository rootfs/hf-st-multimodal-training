FROM rocm/pytorch:latest

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HSA_ENABLE_SDMA=0 \
    GPU_MAX_HEAP_SIZE=100 \
    GPU_MAX_ALLOC_PERCENT=100 \
    MIOPEN_FIND_MODE=3 \
    NCCL_ALGO=Ring \
    NCCL_PROTO=Simple \
    NCCL_TIMEOUT=3600 \
    RCCL_TIMEOUT=3600

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /workspace/requirements.txt

COPY . /workspace/app
WORKDIR /workspace/app

CMD ["bash"]
