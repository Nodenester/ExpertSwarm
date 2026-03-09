FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git curl build-essential && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip install --no-cache-dir --break-system-packages git+https://github.com/huggingface/transformers.git

COPY . .

ENV CODEGATE_URL=http://localhost:9212
ENV CODEGATE_API_KEY=proxy-local-key
ENV TEACHER_MODEL=claude-sonnet-4-20250514
ENV SWARM_RUNNER_URL=http://localhost:8100
ENV BASE_MODEL=Qwen/Qwen3.5-0.8B
ENV LORA_OUTPUT_DIR=/checkpoints
ENV DATASET_DIR=/datasets
ENV CUDA_VISIBLE_DEVICES=0

ENTRYPOINT ["python"]
