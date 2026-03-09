FROM vllm/vllm-openai:latest

ENV SWARM_VLLM_PORT=8000 \
    SWARM_BASE_MODEL=Qwen/Qwen3.5-0.8B \
    SWARM_MAX_LORAS=64 \
    SWARM_MAX_CPU_LORAS=128 \
    SWARM_MAX_LORA_RANK=64 \
    SWARM_LORA_DIR=/loras \
    SWARM_MODEL_DIR=/models \
    SWARM_GPU_MEMORY_UTIL=0.90 \
    SWARM_MAX_MODEL_LEN=8192 \
    SWARM_GUIDED_BACKEND=xgrammar \
    HF_TOKEN=""

RUN mkdir -p /loras /models /schemas

COPY config/schemas/ /schemas/

EXPOSE ${SWARM_VLLM_PORT}

ENTRYPOINT ["sh", "-c", "\
    python3 -m vllm.entrypoints.openai.api_server \
    --model ${SWARM_BASE_MODEL} \
    --port ${SWARM_VLLM_PORT} \
    --enable-lora \
    --max-loras ${SWARM_MAX_LORAS} \
    --max-cpu-loras ${SWARM_MAX_CPU_LORAS} \
    --max-lora-rank ${SWARM_MAX_LORA_RANK} \
    --gpu-memory-utilization ${SWARM_GPU_MEMORY_UTIL} \
    --max-model-len ${SWARM_MAX_MODEL_LEN} \
    --download-dir ${SWARM_MODEL_DIR} \
    --trust-remote-code \
"]
