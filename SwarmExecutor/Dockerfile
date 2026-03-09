FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY src/swarms/*.yaml ./swarms/

ENV PYTHONUNBUFFERED=1
ENV SWARM_EXECUTOR_PORT=8500
ENV SWARM_CONFIG_DIR=/swarms

EXPOSE 8500

CMD ["python", "-m", "src.main"]
