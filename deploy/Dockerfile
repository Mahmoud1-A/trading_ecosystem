FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -e .

ENV DATA_DIR=/app/data
ENV LOG_LEVEL=INFO
EXPOSE 8080

# Default: monitor. Paper/exec override command in compose/systemd.
CMD ["te-monitor", "--host", "0.0.0.0", "--port", "8080"]
