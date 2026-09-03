FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY backend/ ./backend/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pyproject.toml .
COPY scripts/start-backend.sh ./scripts/start-backend.sh
COPY scripts/download_serving_artifacts.sh ./scripts/download_serving_artifacts.sh

RUN chmod +x ./scripts/start-backend.sh ./scripts/download_serving_artifacts.sh

ENV PYTHONPATH=/app/backend
ENV PORT=8000
ENV INFERENCE_DEVICE=cpu
ENV ENABLE_RANKING=false
ENV WARMUP_ON_STARTUP=false

EXPOSE 8000

CMD ["./scripts/start-backend.sh"]
