FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pyproject.toml .
COPY scripts/start-backend.sh ./scripts/start-backend.sh

RUN chmod +x ./scripts/start-backend.sh

ENV PYTHONPATH=/app/backend
ENV PORT=8000

EXPOSE 8000

CMD ["./scripts/start-backend.sh"]
