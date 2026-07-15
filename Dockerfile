# LG Tutor API
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir poetry==1.8.*

WORKDIR /app

# Install deps first so they cache independently of code changes
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

COPY core/ ./core/

# Checkpointer DB lives on a volume mounted at /data (see docker-compose.yml)
ENV THREADS_DB_PATH=/data/threads.db
RUN mkdir -p /data

# api.py uses flat imports (from graph import ...), so run from core/
WORKDIR /app/core

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
