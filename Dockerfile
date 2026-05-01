# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PIP_DEFAULT_TIMEOUT=180

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 180 --upgrade pip && \
    pip install --retries 10 --timeout 180 --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 && \
    pip install --retries 10 --timeout 180 --prefer-binary -r requirements.txt

COPY app.py ./app.py
COPY artifacts ./artifacts

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
