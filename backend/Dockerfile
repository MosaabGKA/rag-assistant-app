# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /workspace

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app ./backend/app
COPY backend/data ./backend/data

ENV PYTHONPATH=/workspace/backend
WORKDIR /workspace/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
