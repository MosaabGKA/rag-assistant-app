# Streamlit frontend image — sits in front of the FastAPI RAG service.
# The API base URL is injected at runtime (default: same host, port 8000).
FROM python:3.12-slim

WORKDIR /workspace

COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/app.py ./app.py
COPY frontend/api_client.py ./api_client.py
COPY frontend/cli.py ./cli.py

ENV API_BASE_URL=http://127.0.0.1:8000
ENV RAG_API=http://127.0.0.1:8000
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
