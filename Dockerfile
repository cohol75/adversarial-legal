FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn openai httpx

COPY main.py graph.py prompts.py ./
COPY static/ ./static/

EXPOSE 8899

CMD ["python", "main.py"]
