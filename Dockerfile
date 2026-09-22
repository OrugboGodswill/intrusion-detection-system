FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IDS_API_URL="http://127.0.0.1:5000"

# Install curl for container healthchecks and bash for entrypoint execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8501 5000

CMD ["./start.sh"]
