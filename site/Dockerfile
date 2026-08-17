FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY templates ./templates
COPY data/page.example.json ./data/page.example.json

ENV SITE_HOST=0.0.0.0
ENV SITE_PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host ${SITE_HOST:-0.0.0.0} --port ${SITE_PORT:-8080}"]
