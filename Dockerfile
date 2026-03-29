FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download VADER lexicon at build time
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/usr/local/nltk_data')"

COPY config.py ./config.py
COPY backend/ ./backend/
COPY src/ ./src/
COPY frontend/ ./frontend/

RUN mkdir -p ./data

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
