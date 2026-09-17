# ==========================================================
# Образ Streamlit-приложения (UI + RAG).
# Инференс вынесен в отдельный сервис Ollama (см. docker-compose.yml).
# ==========================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Зависимости отдельным слоем — кэшируются при изменении кода
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY src ./src
COPY scripts ./scripts
COPY modelfiles ./modelfiles
COPY docs ./docs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", "--server.port=8501"]
