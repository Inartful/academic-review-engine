"""Централизованная конфигурация пайплайна авторецензирования.

Все пути считаются от корня репозитория, поэтому один и тот же код работает
из `app.py`, из `scripts/*` и из ноутбуков в `notebooks/`.
Значения, помеченные как env-driven, переопределяются переменными окружения
(см. `.env.example`), что позволяет запускать приложение в docker-compose.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Локальный инференс (Ollama)
# ==========================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-reviewer:latest")
TEMPERATURE = 0.1
NUM_CTX = 8192
REQUEST_TIMEOUT = 900  # секунд: 7B Q4 на CPU может думать долго

# ==========================================================
# RAG-контур проверки научной новизны
# ==========================================================
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", str(REPO_ROOT / "chroma_db")))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
RAG_TOP_K = 3
RAG_MAX_DISTANCE = 0.6  # L2: всё, что дальше — считаем "не похоже"
RAG_QUERY_CHARS = 1500  # сколько символов аннотации отправляем в поиск
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_SNIPPET_CHARS = 300  # сколько символов найденного фрагмента подмешиваем в промпт

# ==========================================================
# Данные
# ==========================================================
SYNTHETIC_DATASET = Path(
    os.getenv("SYNTHETIC_DATASET", str(REPO_ROOT / "synthetic_reviews_dataset_v2.jsonl"))
)
RAW_ARTICLES_DIR = Path(os.getenv("RAW_ARTICLES_DIR", str(REPO_ROOT / "dataset" / "data_3")))

# ==========================================================
# Артефакты экспериментов
# ==========================================================
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "docs" / "figures" / "metrics"
LORA_OUTPUT_DIR = REPO_ROOT / "lora_reviewer_final"
GGUF_OUTPUT_DIR = REPO_ROOT / "qwen-reviewer"

# ==========================================================
# Детерминированный "алгоритмический цензор"
# ==========================================================
CENSOR_REJECT_BELOW = 4  # методология < 4 ИЛИ новизна < 4 -> Отклонить
CENSOR_ACCEPT_METHODOLOGY = 7  # методология >= 7 И новизна >= 6 -> Принять
CENSOR_ACCEPT_NOVELTY = 6
