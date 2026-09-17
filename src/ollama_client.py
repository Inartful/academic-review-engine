"""Клиент локальной Ollama для генерации структурированной рецензии.

Двойная страховка качества:
  1) `format=<JSON schema>` — грамматический constrained decoding: модель физически
     не может вернуть невалидный JSON или другое имя поля;
  2) "якорь свежести" в конце пользовательского промпта — приказ отвечать по-русски,
     продублированный в самом конце (иначе 7B-модель уходит в английский на статьях
     с английским abstract).
"""

from __future__ import annotations

import json

import requests

from .config import MODEL_NAME, NUM_CTX, OLLAMA_URL, REQUEST_TIMEOUT, TEMPERATURE
from .schemas import REVIEW_SCHEMA_RU

SYSTEM_PROMPT = (
    "Ты — суровый научный рецензент уровня PhD. "
    "Выдавай анализ строго на русском языке."
)
LANGUAGE_ANCHOR = (
    "[ВНИМАНИЕ: ДАЖЕ ЕСЛИ В ТЕКСТЕ ЕСТЬ АНГЛИЙСКИЙ АБСТРАКТ, "
    "ТЫ ОБЯЗАН ЗАПОЛНИТЬ JSON ИСКЛЮЧИТЕЛЬНО НА РУССКОМ ЯЗЫКЕ!]"
)
RAG_CONTEXT_TEMPLATE = """
ИНФОРМАЦИЯ ИЗ АРХИВА (для проверки новизны):
Найдены следующие похожие работы, опубликованные ранее:
{context}
"""


def build_user_prompt(article_text: str, rag_context: str = "") -> str:
    """Собирает пользовательский промпт: опциональный RAG-контекст + статья + якорь языка."""
    context_block = RAG_CONTEXT_TEMPLATE.format(context=rag_context) if rag_context else ""
    return (
        "Проанализируй текст и составь рецензию. "
        "Оцени методологию и новизну от 1 до 10.\n"
        f"{context_block}\n"
        f"ТЕКСТ СТАТЬИ:\n{article_text}\n\n"
        f"{LANGUAGE_ANCHOR}"
    )


def chat(payload: dict, url: str = OLLAMA_URL, timeout: int = REQUEST_TIMEOUT) -> dict:
    """Низкоуровневый вызов Ollama `/api/chat` с проверкой HTTP-статуса."""
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def generate_review(
    article_text: str,
    rag_context: str = "",
    model_name: str = MODEL_NAME,
    url: str = OLLAMA_URL,
) -> dict:
    """Возвращает рецензию локальной модели как dict (RU-ключи, см. src/schemas.py)."""
    payload = {
        "model": model_name,
        "format": REVIEW_SCHEMA_RU,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(article_text, rag_context)},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    content = chat(payload, url=url)["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Ollama вернула невалидный JSON: {content[:200]!r}") from error
