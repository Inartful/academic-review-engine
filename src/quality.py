"""Проверки качества ответов модели.

Главный эмпирический сбой, который поймали в этой работе, — *языковой сдвиг*:
если в статье есть английский abstract, 7B-модель после LoRA начинала генерировать
поля JSON по-английски. Поэтому в промпт добавлен "якорь свежести" на русском,
а здесь — метрика, которой этот сбой измеряется.
"""

from __future__ import annotations

import re

LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
DEFAULT_MAX_ENGLISH_WORDS = 5  # допускаем аббревиатуры вида API, JSON, IT, DOI
_TEXT_KEYS = ("резюме", "summary")
_LIST_KEYS = ("сильные_стороны", "слабые_стороны", "strengths", "weaknesses")


def extract_text_fields(review: dict) -> str:
    """Склеивает все текстовые поля рецензии (RU- или EN-ключи) в одну строку."""
    chunks: list[str] = []
    for key in _TEXT_KEYS:
        value = review.get(key)
        if isinstance(value, str):
            chunks.append(value)
    for key in _LIST_KEYS:
        value = review.get(key)
        if isinstance(value, list):
            chunks.extend(str(item) for item in value)
    return " ".join(chunks)


def count_latin_words(review: dict) -> int:
    """Сколько латинских слов встретилось в текстовых полях рецензии."""
    return len(LATIN_WORD_RE.findall(extract_text_fields(review)))


def check_language_consistency(
    review: dict, max_english_words: int = DEFAULT_MAX_ENGLISH_WORDS
) -> tuple[bool, int]:
    """Возвращает (ответ_на_русском, количество_латинских_слов)."""
    latin_words = count_latin_words(review)
    return latin_words <= max_english_words, latin_words
