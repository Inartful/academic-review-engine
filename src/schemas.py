"""JSON-схемы structured output и маппинг русских/английских ключей.

Контекст решения: синтетический датасет (ноутбук 01) генерировался моделью-учителем
с англоязычными ключами (`summary`, `methodology_score`, ...), а рантайм-промпты
и UI работают с русскими (`резюме`, `оценка_методологии`, ...). Схема Ollama
(`format=...`) детерминированно задаёт имена полей на этапе декодирования,
поэтому обе версии держим рядом и конвертируем явно — чтобы не расползалось.
"""

from __future__ import annotations

VERDICTS_RU: tuple[str, ...] = ("Принять", "Отправить на доработку", "Отклонить")

# --- Схема для локальной модели (Ollama format=<JSON schema>) ---
REVIEW_SCHEMA_RU: dict = {
    "type": "object",
    "properties": {
        "резюме": {
            "type": "string",
            "description": "НА РУССКОМ ЯЗЫКЕ. Краткое резюме статьи (2-3 предложения).",
        },
        "сильные_стороны": {
            "type": "array",
            "items": {"type": "string", "description": "НА РУССКОМ ЯЗЫКЕ."},
        },
        "слабые_стороны": {
            "type": "array",
            "items": {"type": "string", "description": "НА РУССКОМ ЯЗЫКЕ."},
        },
        "оценка_методологии": {"type": "integer", "description": "Целое число от 1 до 10."},
        "оценка_новизны": {"type": "integer", "description": "Целое число от 1 до 10."},
        "вердикт": {"type": "string", "enum": list(VERDICTS_RU)},
    },
    "required": [
        "резюме",
        "сильные_стороны",
        "слабые_стороны",
        "оценка_методологии",
        "оценка_новизны",
        "вердикт",
    ],
}

# --- Схема модели-учителя (DeepSeek-V3) при генерации синтетического датасета ---
REVIEW_SCHEMA_EN: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "methodology_score": {"type": "integer"},
        "novelty_score": {"type": "integer"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "final_verdict": {"type": "string", "enum": list(VERDICTS_RU)},
    },
    "required": [
        "summary",
        "methodology_score",
        "novelty_score",
        "strengths",
        "weaknesses",
        "final_verdict",
    ],
}

RU_TO_EN: dict[str, str] = {
    "резюме": "summary",
    "сильные_стороны": "strengths",
    "слабые_стороны": "weaknesses",
    "оценка_методологии": "methodology_score",
    "оценка_новизны": "novelty_score",
    "вердикт": "final_verdict",
}
EN_TO_RU: dict[str, str] = {en: ru for ru, en in RU_TO_EN.items()}


def _remap(review: dict, mapping: dict[str, str]) -> dict:
    """Переименовывает ключи, не теряя поля, которых нет в маппинге."""
    return {(mapping.get(key, key)): value for key, value in review.items()}


def to_english_keys(review: dict) -> dict:
    """RU-ключи -> EN-ключи (для сравнения с датасетом / судьёй)."""
    return _remap(review, RU_TO_EN)


def to_russian_keys(review: dict) -> dict:
    """EN-ключи -> RU-ключи."""
    return _remap(review, EN_TO_RU)
