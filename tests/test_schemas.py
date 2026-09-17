"""Тесты схем и маппинга ключей RU/EN (src/schemas.py)."""

from __future__ import annotations

from src.schemas import (
    EN_TO_RU,
    REVIEW_SCHEMA_EN,
    REVIEW_SCHEMA_RU,
    RU_TO_EN,
    VERDICTS_RU,
    to_english_keys,
    to_russian_keys,
)


def test_required_fields_are_declared_in_properties() -> None:
    """Классический баг structured output: required содержит поле без описания."""
    for schema in (REVIEW_SCHEMA_RU, REVIEW_SCHEMA_EN):
        assert set(schema["required"]) <= set(schema["properties"])


def test_verdict_enum_is_consistent() -> None:
    assert REVIEW_SCHEMA_RU["properties"]["вердикт"]["enum"] == list(VERDICTS_RU)
    assert REVIEW_SCHEMA_EN["properties"]["final_verdict"]["enum"] == list(VERDICTS_RU)


def test_key_mapping_round_trip() -> None:
    review_ru = {
        "резюме": "Резюме",
        "сильные_стороны": ["a"],
        "слабые_стороны": ["b"],
        "оценка_методологии": 8,
        "оценка_новизны": 7,
        "вердикт": "Принять",
    }
    review_en = to_english_keys(review_ru)
    assert set(review_en) == {"summary", "strengths", "weaknesses",
                              "methodology_score", "novelty_score", "final_verdict"}
    assert to_russian_keys(review_en) == review_ru


def test_mapping_is_bijective() -> None:
    assert len(RU_TO_EN) == len(EN_TO_RU)
    assert all(EN_TO_RU[en] == ru for ru, en in RU_TO_EN.items())


def test_unmapped_keys_are_preserved() -> None:
    """Поля, которых нет в маппинге, не должны теряться (например, от LLM-судьи)."""
    assert to_english_keys({"резюме": "x", "extra": 1}) == {"summary": "x", "extra": 1}
