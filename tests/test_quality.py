"""Тесты метрики языковой консистентности (src/quality.py)."""

from __future__ import annotations

from src.quality import check_language_consistency, count_latin_words, extract_text_fields


def test_pure_russian_review_passes() -> None:
    review = {
        "резюме": "Статья посвящена анализу устойчивости конструкций.",
        "сильные_стороны": ["Корректная методика расчёта"],
        "слабые_стороны": ["Малая выборка испытаний"],
        "оценка_методологии": 6,
        "оценка_новизны": 5,
        "вердикт": "Отправить на доработку",
    }
    is_russian, latin_words = check_language_consistency(review)
    assert is_russian is True
    assert latin_words == 0


def test_english_review_fails() -> None:
    review = {
        "резюме": "The article is devoted to the stability analysis of the structures under load.",
        "сильные_стороны": ["Correct calculation methodology"],
        "слабые_стороны": ["Small test sample"],
    }
    is_russian, latin_words = check_language_consistency(review)
    assert is_russian is False
    assert latin_words > 5


def test_abbreviations_are_tolerated() -> None:
    """Аббревиатуры (JSON, API, DOI, ANSYS) не считаются языковым сдвигом."""
    review = {
        "резюме": "Сравнение методов расчёта в ANSYS Maxwell и по методике ГОСТ.",
        "сильные_стороны": ["Валидация JSON-схемы через API"],
        "слабые_стороны": ["Нет DOI у части ссылок"],
    }
    is_russian, latin_words = check_language_consistency(review)
    assert is_russian is True
    assert latin_words == 5


def test_english_keys_are_supported() -> None:
    review = {"summary": "Резюме на русском", "strengths": ["Плюс"], "weaknesses": ["Минус"]}
    assert count_latin_words(review) == 0
    assert "Резюме на русском" in extract_text_fields(review)


def test_custom_threshold_is_respected() -> None:
    review = {"резюме": "Расчёт ANSYS Maxwell корректен."}
    assert count_latin_words(review) == 2
    assert check_language_consistency(review, max_english_words=1)[0] is False
    assert check_language_consistency(review, max_english_words=2)[0] is True
