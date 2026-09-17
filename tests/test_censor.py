"""Тесты детерминированного "алгоритмического цензора" (src/censor.py)."""

from __future__ import annotations

import pytest

from src.censor import CSS_BY_VERDICT, decide


@pytest.mark.parametrize(
    ("methodology", "novelty", "expected"),
    [
        # Явный отказ: любая из оценок ниже порога
        (3, 9, "Отклонить"),
        (9, 3, "Отклонить"),
        (1, 1, "Отклонить"),
        # Граница: ровно 4 — уже не отказ
        (4, 4, "Отправить на доработку"),
        # Принятие требует методологии >= 7 И новизны >= 6
        (7, 6, "Принять"),
        (10, 10, "Принять"),
        (7, 5, "Отправить на доработку"),
        (6, 6, "Отправить на доработку"),
        # Середина
        (5, 5, "Отправить на доработку"),
    ],
)
def test_decide_verdicts(methodology: int, novelty: int, expected: str) -> None:
    verdict, _ = decide(methodology, novelty)
    assert verdict == expected


def test_decide_returns_matching_css_class() -> None:
    for methodology, novelty in [(1, 1), (5, 5), (9, 9)]:
        verdict, css_class = decide(methodology, novelty)
        assert css_class == CSS_BY_VERDICT[verdict]
