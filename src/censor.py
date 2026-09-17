"""Детерминированная логика принятия решения ("алгоритмический цензор").

Проблема: LLM выставляет числовые оценки и текстовый вердикт независимо, поэтому
вердикт часто не согласован с оценками (см. `reports/evaluation_results.jsonl`,
где судья-LLM снижает балл именно за нарушение логики). Решение: финальное решение
принимает простая воспроизводимая функция от двух оценок, а поле `вердикт`
от модели используется только как её "мнение", а не как итог.
"""

from __future__ import annotations

from .config import CENSOR_ACCEPT_METHODOLOGY, CENSOR_ACCEPT_NOVELTY, CENSOR_REJECT_BELOW

VERDICT_ACCEPT = "Принять"
VERDICT_REVISE = "Отправить на доработку"
VERDICT_REJECT = "Отклонить"

CSS_BY_VERDICT: dict[str, str] = {
    VERDICT_ACCEPT: "verdict-accept",
    VERDICT_REVISE: "verdict-revise",
    VERDICT_REJECT: "verdict-reject",
}


def decide(methodology_score: int, novelty_score: int) -> tuple[str, str]:
    """Возвращает пару (вердикт, css-класс для UI).

    Правила:
      * методология < 4 или новизна < 4     -> Отклонить
      * методология >= 7 и новизна >= 6     -> Принять
      * иначе                               -> Отправить на доработку
    """
    if methodology_score < CENSOR_REJECT_BELOW or novelty_score < CENSOR_REJECT_BELOW:
        verdict = VERDICT_REJECT
    elif (
        methodology_score >= CENSOR_ACCEPT_METHODOLOGY
        and novelty_score >= CENSOR_ACCEPT_NOVELTY
    ):
        verdict = VERDICT_ACCEPT
    else:
        verdict = VERDICT_REVISE
    return verdict, CSS_BY_VERDICT[verdict]
