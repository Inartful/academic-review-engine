"""Streamlit-приложение: DSS авторецензирования научных статей.

Пайплайн:
    PDF/TXT -> извлечение текста -> (опц.) поиск похожих работ в архиве (RAG)
    -> генерация структурированной рецензии локальной Qwen2.5-7B-LoRA (Ollama)
    -> детерминированный "алгоритмический цензор" -> вердикт и отчёт в UI.

Запуск:
    streamlit run app.py

Требуется запущенная локальная Ollama с моделью `qwen-reviewer:latest`
(как собрать — см. README.md).
"""

from __future__ import annotations

import streamlit as st

from src.censor import decide
from src.config import MODEL_NAME
from src.documents import extract_text_from_file
from src.ollama_client import generate_review
from src.rag import build_rag_context, load_vector_db, search_similar_articles

CUSTOM_CSS = """
<style>
    .report-box {
        background-color: #f8f9fa;
        border-left: 5px solid #4C72B0;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #333;
    }
    .verdict-accept { background-color: #d4edda; color: #155724; padding: 15px;
        border-radius: 5px; font-size: 18px; text-align: center; font-weight: bold; }
    .verdict-revise { background-color: #fff3cd; color: #856404; padding: 15px;
        border-radius: 5px; font-size: 18px; text-align: center; font-weight: bold; }
    .verdict-reject { background-color: #f8d7da; color: #721c24; padding: 15px;
        border-radius: 5px; font-size: 18px; text-align: center; font-weight: bold; }
</style>
"""

st.set_page_config(
    page_title="DSS Reviewer | AI Ассистент Редактора",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_vector_db():
    """Загружает и кэширует векторный индекс архива (RAG)."""
    return load_vector_db()


def render_sidebar() -> bool:
    """Боковая панель с настройками. Возвращает флаг использования RAG."""
    with st.sidebar:
        st.title("⚙️ Настройки DSS")
        st.info(
            f"**Режим работы:** On-Premise\n\n"
            f"**Модель:** {MODEL_NAME} (Qwen2.5-7B + LoRA)\n\n"
            "**RAG База:** ChromaDB + multilingual-e5-large"
        )
        use_rag = st.checkbox(
            "Включить проверку новизны (RAG)",
            value=True,
            help="Искать похожие статьи в локальном архиве. Индекс собирается "
            "скриптом `scripts/build_rag_index.py`.",
        )
        st.divider()
        st.markdown(
            """
            **Как работает Алгоритмический Цензор:**
            - Методология или новизна < 4 ➔ **Отклонить**
            - Методология ≥ 7 и новизна ≥ 6 ➔ **Принять**
            - В остальных случаях ➔ **Доработать**

            Числа оценок берутся у LLM, а итоговый вердикт считает
            детерминированная функция — это защита от рассогласованных
            вердиктов модели.
            """
        )
        st.divider()
        st.caption("Дипломный проект: DSS для предварительной экспертизы рукописей.")
    return use_rag


def render_result(review: dict, rag_results: list, use_rag: bool) -> None:
    """Отрисовывает вердикт, оценки, сильные/слабые стороны и справку по новизне."""
    methodology_score = review.get("оценка_методологии", 0)
    novelty_score = review.get("оценка_новизны", 0)
    final_decision, css_class = decide(methodology_score, novelty_score)

    st.divider()
    col_verdict, col_methodology, col_novelty = st.columns([2, 1, 1])
    with col_verdict:
        st.markdown("### Рекомендация системы:")
        st.markdown(
            f"<div class='{css_class}'>{final_decision.upper()}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"*(Изначальный вердикт модели «{review.get('вердикт', '')}» "
            "переопределён Алгоритмическим Цензором)*"
        )
    with col_methodology:
        st.metric(label="Методология (из 10)", value=methodology_score)
    with col_novelty:
        st.metric(label="Научная новизна (из 10)", value=novelty_score)

    st.markdown("### 📝 Резюме статьи")
    st.markdown(
        f"<div class='report-box'>{review.get('резюме', 'Нет данных')}</div>",
        unsafe_allow_html=True,
    )

    col_strengths, col_weaknesses = st.columns(2)
    with col_strengths:
        st.markdown("### ✅ Сильные стороны")
        for item in review.get("сильные_стороны", []):
            st.success(item)
    with col_weaknesses:
        st.markdown("### ❌ Слабые стороны (Критика)")
        for item in review.get("слабые_стороны", []):
            st.error(item)

    if not use_rag:
        return
    if rag_results:
        with st.expander("📚 Справка о научной новизне (похожие статьи из архива)"):
            st.markdown("Система опиралась на следующие фрагменты локального архива:")
            for index, (document, score) in enumerate(rag_results, start=1):
                st.info(
                    f"**Совпадение {index} (L2 Distance: {score:.2f})**\n\n"
                    f"{document.page_content}"
                )
    else:
        st.info("В архиве не найдено близких по смыслу статей — новизна выглядит высокой.")


def main() -> None:
    use_rag = render_sidebar()
    db = get_vector_db() if use_rag else None
    if use_rag and db is None:
        st.sidebar.warning(
            "Индекс не найден: соберите его командой `python scripts/build_rag_index.py`. "
            "Проверка новизны отключена."
        )
        use_rag = False

    st.title("🎓 Система авторецензирования научных статей")
    st.markdown(
        "Загрузите рукопись для первичного анализа (Desk Reject) "
        "и формирования структурированной справки."
    )

    uploaded_file = st.file_uploader("Загрузите файл статьи (PDF или TXT)", type=["txt", "pdf"])
    if uploaded_file is None or not st.button("🚀 Начать анализ статьи", type="primary"):
        return

    with st.status("Идет обработка рукописи...", expanded=True) as status:
        st.write("📄 Чтение документа...")
        article_text = extract_text_from_file(uploaded_file)
        if not article_text.strip():
            status.update(label="Не удалось извлечь текст", state="error")
            st.error("Файл пуст или текст не извлекается (возможно, скан без OCR).")
            return

        rag_results: list = []
        rag_context = ""
        if use_rag:
            st.write("🔍 Поиск похожих статей в архиве (RAG)...")
            rag_results = search_similar_articles(article_text, db)
            rag_context = build_rag_context(rag_results)

        st.write("🧠 Работа генеративной модели (локальная Ollama)...")
        try:
            review = generate_review(article_text, rag_context)
        except Exception as error:  # noqa: BLE001 - показываем пользователю текст ошибки
            status.update(label="Ошибка генерации", state="error")
            st.error(
                f"Сбой при обращении к модели: {error}\n\n"
                "Проверьте, что Ollama запущена и модель создана: "
                "`ollama create qwen-reviewer -f modelfiles/Modelfile`"
            )
            return
        status.update(label="Анализ завершен!", state="complete", expanded=False)

    render_result(review, rag_results, use_rag)


if __name__ == "__main__":
    main()
