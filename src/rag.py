"""RAG-контур: поиск похожих статей в локальном архиве для оценки новизны.

Эмбеддинги: `intfloat/multilingual-e5-large`. Важная деталь этой модели —
обязательные префиксы `query:` / `passage:` для запросов и документов;
класс `E5Embeddings` добавляет их прозрачно, чтобы индекс и поиск были
построены согласованно (рассинхрон префиксов ломает косинусную близость).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

try:  # актуальный пакет
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback для старых окружений
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_MAX_DISTANCE,
    RAG_QUERY_CHARS,
    RAG_SNIPPET_CHARS,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)


class E5Embeddings(HuggingFaceEmbeddings):
    """e5-эмбеддинги с автоматическими префиксами `passage:` / `query:`."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> E5Embeddings:
    """Создаёт модель эмбеддингов (нормализация нужна для L2-поиска в Chroma)."""
    return E5Embeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def load_vector_db(persist_directory: str | Path = CHROMA_DB_DIR) -> Chroma | None:
    """Открывает существующий индекс. None — если база ещё не собрана."""
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        logger.warning(
            "Векторная база не найдена: %s (см. scripts/build_rag_index.py)", persist_path
        )
        return None
    return Chroma(persist_directory=str(persist_path), embedding_function=get_embeddings())


def search_similar_articles(text: str, db: Chroma | None, k: int = RAG_TOP_K):
    """Топ-k ближайших статей из архива (L2 distance, меньше — ближе)."""
    if db is None:
        return []
    query = text[:RAG_QUERY_CHARS]
    if not query.strip():
        return []
    try:
        return db.similarity_search_with_score(query, k=k)
    except Exception as error:  # noqa: BLE001 - поиск не должен ронять UI
        logger.warning("Поиск по векторной базе не удался: %s", error)
        return []


def build_rag_context(
    results: Iterable,
    max_distance: float = RAG_MAX_DISTANCE,
    snippet_chars: int = RAG_SNIPPET_CHARS,
) -> str:
    """Собирает текстовый контекст из результатов поиска (только релевантные)."""
    snippets = [
        f"- {doc.page_content[:snippet_chars]}..."
        for doc, score in results
        if score < max_distance
    ]
    return "\n".join(snippets)


def build_vector_db(
    articles: Iterable[tuple[str, dict]],
    persist_directory: str | Path = CHROMA_DB_DIR,
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
) -> Chroma:
    """Строит и сохраняет индекс из пар (текст_статьи, метаданные).

    Используется в `scripts/build_rag_index.py`.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "]
    )
    texts: list[str] = []
    metadatas: list[dict] = []
    for article_text, meta in articles:
        for chunk in splitter.split_text(article_text):
            if len(chunk.strip()) < 200:  # огрызки не индексируем
                continue
            texts.append(chunk)
            metadatas.append(dict(meta))

    logger.info("Чанков к индексации: %s", len(texts))
    return Chroma.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        metadatas=metadatas,
        persist_directory=str(Path(persist_directory)),
    )
