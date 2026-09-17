"""Сборка векторного индекса архива статей для RAG-контура проверки новизны.

Индексируется локальный архив выгрузок eLibrary (`dataset/data_3/**/*.txt`).
Скрипт не входит в git-историю данных: исходные статьи не публикуются,
но воспроизводимость индекса обеспечена самим кодом.

Запуск из корня репозитория:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --dataset-root dataset/data_3 --limit 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

# Чтобы `import src.*` работал при запуске как `python scripts/build_rag_index.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CHROMA_DB_DIR,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAW_ARTICLES_DIR,
)
from src.documents import read_text_file  # noqa: E402
from src.rag import build_vector_db  # noqa: E402

logger = logging.getLogger("build_rag_index")
MIN_ARTICLE_CHARS = 500


def iter_articles(root: Path, limit: int | None = None) -> Iterator[tuple[str, dict]]:
    """Пробегает по архиву и отдаёт пары (текст_статьи, метаданные)."""
    paths = sorted(root.rglob("*.txt"))
    if limit is not None:
        paths = paths[:limit]
    logger.info("Найдено файлов: %s", len(paths))
    for path in paths:
        text = read_text_file(path)
        if len(text) < MIN_ARTICLE_CHARS:
            continue
        yield text, {"source": path.name, "category": path.parent.name, "path": str(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка ChromaDB-индекса архива статей")
    parser.add_argument("--dataset-root", type=Path, default=RAW_ARTICLES_DIR,
                        help="Каталог с .txt-статьями (рекурсивно)")
    parser.add_argument("--persist-dir", type=Path, default=CHROMA_DB_DIR,
                        help="Куда сохранить векторный индекс")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число статей")
    parser.add_argument("--chunk-size", type=int, default=RAG_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=RAG_CHUNK_OVERLAP)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    if not args.dataset_root.exists():
        logger.error("Каталог с архивом не найден: %s", args.dataset_root)
        return 1

    articles = list(iter_articles(args.dataset_root, args.limit))
    if not articles:
        logger.error("Не найдено ни одного .txt длиннее %s символов", MIN_ARTICLE_CHARS)
        return 1

    db = build_vector_db(
        articles,
        persist_directory=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info(
        "Готово. Индекс сохранён в %s (коллекция: %s)", args.persist_dir, db._collection.name
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
