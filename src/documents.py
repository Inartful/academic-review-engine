"""Извлечение текста из рукописей: PDF (PyPDF2) и TXT."""

from __future__ import annotations

from pathlib import Path

import PyPDF2


def extract_text_from_pdf(file_obj) -> str:
    """Читает все страницы PDF и склеивает текст."""
    reader = PyPDF2.PdfReader(file_obj)
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_file(uploaded_file) -> str:
    """Достаёт текст из файла, загруженного через Streamlit.

    `uploaded_file` — объект с атрибутами `.name` и `.getvalue()`
    (в тестах достаточно подсунуть любой объект с таким интерфейсом).
    """
    name = getattr(uploaded_file, "name", "")
    if str(name).lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def read_text_file(path: str | Path) -> str:
    """Читает локальный .txt с исходной статьёй (архив eLibrary бывает с битой кодировкой)."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")
