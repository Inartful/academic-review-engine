"""Проверка, что ноутбуки в notebooks/ — валидный JSON без сохранённых выводов.

Запускается в CI: чтобы «тяжёлые» outputs (логи, base64-картинки) и рассинхрон
номеров ячеек не возвращались в репозиторий.

    python scripts/check_notebooks.py
"""

from __future__ import annotations

import json
import pathlib

NOTEBOOKS_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
MAX_NOTEBOOK_BYTES = 300 * 1024


def check_notebook(path: pathlib.Path) -> list[str]:
    """Возвращает список проблем для одного ноутбука."""
    problems: list[str] = []
    size = path.stat().st_size
    if size > MAX_NOTEBOOK_BYTES:
        problems.append(f"{path.name}: {size // 1024} КБ — слишком большой для «чистого» ноутбука")

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return problems + [f"{path.name}: невалидный JSON ({error})"]

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            problems.append(f"{path.name}: ячейка {index} содержит сохранённые выводы")
        if cell.get("execution_count") is not None:
            problems.append(f"{path.name}: ячейка {index} содержит execution_count")
    return problems


def main() -> int:
    if not NOTEBOOKS_DIR.exists():
        print(f"Каталог {NOTEBOOKS_DIR} не найден")
        return 1

    problems: list[str] = []
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    for path in notebooks:
        problems.extend(check_notebook(path))

    if problems:
        print("❌ Проблемы с ноутбуками:")
        for problem in problems:
            print("  -", problem)
        return 1

    print(f"✅ Ноутбуки в порядке: {len(notebooks)} шт., выводы очищены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
