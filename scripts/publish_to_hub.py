"""Публикация артефактов модели (LoRA-адаптер, GGUF) на HuggingFace Hub.

Веса слишком тяжёлые для git (адаптер ~150 МБ, GGUF ~4.4 ГБ), поэтому
они живут на HF Hub, а в репозитории остаются только код и карточка модели
(см. docs/model_card.md).

Перед запуском:
    pip install huggingface_hub
    huggingface-cli login

Примеры:
    python scripts/publish_to_hub.py --repo-id <user>/qwen2.5-7b-reviewer-lora \
        --folder lora_reviewer_final
    python scripts/publish_to_hub.py --repo-id <user>/qwen2.5-7b-reviewer-gguf \
        --folder qwen-reviewer_gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Загрузка артефактов модели на HuggingFace Hub")
    parser.add_argument("--repo-id", required=True, help="Например: username/model-name")
    parser.add_argument("--folder", type=Path, default=ROOT / "lora_reviewer_final",
                        help="Локальная папка с адаптером или GGUF")
    parser.add_argument("--private", action="store_true", help="Создать приватный репозиторий")
    parser.add_argument("--commit-message", default="Upload diploma reviewer model artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.folder if args.folder.is_absolute() else (ROOT / args.folder)
    if not folder.exists():
        raise SystemExit(f"Папка не найдена: {folder}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=args.commit_message,
    )
    print(f"✅ Загружено: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
