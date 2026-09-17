# Сэмпл синтетического датасета

`dataset_sample.jsonl` — первые 10 записей из полного датасета
`../synthetic_reviews_dataset_v2.jsonl` (1250 записей, ~68 МБ), который не хранится
в git из-за размера. Здесь важен **формат**, а не объём.

## Формат записи (SFT, instruction-tuning)

```json
{
  "instruction": "Проанализируй предоставленный отрывок научной статьи и составь формальную рецензию в формате JSON.",
  "input": "<текст статьи (в сэмпле обрезан до 4000 символов)>",
  "output": "{\"summary\": \"...\", \"methodology_score\": 6, \"novelty_score\": 4, ...}",
  "metadata": {
    "source_file": "elibrary_32767712_92688271.txt",
    "augmentation_type": "shuffled_logic",
    "sample_truncated": true
  }
}
```

Поля:

| Поле | Описание |
|---|---|
| `instruction` | фиксированная инструкция (единая для всех записей) |
| `input` | текст статьи после очистки и аугментации |
| `output` | JSON-строка с рецензией модели-учителя (DeepSeek-V3): `summary`, `methodology_score`, `novelty_score`, `strengths`, `weaknesses`, `final_verdict` |
| `metadata.source_file` | имя исходного `.txt` из архива eLibrary |
| `metadata.augmentation_type` | `original` / `truncated_no_conclusions` / `shuffled_logic` |
| `metadata.sample_truncated` | `true` только в этом сэмпле (для читаемости на GitHub) |

## Распределение полного датасета

| Аугментация | Записей | Доля |
|---|---|---|
| `original` | 863 | 69.0 % |
| `shuffled_logic` (перемешаны абзацы) | 199 | 15.9 % |
| `truncated_no_conclusions` (обрезаны 30 % хвоста) | 188 | 15.0 % |
| **Итого** | **1250** | 100 % |

Вердикты учителя: «Отправить на доработку» — 1070 (85.6 %), «Отклонить» — 113 (9.0 %),
«Принять» — 67 (5.4 %). Средние оценки: методология 6.17 / 10, новизна 5.25 / 10.

Подробнее о метриках и их обсуждении — `docs/results.md`.
