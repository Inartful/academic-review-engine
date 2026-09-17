# Model Card: Qwen2.5-7B Reviewer (LoRA + GGUF)

## Общее

| Параметр | Значение |
|---|---|
| Базовая модель | `unsloth/Qwen2.5-7B-Instruct` (4-bit, `unsloth-bnb-4bit`) |
| Метод | QLoRA / SFT (TRL `SFTTrainer`, `packing=False`) |
| LoRA | `r=16`, `lora_alpha=16`, `lora_dropout=0`, `bias="none"` |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Контекст | 4096 токенов |
| Батч / накопление | 1 / 8, `adamw_8bit`, `lr=2e-4` (linear, 10 warmup) |
| Эпох / шагов | 2 / 282 |
| Итог | `train_loss=0.6708`, `eval_loss=1.4642` |
| Артефакты | `lora_reviewer_final/` (адаптер, ~150 МБ), `qwen-reviewer_gguf/` (GGUF Q4_K_M, ~4.4 ГБ) |
| Датасет | `synthetic_reviews_dataset_v2.jsonl` (1250 записей, distillation DeepSeek-V3) |
| Лицензия | MIT (код); для базовой модели — лицензия Qwen2.5 (Apache-2.0) |

## Назначение

Модель — часть DSS авторецензирования: получает текст научной статьи и возвращает
JSON-рецензию (резюме, сильные/слабые стороны, оценки методологии и новизны, вердикт).

**Область применения:** предварительная экспертиза (desk review), помощь редактору,
учебные задачи. **Не является** заменой рецензента-человека и не должна использоваться
как единственный критерий принятия решения о публикации.

## Формат вывода

Схема задаётся на уровне декодирования (`format` в Ollama), ключи — русские:
`резюме`, `сильные_стороны`, `слабые_стороны`, `оценка_методологии`, `оценка_новизны`, `вердикт`.

```bash
# 1. Положить в папку файл qwen2.5-7b-instruct.Q4_K_M.gguf (~4.4 ГБ)
# 2. Создать модель в Ollama
ollama create qwen-reviewer -f modelfiles/Modelfile
# 3. Проверить
ollama run qwen-reviewer "Проанализируй текст: <текст статьи>"
```

## Ограничения и риски

* **Наследование смещений учителя**: вердикты в датасете однообразны
  (85.6 % — «Отправить на доработку», «Принять» — всего 5.4 %), поэтому модель
  редко уверенно принимает работы.
* **Языковой сдвиг** на статьях с большим английским фрагментом (1 кейс из 20
  в robustness-тесте) — лечится промптом, но не гарантирован.
* **Рассогласование вердикта и оценок** — закрыто на уровне системы
  (`src/censor.py`), сама модель по-прежнему может противоречить себе.
* **Слабая конкретика в критике**: средний балл полноты у LLM-судьи — 5.2/10.
* Модель не проверяет фактологию, не ищет плагиат и не валидирует статистику.

## Как опубликовать артефакты (HuggingFace Hub)

Веса не хранятся в git (лимиты GitHub — 100 МБ на файл):

```bash
pip install huggingface_hub
huggingface-cli login

# адаптер LoRA
python scripts/publish_to_hub.py \
    --repo-id <username>/qwen2.5-7b-reviewer-lora \
    --folder lora_reviewer_final

# GGUF для локального инференса
python scripts/publish_to_hub.py \
    --repo-id <username>/qwen2.5-7b-reviewer-gguf \
    --folder qwen-reviewer_gguf
```

После публикации добавьте ссылку в `README.md` (бейдж модели) — так ревьюер увидит,
что веса реально существуют, не скачивая 4.4 ГБ из релиза.

## Как воспроизвести

1. `notebooks/01_dataset_generation.ipynb` — генерация датасета (нужен `CHUTES_API_TOKEN`);
2. `notebooks/02_lora_finetuning.ipynb` — обучение и экспорт GGUF (~2.6 ч на одной GPU);
3. `notebooks/03_evaluation.ipynb` — robustness-тест и LLM-as-a-Judge.
