# PLAN.md — ProcessTransformer PBPM Project

## О проекте

**Курс:** Process Mining, Hof University of Applied Sciences  
**Преподаватель:** Prof. Dr. Lars Ackermann-Igl  
**Тип экзамена:** Term Paper + Poster + Video + Peer Review  
**Группа:** Emir Orozbekov, Myktybek Abdykayimov, Ramazan Batanov  
*(Фактически работает один Emir — все вклады помечать именем!)*

**Тема:**  
> *Transformer-Based Multi-Task Predictive Business Process Monitoring: A Re-Implementation and Empirical Evaluation of ProcessTransformer on Real-Life Event Logs*

**Реализуемый алгоритм:** ProcessTransformer (Bukhsh et al., 2021) — Transformer-based модель для PBPM. Модель **обучается на event log данных** (не prompt engineering) — это соответствует требованию профессора о том, что решение должно включать обучение AI модели.

**Датасеты:** BPI Challenge 2012, BPI Challenge 2017 (event logs, XES формат)

**Baselines:**
- Most Frequent Class (MFC) — для предсказания следующей активности (classification)
- Mean Remaining Time — для предсказания оставшегося времени (regression)

---

## Требования профессора (критически важно)

### Правила vibe coding (LISA)
- Весь код пишется **только через LISA** (lisa-flash-03-2026) — университетский AI
- **Нельзя вручную редактировать код** — даже мелкие исправления делаются только через LISA-промпты
- **Нельзя передавать название алгоритма напрямую** — нужно описывать логику шаг за шагом
- Каждый промпт — **один логический шаг** (одна функция / один модуль)
- Нельзя вставлять в LISA чужой код или копипасту из статей
- `pm4py` разрешён **только** для: загрузки XES → pd.DataFrame и реализации/конвертации Petri Net
- Реализация разбита на **3 отдельных LISA-чата**

### Требование: AI модель должна обучаться на данных
Профессор уточнил (Moodle announcement): подход, основанный исключительно на prompt engineering LLM для предсказания, **не допускается**. ProcessTransformer соответствует требованию — он обучается на event log данных с нуля (нет fine-tuning, нет prompt engineering, есть полноценный training loop).

### Требование: цитирование
Профессор уточнил: использовать **IEEE citation style** (номерные ссылки [1], [2]...), но логику расстановки цитат — как в APA: каждое утверждение, не являющееся собственным выводом, должно иметь ссылку. Если целый абзац основан на одном источнике — одна ссылка в конце. Подробнее: https://apastyle.apa.org/style-grammar-guidelines/citations/appropriate-citation

### Структура LISA-чатов
- **Chat 1:** Baseline техники + загрузка данных
- **Chat 2:** ProcessTransformer (основная модель)
- **Chat 3:** Experiment pipeline (split, save, plot)

### Deliverables (дедлайн **7 июня 2026, 23:59**)
Всё сдаётся **одним ZIP через Moodle** — поздние сдачи не принимаются:
1. `term_paper.pdf` — 6–8 стр., IEEE two-column
2. `poster.pdf` — DIN A0, 300–500 слов
3. `video.mp4` (или ссылка) — 10 минут
4. `chat_exports/` — PDF + JSON всех 3 LISA-чатов
5. `src/` — весь код
6. `experiments/results/` — CSV + PNG с результатами
7. `LISA_MODEL_VERSION.txt`
8. `run_experiments.py` — должен воспроизводить эксперименты одной командой

**Дедлайн Peer Review:** 21 июня 2026, 23:59

### Важные требования к paper
- Каждый вклад участника **помечается именем** (иначе не засчитывается)
- IEEE two-column template (шаблон на Moodle)
- Обязательный раздел **Experience Report** — анализ LISA-чатов
- Appendix: все экспортированные LISA-чаты (не считается в 6–8 стр.)
- Версия LISA задокументирована: `lisa-flash-03-2026`

### Требования к video
- 10 минут **строго**
- Не обязательно показывать лицо, но голос должен идентифицировать спикера
- Все участники группы должны присутствовать голосом (Emir один — это ок)
- Рекомендуется фокус на: результаты экспериментов + experience с vibe coding
- Кратко: датасеты, реализованные подходы, experiment setup — затем основной фокус

---

## Структура проекта

```
project-root/
├── data/                            # НЕ в git (.gitignore)
│   ├── BPI_Challenge_2012.xes
│   └── BPI_Challenge_2017.xes
├── src/
│   ├── data_loader.py               ✅ DONE
│   ├── preprocessor.py              ✅ DONE
│   ├── baseline.py                  ✅ DONE
│   ├── evaluate.py                  ✅ DONE
│   ├── model.py                     ✅ DONE
│   └── train.py                     ✅ DONE
├── experiments/
│   ├── run_experiments.py           ✅ DONE
│   └── results/                     ✅ DONE (результаты сохранены)
├── src/utils.py                     ✅ DONE (split, save, plot)
├── chat_exports/
│   ├── chat1_baseline.pdf/.json     ❌ TODO (экспортировать из LISA)
│   ├── chat2_model.pdf/.json        ❌ TODO
│   └── chat3_experiments.pdf/.json  ❌ TODO
├── LISA_MODEL_VERSION.txt           ✅ DONE
├── .gitignore                       ✅ DONE
└── README.md                        ❌ TODO
```

---

## Что уже сделано (детально)

### Chat 1 — Baseline (ЗАВЕРШЁН ✅)

| Файл | Что делает | Статус |
|------|-----------|--------|
| `data_loader.py` | Загружает XES через pm4py, фильтрует COMPLETE lifecycle events (~164K events) | ✅ |
| `preprocessor.py` | Генерирует все префиксы трейсов (~151K префиксов) | ✅ |
| `baseline.py` | MFC accuracy: **~15.83%** · Mean RT MAE: **~9.64 days** | ✅ |
| `evaluate.py` | Метрики: accuracy, MAE | ✅ |

**Ключевые числа:**
- Событий после фильтрации: ~164,000
- Префиксов: ~151,419
- Max prefix length: 95
- MFC Baseline Accuracy: ~15.83%
- Mean Remaining Time MAE: ~9.64 days

### Chat 2 — ProcessTransformer (ЗАВЕРШЁН ✅)

| Компонент | Детали | Статус |
|-----------|--------|--------|
| Vocabulary | 23 активности | ✅ |
| `encode_and_pad` | Padded tensors (151,419 × 95), dtype int64 | ✅ |
| `SelfAttentionBlock` | MultiheadAttention + residual + LayerNorm, batch_first=True | ✅ |
| `ProcessTransformer` | ~36,632 параметров, embedding + attention stack + 2 heads | ✅ |
| `train.py` | Adam, CrossEntropyLoss + MSELoss (normalized RT), MPS backend | ✅ |

**Архитектура:**
- Embedding: (vocab_size+1) × embed_dim, padding_idx=0
- Stack: nn.ModuleList из SelfAttentionBlock'ов
- Output head 1: Linear → num_activities (next activity classification)
- Output head 2: Linear → 1 (remaining time regression)
- Pooling: mean over sequence dimension

**Важно:** remaining time нормализован перед обучением (иначе MSE нестабилен)

### Chat 3 — Experiment Pipeline (ЗАВЕРШЁН ✅)

| Функция | Детали | Статус |
|---------|--------|--------|
| `split_by_time` | 80/20 temporal split → ~123,880 train / ~27,539 test | ✅ |
| `save_results` | CSV в `experiments/results/` | ✅ |
| `plot_comparison` | matplotlib сравнение baseline vs model | ✅ |

**Производительность:** ~1m45s на эпоху на M1 MPS

---

## Что ещё нужно сделать

### 1. Тиммейты: добавить свои техники ← СЛЕДУЮЩИЙ ШАГ
Myktybek и Ramazan реализуют свои PBPM-техники через LISA (каждый в своём чате).
Свои файлы кладут в `src/`. Потом добавляют блок в `run_experiments.py` через корректирующий промпт в Chat 3.

### 2. Term Paper (6–8 стр., IEEE two-column)
Структура:
1. Abstract (~150 слов)
2. Introduction
3. Background (PBPM pipeline, Transformer, prefix-based approach)
4. Methodology (baseline + ProcessTransformer, детали реализации)
5. Experiment Setup (датасеты, метрики, split strategy)
6. Results & Discussion (таблицы + графики из экспериментов)
7. **Experience Report** ← special ingredient (анализ LISA-чатов)
8. Conclusion
9. References (IEEE стиль, [1]...[6])
10. Appendix: LISA-чаты (не в лимит 6–8 стр.)

**Про цитирование в paper:** IEEE стиль [1], логика расстановки — как APA. Каждое фактическое утверждение со ссылкой. Цитат дословных избегать, парафраз — норма в CS.

### 4. Poster (Canva, DIN A0)
- 300–500 слов, высокий контраст
- Акцент: результаты + GenAI/vibe coding insights
- Экспорт PDF

### 5. Video (10 минут)
- Лицо необязательно, голос обязателен
- Структура: краткое intro (датасет, подходы, setup) → основной фокус на результаты и experience с LISA
- Строго 10 минут

### 6. Экспорт LISA-чатов
Экспортировать Chat 1, 2, 3 из LISA как PDF + JSON → `chat_exports/`

### 7. ZIP и сдача через Moodle (до 7 июня 23:59)

### 8. Peer Review (до 21 июня 23:59)

---

## Результаты экспериментов (31 мая 2026)

### BPI Challenge 2012
| Метрика | Statistical Baseline | ProcessTransformer |
|---------|---------------------|--------------------|
| Avg Accuracy | 8.79% | **63.08%** |
| Avg MAE | 9.47 days | **4.88 days** |

**Данные:** 164,506 events · 151,419 prefixes · vocab 23 · max_len 95  
**Split:** 123,880 train / 27,539 test  
**Training:** 10 epochs · final loss 1.3571 · ~1m45s/epoch on M1 MPS

### BPI Challenge 2017
| Метрика | Statistical Baseline | ProcessTransformer |
|---------|---------------------|--------------------|
| Avg Accuracy | 3.74% | **75.17%** |
| Avg MAE | 13.19 days | **4.91 days** |

**Данные:** 475,306 events (no lifecycle filter — BPI 2017 не имеет COMPLETE тега) · 443,797 prefixes · vocab 24 · max_len 53  
**Split:** 355,371 train / 88,426 test  
**Training:** 10 epochs · final loss 1.0866

**Примечание:** BPI 2017 потребовал fallback в data_loader (lifecycle:transition отсутствует) — это Challenge #5 для Experience Report.

---

## Experience Report — материал для paper

Пять задокументированных challenge при vibe coding:

1. **Multi-lifecycle transitions** — BPI 2012 содержит несколько lifecycle transition-ов на событие. Изначальный data loading prompt не учитывал это. Потребовался дополнительный prompt для фильтрации только COMPLETE событий. *Урок: инспектировать датасет перед первым промптом.*

2. **Typo в имени класса** — LISA сгенерировала `SeldAttentionBlock` вместо `SelfAttentionBlock`. Поскольку ручное исправление запрещено правилами, потребовался corrective prompt. *Урок: внимательно проверять сгенерированный код перед интеграцией.*

3. **Sparse padding** — max prefix length 95 создаёт сильно разреженные входные тензоры (большинство префиксов намного короче). *Урок: анализировать распределение длин трейсов заранее.*

4. **Нормализация remaining time** — без нормализации MSE loss нестабилен из-за масштаба значений в секундах. Потребовался дополнительный prompt для добавления нормализации. *Урок: для регрессионных задач всегда нормализовать таргет.*

5. **BPI 2017 lifecycle filter** — BPI 2017 не содержит стандартного `lifecycle:transition == 'COMPLETE'`, поэтому data_loader возвращал 0 событий. Потребовался corrective prompt для добавления fallback-логики (case-insensitive → no filter). *Урок: датасеты из разных источников имеют разные форматы — проверять pipeline на каждом новом датасете.*

---

## Ключевые технические детали

**Среда:** MacBook Air M1 · Anaconda · Python 3.10 · VS Code · PyTorch MPS  
**Repo:** GitHub (private)  
**LISA version:** `lisa-flash-03-2026`  
**pm4py:** только XES загрузка + Petri Net

**Референсы:**
1. Bukhsh et al. (2021) — ProcessTransformer, arXiv:2104.00721
2. Tax et al. (2017) — LSTM baseline, DOI:10.1007/978-3-319-59536-8_30
3. Rama-Maneiro et al. (2023) — Deep Learning for PBPM review, IEEE TSC
4. Vaswani et al. (2017) — Attention Is All You Need, NeurIPS
5. van Dongen (2012) — BPI Challenge 2012, 4TU.ResearchData
6. van Dongen (2017) — BPI Challenge 2017, 4TU.ResearchData
