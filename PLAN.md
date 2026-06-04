# PLAN.md — ProcessTransformer PBPM Project

## О проекте

**Курс:** Process Mining, Hof University of Applied Sciences  
**Преподаватель:** Prof. Dr. Lars Ackermann-Igl  
**Тип экзамена:** Term Paper + Poster + Video + Peer Review  
**Группа:** Emir Orozbekov, Myktybek Abdykayimov, Ramazan Batanov, Aizhan Kozhamuratova

**Тема:**  
> *Transformer-Based Multi-Task Predictive Business Process Monitoring: A Re-Implementation and Empirical Evaluation of ProcessTransformer on Real-Life Event Logs*

**Датасеты:** BPI Challenge 2012, BPI Challenge 2017 (XES формат)

**Подходы:**
- Statistical Baseline (MFC + Mean RT) — Emir
- ProcessTransformer (Bukhsh et al., 2021) — Emir
- CNN (Pasquadibisceglie et al., 2019) — Myktybek
- GRU (custom) — Ramazan
- RETAIN (Choi et al., 2016) — Aizhan

---

## Требования профессора (критически важно)

### Правила vibe coding (LISA)
- Весь код пишется **только через LISA** (lisa-flash-03-2026)
- Нельзя вручную редактировать код — только через corrective LISA-промпты
- Нельзя называть алгоритм по имени в промпте
- Каждый промпт — один логический шаг
- `pm4py` только для: загрузки XES → pd.DataFrame и Petri Net

### Распределение ролей (n=4)
Профессор требует: 1 студент = только baseline+framework, 3 студента = по одной технике.
Emir делает и baseline+framework И ProcessTransformer — профессор указал на это. В paper нужно чётко разграничить authorship.

### Deliverables (дедлайн **7 июня 2026, 23:59**)
ZIP через Moodle:
1. `term_paper.pdf` — 6–8 стр., IEEE two-column
2. `poster.pdf` — DIN A0, 300–500 слов
3. `video.mp4` — 10 минут строго, все 4 участника голосом
4. `chat_exports/` — PDF + JSON всех LISA-чатов каждого участника
5. `src/` — весь код
6. `experiments/results/` — CSV + PNG
7. `LISA_MODEL_VERSION.txt`
8. `run_experiments.py`

**Peer Review:** 21 июня 2026, 23:59

---

## Статус проекта (на 2 июня 2026)

| Задача | Статус |
|--------|--------|
| Chat 1 (Emir): data_loader, preprocessor, baseline, evaluate | ✅ Done |
| Chat 2 (Emir): ProcessTransformer, train | ✅ Done |
| Chat 3 (Emir): utils, run_experiments.py | ✅ Done |
| CNN (Myktybek) | ✅ Done |
| GRU (Ramazan) | ⚠️ Баг r_concat исправляется |
| RETAIN (Aizhan) | ✅ Done |
| run_experiments.py полный (все модели) | ✅ Done |
| **Финальный запуск экспериментов** | ❌ **TODO ← СЛЕДУЮЩИЙ ШАГ** |
| Term Paper | ❌ TODO |
| Poster | ❌ TODO |
| Video | ❌ TODO |
| Экспорт LISA чатов | ❌ TODO |
| README.md | ❌ TODO |
| ZIP + Moodle | ❌ TODO |
| Peer Review | ❌ TODO |

---

## Структура проекта

```
project-root/
├── data/                              # НЕ в git
│   ├── BPI_Challenge_2012.xes.gz
│   └── BPI_Challenge_2017.xes.gz
├── src/
│   ├── data_loader.py                 ✅ Emir
│   ├── preprocessor.py                ✅ Emir
│   ├── baseline.py                    ✅ Emir
│   ├── evaluate.py                    ✅ Emir
│   ├── model.py                       ✅ Emir (ProcessTransformer)
│   ├── train.py                       ✅ Emir
│   ├── utils.py                       ✅ Emir (split, save, plot)
│   ├── myktybek_model.py              ✅ Myktybek (CNN)
│   ├── myktybek_train.py              ✅ Myktybek
│   ├── gru_model.py                   ⚠️ Ramazan (баг исправляется)
│   ├── gru_train.py                   ✅ Ramazan
│   ├── aizhan_retain_model.py         ✅ Aizhan (RETAIN)
│   └── aizhan_retain_train.py         ✅ Aizhan
├── experiments/
│   ├── run_experiments.py             ✅ все 4 модели
│   └── results/                       ✅ в git (предварительные данные)
├── chat_exports/
│   ├── emir/                          ❌ TODO
│   ├── myktybek/                      ❌ TODO
│   ├── ramazan/                       ❌ TODO
│   └── aizhan/                        ❌ TODO
├── LISA_MODEL_VERSION.txt             ✅ Done
├── README.md                          ❌ TODO
└── .gitignore                         ✅ Done
```

---

## Детали реализации

### Emir — Chat 1 (Baseline + Data)
| Файл | Что делает |
|------|-----------|
| `data_loader.py` | XES → DataFrame, COMPLETE filter с fallback (case-insensitive → no filter) |
| `preprocessor.py` | Prefix generation, vocabulary (1..N, 0=pad), encode_and_pad |
| `baseline.py` | MostFrequentClassBaseline, MeanRemainingTimeBaseline |
| `evaluate.py` | accuracy(), mae(), evaluate_by_prefix_length() |

### Emir — Chat 2 (ProcessTransformer)
- SelfAttentionBlock: MultiheadAttention + residual + LayerNorm, batch_first=True
- ProcessTransformer: Embedding(vocab_size+1) → attention stack → mean pool → 2 heads
- ~36,632 параметров
- train.py: Adam, CrossEntropyLoss + MSELoss, normalized RT, MPS support

### Emir — Chat 3 (Experiments)
- split_by_time: 80/20 temporal split по case order
- save_results: CSV в experiments/results/
- plot_comparison: matplotlib линейные графики по prefix length
- run_experiments.py: оркестрация всех моделей

### Myktybek — CNN
- MyktbekModel: Embedding → 3 параллельных Conv1d (kernel 3/5/7) → BN → ReLU → MaxPool → concat → 2 heads
- Совместим с общим DataLoader (batch, seq_len)

### Ramazan — GRU
- SequenceActivityPredictor: custom GRU cell (W_z, W_r, W_h) → 2 heads
- **Текущий баг:** r_concat использует `r * x_t` вместо `r * h` → shape mismatch (128 vs 64) → исправляется

### Aizhan — RETAIN
- RETAINModel: Embedding → reversed sequence → GRU_alpha (scalar weights) + GRU_beta (vector weights) → context = sum(alpha * beta) → 2 heads
- Отличие от оригинала: context не умножается на оригинальные embeddings (упрощение LISA)

---

## Предварительные результаты (31 мая 2026, только ProcessTransformer)

### BPI Challenge 2012
| Метрика | Statistical Baseline | ProcessTransformer |
|---------|---------------------|--------------------|
| Avg Accuracy | 8.79% | **63.08%** |
| Avg MAE | 9.47 days | **4.88 days** |

164,506 events · 151,419 prefixes · vocab 23 · max_len 95  
Train 123,880 / Test 27,539 · 10 epochs · final loss 1.3571

### BPI Challenge 2017
| Метрика | Statistical Baseline | ProcessTransformer |
|---------|---------------------|--------------------|
| Avg Accuracy | 3.74% | **75.17%** |
| Avg MAE | 13.19 days | **4.91 days** |

475,306 events · 443,797 prefixes · vocab 24 · max_len 53  
Train 355,371 / Test 88,426 · 10 epochs · final loss 1.0866

**Финальные результаты (CNN + GRU + RETAIN) — ещё не получены.**

---

## Experience Report — все challenges

### Chat 1–3 (Emir):
1. **Multi-lifecycle transitions** — BPI 2012 имеет несколько lifecycle transitions на событие. Первый промпт не фильтровал COMPLETE → corrective prompt. *Урок: инспектировать датасет до написания промпта.*
2. **Typo `SeldAttentionBlock`** — LISA сгенерировала опечатку в имени класса. Ручное исправление запрещено → corrective prompt. *Урок: проверять сгенерированный код до интеграции.*
3. **Sparse padding** — max_len 95, большинство трейсов намного короче → плотность входных тензоров низкая. *Урок: анализировать распределение длин заранее.*
4. **Remaining time нормализация** — MSE loss нестабилен из-за масштаба в секундах → corrective prompt для нормализации. *Урок: для регрессии всегда нормализовать таргет.*
5. **BPI 2017 lifecycle filter** — data_loader возвращал 0 событий, так как BPI 2017 не использует COMPLETE → corrective prompt с fallback-логикой. *Урок: разные датасеты имеют разные форматы.*

### Интеграция тиммейтов:
6. **RETAIN → LSTM → RETAIN** — Aizhan переключилась на LSTM без согласования, профессор потребовал вернуться к RETAIN → corrective prompts.
7. **Дублирование аргументов** — LISA продублировала аргументы функций при добавлении LSTM в run_experiments.py → corrective prompt.
8. **GRU Embedding размер** — `nn.Embedding(vocab_size)` вместо `(vocab_size+1)` → потенциальный IndexError → corrective prompt.
9. **GRU missing `return`** — forward() без return statement → corrective prompt.
10. **GRU r_concat shape mismatch** — reset gate применяется к x_t (embed_dim=64) вместо h (hidden_dim=128) → RuntimeError → corrective prompt (в процессе).
11. **MSELoss broadcasting** — shape (batch,) vs (batch,1) у нескольких моделей → неправильный градиент → corrective prompts.
12. **Двойное деление на 86400** — MAE конвертировался в дни дважды → corrective prompt.

---

## Что ещё нужно сделать

### 1. Исправить GRU баг (Ramazan) — срочно
Corrective prompt: заменить `r_concat = torch.cat([r * x_t, h], dim=1)` на `r_concat = torch.cat([x_t, r * h], dim=1)`

### 2. Финальный запуск экспериментов
```bash
python experiments/run_experiments.py
```
Ожидаемое время: ~3–4 часа (4 модели × 2 датасета)

### 3. Term Paper (6–8 стр., IEEE two-column)
Структура:
1. Abstract (~150 слов)
2. Introduction
3. Background (PBPM, prefix encoding, attention, GRU, CNN)
4. Methodology (4 техники + baseline, детали каждой)
5. Experiment Setup (датасеты, метрики, 80/20 temporal split)
6. Results & Discussion (таблицы + графики всех моделей)
7. **Experience Report** ← special ingredient (12 challenges выше)
8. Conclusion
9. References [1]–[7]
10. Appendix: все LISA-чаты

Каждый раздел помечается именем автора. Без этого — не засчитывается.

### 4. Poster (Canva, DIN A0)
- 300–500 слов, высокий контраст
- Акцент: сравнение результатов + vibe coding insights

### 5. Video (10 минут строго)
- Все 4 участника говорят голосом
- Структура: краткое intro → результаты → experience с LISA

### 6. Экспорт LISA-чатов
Каждый экспортирует свои чаты → PDF + JSON → `chat_exports/<имя>/`

### 7. README.md

### 8. ZIP + Moodle (до 7 июня 23:59)

### 9. Peer Review (до 21 июня 23:59)

---

## Референсы

1. Bukhsh, Z.A., Saeed, A., Dijkman, R., van Dongen, B., Kaymak, U. (2021). *ProcessTransformer: Predictive Business Process Monitoring with Transformer Network*. arXiv:2104.00721
2. Choi, E., Bahadori, M.T., Sun, J., Kulas, J., Schuetz, A., Stewart, W.F. (2016). *RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism*. NeurIPS 29
3. Pasquadibisceglie, V., Appice, A., Castellano, G., Malerba, D. (2019). *Using Convolutional Neural Networks for Predictive Process Analytics*. ICPM 2019. IEEE
4. Vaswani, A., Shazeer, N., Parmar, N. et al. (2017). *Attention Is All You Need*. NeurIPS
5. Rama-Maneiro, E., Vidal, J.C., Lama, M. (2023). *Deep Learning for Predictive Business Process Monitoring*. IEEE TSC
6. van Dongen, B.F. (2012). *BPI Challenge 2012*. 4TU.ResearchData
7. van Dongen, B.F. (2017). *BPI Challenge 2017*. 4TU.ResearchData

---

## Технические детали среды

**Среда:** MacBook Air M1 · Anaconda · Python 3.10 · VS Code · PyTorch MPS  
**Repo:** GitHub (private, Emirkeenz)  
**LISA version:** `lisa-flash-03-2026`
