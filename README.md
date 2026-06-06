# Predictive Business Process Monitoring with Deep Learning

**Course:** Process Mining — Hof University of Applied Sciences, 2026  
**Group:** Emir Orozbekov, Myktybek Abdykayimov, Ramazan Batanov, Aizhan Kozhamuratova

A comparative study of four deep learning models for next-activity prediction and remaining-time estimation on real-life event logs. All source code was written exclusively via LISA (lisa-flash-03-2026) following the vibe coding methodology.

---

## Team & Contributions

| Member | Contribution |
|--------|-------------|
| Emir Orozbekov | Data pipeline, Statistical Baseline, ProcessTransformer, experiment framework |
| Myktybek Abdykayimov | CNN model |
| Ramazan Batanov | GRU model |
| Aizhan Kozhamuratova | RETAIN model |

---

## Datasets

Datasets are **not included** in this repository. Download and place in `data/`:

- **BPI Challenge 2012:** https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- **BPI Challenge 2017:** https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b

Expected filenames:
```
data/BPI_Challenge_2012.xes.gz
data/BPI_Challenge_2017.xes.gz
```

---

## Setup

```bash
conda activate processtransformer
pip install -r requirements.txt
```

**Requirements:** Python 3.10, PyTorch 2.11, pm4py, pandas, matplotlib, scikit-learn (see `requirements.txt` for full list).  
MPS acceleration is used automatically on Apple Silicon if available.

---

## Running Experiments

```bash
python experiments/run_experiments.py
```

Runs all five approaches (Statistical Baseline, ProcessTransformer, CNN, GRU, RETAIN) on both datasets and saves results to `experiments/results/`.

Expected runtime: ~3–4 hours (4 models × 2 datasets, CPU/MPS).

---

## Project Structure

```
├── src/
│   ├── data_loader.py          # XES loading, COMPLETE filter with fallback
│   ├── preprocessor.py         # Prefix generation, vocabulary, encode_and_pad
│   ├── baseline.py             # MostFrequentClass + MeanRemainingTime baselines
│   ├── evaluate.py             # accuracy(), mae(), evaluate_by_prefix_length()
│   ├── model.py                # ProcessTransformer (Emir)
│   ├── train.py                # ProcessTransformer training loop
│   ├── utils.py                # split_by_time, save_results, plot_comparison
│   ├── myktybek_model.py       # CNN — 3x parallel Conv1d (Myktybek)
│   ├── myktybek_train.py       # CNN training
│   ├── gru_model.py            # Custom GRU cell (Ramazan)
│   ├── gru_train.py            # GRU training
│   ├── aizhan_retain_model.py  # RETAIN — reverse-time dual-GRU attention (Aizhan)
│   └── aizhan_retain_train.py  # RETAIN training
│
├── experiments/
│   ├── run_experiments.py      # Main script — all models, both datasets
│   └── results/
│       ├── accuracy_bpi2012.png
│       ├── accuracy_bpi2017.png
│       ├── mae_bpi2012.png
│       ├── mae_bpi2017.png
│       ├── *_results.csv          # Per-prefix-length accuracy and MAE
│       └── *_raw_predictions.csv  # Raw model predictions (input + output)
│
├── chat_exports/               # LISA chat logs (PDF + JSON per contributor)
│   ├── emir/                   # Chats 1–4
│   ├── myktybek/               # Chat 1
│   ├── ramazan/                # Chats 1–4
│   └── aizhan/                 # Chat 1
│
├── data/                       # Not in git — download separately (see above)
├── requirements.txt
└── LISA_MODEL_VERSION.txt
```

---

## Results

| Model | BPI-2012 Acc | BPI-2012 MAE | BPI-2017 Acc | BPI-2017 MAE |
|-------|-------------|-------------|-------------|-------------|
| Statistical Baseline | 8.79% | 9.47 d | 3.74% | 13.19 d |
| ProcessTransformer | 63.76% | 4.87 d | 74.97% | **4.34 d** |
| CNN | 67.46% | 4.76 d | **79.46%** | 4.68 d |
| GRU | 67.76% | 4.36 d | 78.08% | 4.37 d |
| RETAIN | **68.20%** | **4.08 d** | 78.15% | 4.44 d |

Full per-prefix-length breakdowns and raw predictions are in `experiments/results/`.

---

## LISA Vibe Coding

All code in `src/` and `experiments/run_experiments.py` was generated exclusively via LISA (`lisa-flash-03-2026`). No manual edits were made to source files. All corrections were applied through corrective prompts.

LISA chat exports (PDF + JSON) for each contributor are in `chat_exports/`.  
LISA model version: see `LISA_MODEL_VERSION.txt`.
