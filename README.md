# Multi-label Emotion Classification with RoBERTa

Fine-tuning RoBERTa-base on the GoEmotions dataset for multi-label emotion classification across 28 emotion categories.

## Overview

This project fine-tunes `roberta-base` to detect multiple emotions simultaneously from short English texts (Reddit comments). The model addresses class imbalance using Focal Loss and optimizes per-label thresholds dynamically.

## Dataset

[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) — A dataset of 58k Reddit comments labeled with 28 emotions (e.g., admiration, anger, joy, sadness, ...).

## Approach

- **Model:** RoBERTa-base fine-tuned for multi-label sequence classification
- **Loss Function:** Focal Loss (α=1, γ=2) to handle class imbalance
- **Threshold Optimization:** Dynamic per-label threshold tuning on validation set instead of fixed 0.5
- **Tokenizer:** Extended with special tokens `[NAME]`, `[RELIGION]` for Reddit-specific text
- **Experiment Tracking:** Weights & Biases (WandB)

## Results

Evaluated on the GoEmotions test set:

| Metric | Score |
|--------|-------|
| Macro F1 | 0.53 |
| Micro F1 | 0.58 |
| Weighted F1 | 0.59 |

Notable per-label results:

| Label | F1 | Support |
|-------|----|---------|
| gratitude | 0.90 | 337 |
| amusement | 0.81 | 289 |
| remorse | 0.73 | 67 |
| grief | 0.36 | 10 |
| relief | 0.34 | 18 |

![F1-score per Label](assets/roberta_f1_chart.png)

> Low F1 on labels like `grief` (10 samples) and `relief` (18 samples) is expected due to severe class imbalance.

## Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Requirements
```
torch
transformers
scikit-learn
pandas
numpy
wandb
emoji
iterative-stratification
```

## Usage
```bash
# Train
jupyter notebook fine-tune_RoBERTa-base.ipynb
```

## Project Structure
```
├── fine-tune_RoBERTa-base.ipynb   # Training & evaluation notebook
├── requirements.txt
└── README.md
```
