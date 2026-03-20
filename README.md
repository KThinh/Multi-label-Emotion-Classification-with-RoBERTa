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
![F1-score per Label](assets/roberta_f1_chart.png)
| Metric | Score |
|--------|-------|
| Macro F1 | 0.53 |
| Micro F1 | 0.58 |
| Weighted F1 | 0.59 |
