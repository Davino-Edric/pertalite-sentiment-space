---
title: Pertalite Sentiment Analysis
emoji: ⛽
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Indonesian TikTok sentiment analysis on fuel subsidy policy
---

# 🛢️ Pertalite Sentiment Analysis

> Classifying Indonesian public opinion on fuel subsidy policy from TikTok comments using Machine Learning.

[![Research Repo](https://img.shields.io/badge/GitHub-Research%20Repo-181717?logo=github)](https://github.com/Davino-Edric/pertalite-sentiment-analysis)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-FFD21E)](https://huggingface.co/spaces/vino-edric/pertalite-sentiment)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Gradio-F97316)](https://gradio.app)

---

## 📌 What This App Does

This app predicts whether an Indonesian-language TikTok comment about **Pertalite** (Indonesia's subsidized fuel) or **Minister Bahlil Lahadalia** carries a **positive** or **negative** sentiment.

Type or paste any comment in the text box and the model returns a confidence score for each class.

---

## 🧠 How It Works

This project uses a **Model Distillation** approach — a two-stage pipeline that gets the best of both worlds between transformer accuracy and classical ML efficiency:

```
Stage 1 — Labeling (offline, training only)
  Raw TikTok comments
      ↓
  w11wo/indonesian-roberta-base-sentiment-classifier
      ↓
  Auto-generated sentiment labels + confidence scores
  (low-confidence samples discarded at threshold < 0.65)

Stage 2 — Serving (what this app runs)
  User input
      ↓
  Text cleaning + Indonesian slang normalization
      ↓
  TF-IDF vectorizer (n-gram 1-2, 5000 features)
      ↓
  Random Forest classifier (trained with SMOTE)
      ↓
  Sentiment prediction + confidence score
```

The deployed model is lightweight — **no neural networks at inference time**. The full RoBERTa model was only used during training to generate labels.

---

## 📊 Model Performance

The final model is a **Random Forest + SMOTE** classifier, chosen for its balanced performance on an imbalanced dataset (~88% negative, ~12% positive).

| Metric | Score |
|:---|:---:|
| Accuracy | ~85% |
| Macro F1 | ~0.70 |
| Positive class F1 | 0.48 |
| Negative class F1 | 0.91 |

> **Why Macro F1 matters here:** Accuracy alone is misleading on imbalanced data. A model that always predicts "negative" would score 88% accuracy while being completely useless. Macro F1 treats both classes equally, making it the honest metric for this dataset.

---

## ⚙️ Tech Stack

| Layer | Technology |
|:---|:---|
| Labeling | `transformers` — `w11wo/indonesian-roberta-base-sentiment-classifier` |
| Feature extraction | `scikit-learn` — TF-IDF |
| Classification | `scikit-learn` — Random Forest |
| Imbalance handling | `imbalanced-learn` — SMOTE |
| Web app | `Gradio` |
| Containerization | `Docker` |
| Deployment | HuggingFace Spaces |

---

## ⚠️ Known Limitations

- **Sarcasm and implicit negativity** — The model struggles with culturally implicit insults or ironic phrasing common in Indonesian social media. TF-IDF has no understanding of sentence structure or context.
- **Domain specificity** — Trained exclusively on Pertalite/Bahlil-related comments. Performance on other topics will degrade.
- **Class imbalance** — Despite SMOTE, the positive class F1 (0.48) reflects the difficulty of learning from a heavily skewed dataset with only ~12% positive samples.

---

## 🗂️ Project Structure

The full research pipeline (scraping → EDA → training) lives in the companion repository:

📎 [**pertalite-sentiment-analysis**](https://github.com/Davino-Edric/pertalite-sentiment-analysis) — Jupyter notebook, raw dataset, preprocessing, model training, and evaluation.

This Space contains only the inference and deployment code:

```
pertalite-sentiment-space/
├── app.py            ← Gradio app + inference pipeline
├── model.pkl         ← Trained RF + TF-IDF vectorizer + slang dictionary
├── requirements.txt  ← Deployment dependencies only
└── Dockerfile        ← Container definition
```

---

## 👤 Author

**Davino Edric** — Data Science student at PENS (Politeknik Elektronika Negeri Surabaya)

[![GitHub](https://img.shields.io/badge/GitHub-Davino--Edric-181717?logo=github)](https://github.com/Davino-Edric)
pinned: false
license: mit
