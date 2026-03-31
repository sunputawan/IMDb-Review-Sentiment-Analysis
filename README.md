# IMDB Sentiment Analysis

NLP project comparing three approaches for binary sentiment classification (positive/negative) on 50,000 IMDB movie reviews, ranging from a simple TF-IDF baseline to a fine-tuned BERT transformer.

## Overview

Sentiment analysis is one of the most widely applied NLP tasks in industry — from product reviews to customer feedback. This project benchmarks three approaches of increasing complexity, evaluating not only accuracy but also real-world linguistic robustness on challenging edge cases.

## Dataset

**Source**: [IMDB Dataset of 50K Movie Reviews - Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

| Property | Value |
|----------|-------|
| Size | 50,000 reviews |
| Classes | Positive / Negative |
| Balance | 25,000 each (perfectly balanced) |
| Language | English |


## Workflow

### 1. EDA
- Reviews contain HTML tags (`<br />`) → must be cleaned
- Average review length: ~231 words, 95th percentile: 590 words → max_length=500 chosen for padding
- Top words dominated by stopwords → must be removed
- No difference in length between positive and negative reviews → length alone cannot predict sentiment

### 2. Preprocessing
Applied uniformly across all 3 models from a single train/test split (80/20) to ensure fair comparison:

1. Replace HTML tags with space (not delete) to prevent word merging
2. Remove special characters → letters only
3. Lowercase all text
4. Remove stopwords via NLTK
5. Encode labels → positive=1, negative=0

### 3. Models

#### Model 1 — MLP + TF-IDF (Baseline)
- TF-IDF vectorizes each review into a 10,000-dimensional vector based on word importance
- MLP (128 → 64 → 1) classifies based on word importance scores
- **Limitation**: Bag-of-words approach — word order ignored ("not good" = "good not")

#### Model 2 — GRU + Word Embedding
- Trainable Embedding layer (10,000 vocab, 128 dimensions) learns word representations from scratch
- GRU reads sequence left-to-right, maintaining memory of previous words
- **Key fix**: `padding='pre'` ensures zeros appear before actual text, so GRU's final hidden state contains review content — not padding zeros

#### Model 3 — Fine-tuned BERT
- `bert-base-uncased` pretrained on Wikipedia and BooksCorpus
- Classification head added on top of `[CLS]` token representation
- Fine-tuned for 3 epochs with lr=2e-5 using AdamW optimizer and linear scheduler
- **Why BERT**: Bidirectional context, pretrained language understanding, attention mechanism

## Results

### Quantitative Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| MLP + TF-IDF | 0.8886 | 0.8891 |
| GRU + Embedding | 0.8821 | 0.8854 |
| **BERT (fine-tuned)** | **0.8991** | **0.9004** |

### Qualitative Evaluation — Linguistic Edge Cases

| Case | MLP | GRU | BERT |
|------|-----|-----|------|
| Sarcasm | ❌ | ❌ | ❌ |
| Double negative | ❌ | ✅ | ✅ |
| Mixed → ends positive | ❌ | ❌ | ✅ |
| Mixed → ends negative | ❌ | ❌ | ✅ |
| Subtle dissatisfaction | ❌ | ❌ | ❌ |
| Complex negation | ✅ | ✅ | ✅ |
| Negation | ✅ | ✅ | ✅ |
| Subtle negative | ❌ | ❌ | ✅ |
| **Score** | **2/8** | **3/8** | **6/8** |

### Performance vs Compute Trade-off

| Model | Training Time | Accuracy |
|-------|--------------|----------|
| MLP + TF-IDF | ~16 sec | 88.9% |
| GRU + Embedding | ~30 sec | 88.2% |
| BERT (fine-tuned) | ~50 min | 90.0% |

## Key Findings

- **MLP outperforms GRU** despite being simpler — sentiment relies more on keyword presence than word order, making TF-IDF surprisingly effective
- **BERT achieves the best results** by leveraging bidirectional context and pretrained language knowledge
- **All models fail on sarcasm** — requires world knowledge and pragmatic understanding beyond what can be learned from text alone
- **Higher accuracy ≠ language understanding** — BERT's 6/8 edge case score vs 2-3/8 for simpler models demonstrates the gap between benchmark performance and real-world robustness
