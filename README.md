# FASEROH: Teaching Neural Networks to Compute Taylor Series

**GSoC 2026 Evaluation Test — ML4Sci (Machine Learning for Science)**  
**Author:** [Vishal Lohiya](https://github.com/vishall4) | IIT Jodhpur | vishallohiya50@gmail.com

> 🔄 **Active project.** Continuously improving models and experimenting with new techniques.

---

## What This Project Does

A neural network that **translates mathematical functions into their Taylor series expansions** — like Google Translate, but for math:

```
Input:  sin(x)          →  Output: x - x³/6
Input:  exp(x)          →  Output: 1 + x + x²/2 + x³/6 + x⁴/24
Input:  cos(2x)         →  Output: 1 - 2x² + 2x⁴/3
Input:  exp(3x²/4)      →  Output: 1 + 3x²/4 + 9x⁴/32
```

**Why it matters:** A neural network could provide instant symbolic approximations for complex functions — useful for physicists who need quick estimates during analysis, complementing slower computer algebra systems like SymPy.

This is the evaluation test for the [FASEROH](https://ml4sci.org/) project under ML4Sci. **Mentors:** Abdulhakim Alnuqaydan (U. of Kentucky), Harrison Prosper (FSU).

---

## Results

| Model | Decoding | Exact Match | Token Acc | BLEU-1 |
|-------|----------|:-----------:|:---------:|:------:|
| Prior Work¹ (3,890 epochs) | greedy | N/A | N/A | N/A |
| **LSTM+Attention (40 epochs)** | **greedy** | **30.19%** | **71.11%** | **94.65%** |
| **LSTM+Attention (40 epochs)** | **beam-5** | **30.64%** | **71.95%** | **95.31%** |
| Transformer (50 epochs) | greedy | 0.67% | 8.03% | 58.77% |
| Transformer (50 epochs) | beam-5 | 0.36% | 8.23% | 56.99% |
| Transformer + TF Decay | — | COLLAPSED | — | — |

¹ [Prosper's symbolic-ai baseline](https://github.com/hbprosper/symbolic-ai) using character-level tokenization, val loss 2.35

### What These Numbers Mean

**Exact Match 30.19%** = the LSTM predicts the perfectly correct Taylor expansion for nearly **1 in 3** test functions.

**The other 70% are almost right.** BLEU-1 of 94.65% means even "wrong" predictions share ~95% of tokens with the correct answer — typically just one coefficient is off:

```
Function: -2*sin(x)² - 2*cos(x)
  Correct:    7x⁴/12  -  x²  - 2
  Predicted:  7x⁴/12  - 3x²  - 2     ← only one coefficient wrong
```

**The Transformer paradox:** lowest val loss (0.64) but worst test accuracy (0.67%). This is **exposure bias** — see the [full explanation below](#phase-3-the-transformer--discovering-exposure-bias).

---

## The Research Journey

### Phase 1: Data Generation & The Tokenization Breakthrough

#### Our Dataset
Generated **18,812 unique function-Taylor pairs** using SymPy across 8 difficulty levels (basic → deep compositions). Also used the mentor's [57K dataset](https://github.com/hbprosper/symbolic-ai) for training. Combined: **45,023 pairs** after filtering.

#### The Key Insight: Math-Aware Tokenization

Prior work tokenized character-by-character (`sin` → `['s','i','n']`). Our tokenizer treats function names as single tokens (`sin` → `['sin']`), with multi-digit numbers split into digits. **Result: 34-token vocabulary, 97x faster convergence** (40 epochs vs 3,890).

```
Prior work:  sin(2*x) → ['s','i','n','(','2','*','x',')']     8 tokens, meaning scattered
Our method:  sin(2*x) → ['sin','(','2','*','x',')']            6 tokens, meaning preserved
```

---

### Phase 2: LSTM + Attention

**Architecture:** BiLSTM encoder (2 layers, 256 hidden) → Bahdanau attention → LSTM decoder. 3.8M parameters.

**Critical design:** Teacher forcing decay from 1.0 → 0.1 over 40 epochs. The model gradually learns to generate from its own (possibly wrong) predictions instead of always seeing correct tokens. This is why it generalizes well at test time.

**Results (greedy / beam-5):** 30.19% / 30.64% exact match, 94.65% / 95.31% BLEU-1.

Beam search fixed 23 predictions that greedy got wrong — for example correcting a sign error from `+20*x²` to `-20*x²`.

**Trained on:** Google Colab T4 GPU (~3 hours for 40 epochs).

---

### Phase 3: The Transformer — Discovering Exposure Bias

**Architecture:** Encoder-decoder Transformer, 256d, 8 heads, 4 layers, 7.4M params. Warmup + cosine LR.

**The shocking result:**
```
Transformer val loss:    0.64    ← Much BETTER than LSTM's 1.51
Transformer exact match: 0.67%  ← Much WORSE than LSTM's 30.19%
```

**Why?** During training, the Transformer always sees correct previous tokens. During inference, it uses its own predictions. One mistake cascades:

```
True:       1 + x + x²/2 + x³/6  + x⁴/24
Predicted:  1 + x + x²/2 + x³/3  + x⁴/9    ← wrong from step 4 onward
```

**The LSTM doesn't have this problem** because teacher forcing decay (1.0 → 0.1) trained it to handle its own mistakes. The Transformer was always given perfect inputs.

**Beam search didn't help the Transformer** — it actually made it slightly worse (0.67% → 0.36%). The exposure bias is too severe for inference-time fixes alone.

**Trained on:** Google Colab T4 GPU (~3 hours for 50 epochs).

---

### Phase 4: The Experiment That Failed — Transformer + TF Decay

**Hypothesis:** If TF decay fixed the LSTM, it should fix the Transformer too.

**What happened:**
```
Epochs 1-11:  TF=1.0  → Val loss: 1.20 → 0.70    ✓ Learning well
Epoch 12:     TF=0.97 → Val loss: 0.70 → 5.93     ✗ INSTANT COLLAPSE
```

**The moment TF dropped below 1.0, the model collapsed.** Even 3% of tokens being wrong (TF=0.97) was catastrophic.

**Why:** The Transformer's multi-head attention amplifies errors — one wrong token is attended to by every position across every head across every layer, polluting the entire representation. The LSTM's recurrent hidden state absorbs perturbations locally.

**Lesson: TF decay is NOT a universal fix.**
- ✅ Works for LSTMs — hidden state absorbs errors gradually
- ❌ Breaks Transformers — attention amplifies errors catastrophically

**Trained on:** Vast.ai H200 NVL GPU (~2 hours for Notebook 2 including beam search evaluation).

---

## Infrastructure Story

| Platform | GPU | Time | Cost | What We Did |
|----------|-----|------|------|-------------|
| Google Colab | T4 (free) | ~3h | $0 | Notebook 1: LSTM + Transformer training & evaluation |
| Vast.ai | H200 NVL | ~2h | ~$3 | Notebook 2: Beam search + TF decay experiment |
| Mac Mini M4 | Apple M4 | ~5min | $0 | SymPy data generation (18,812 pairs) |
| **Total** | | **~5h** | **~$3** | |

---

## Repository Structure

| File | Description |
|------|-------------|
| `01_LSTM_vs_Transformer.ipynb` | **Notebook 1:** Data generation, LSTM training, Transformer training, greedy evaluation, exposure bias analysis |
| `02_Beam_Search_and_Experiments.ipynb` | **Notebook 2:** Beam search decoding, greedy vs beam comparison, TF decay experiment (shows collapse), final results |
| `models/best_lstm.pth` | Trained LSTM weights (3.8M params, val loss 1.51) |
| `models/best_transformer.pth` | Trained Transformer weights (7.4M params, val loss 0.64) |
| `data/our_19k_data.csv` | Our SymPy-generated dataset (18,812 pairs, 8 difficulty levels) |
| `data/prosper_60k.txt` | Mentor's 57K dataset |
| `README.md` | This document |

## Quick Start

```bash
pip install torch sympy numpy pandas matplotlib scikit-learn

# Run Notebook 1 first (trains both models)
jupyter notebook 01_LSTM_vs_Transformer.ipynb

# Then Notebook 2 (beam search + experiments)
jupyter notebook 02_Beam_Search_and_Experiments.ipynb
```

## What's Next

- [ ] Scale SymPy dataset to 100K+ pairs
- [ ] Implement curriculum learning (easy → hard functions)
- [ ] Explore alternative exposure bias fixes (minimum risk training, RL-based methods)
- [ ] Accelerate SymPy data generation (currently CPU-bound)
- [ ] Extend to core FASEROH goal: histogram → symbolic function mapping

---

**Tech Stack:** Python · PyTorch · SymPy · NumPy · Pandas · Matplotlib  
**Trained on:** Google Colab T4, Vast.ai H200 NVL  
**Mentors:** Abdulhakim Alnuqaydan · Harrison Prosper · [ML4Sci](https://ml4sci.org/)

*Built by [Vishal Lohiya](https://github.com/vishall4) — IIT Jodhpur — GSoC 2026*
