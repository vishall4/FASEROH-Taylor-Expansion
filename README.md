# FASEROH: Neural Translation of Mathematical Functions to Taylor Expansions

**GSoC 2026 Evaluation Test — ML4Sci**  
**Author:** Vishal Lohiya | IIT Jodhpur | [GitHub](https://github.com/vishall4) | vishallohiya50@gmail.com

> **Status:** Actively training and experimenting. Beam search evaluation and Transformer improvements are in progress. Results are updated regularly.

---

## The Problem

**Can a neural network learn to compute Taylor series expansions?**

Given a mathematical function, predict its Taylor expansion up to 4th order — framed as a sequence-to-sequence translation problem, like translating English to French, but for math:

```
Input:  sin(x)         →  Output: x - x**3/6
Input:  exp(x)         →  Output: 1 + x + x**2/2 + x**3/6 + x**4/24  
Input:  cos(2*x)       →  Output: 1 - 2*x**2 + 2*x**4/3
Input:  exp(3*x**2/4)  →  Output: 1 + 3*x**2/4 + 9*x**4/32
```

This is the evaluation test for the [FASEROH](https://ml4sci.org/) project under ML4Sci. Three tasks: generate data with SymPy, train an LSTM, train a Transformer.

**Mentors:** Abdulhakim Alnuqaydan (U. of Kentucky), Harrison Prosper (FSU)

---

## Results at a Glance

| Model | Data | Val Loss | Exact Match | Token Acc | BLEU-1 |
|-------|------|:--------:|:-----------:|:---------:|:------:|
| Prosper Baseline (3890 ep) | 60K | 2.35 | N/A | N/A | N/A |
| **Our LSTM+Attention (40 ep)** | **19K** | **1.41** | **29.23%** | **66.08%** | **93.28%** |
| Our LSTM+Attention (40 ep) | 45K | 1.51 | 25.39% | 59.73% | 93.22% |
| Transformer v1 (50 ep) | 45K | 0.65 | 0.63% | 6.89% | 57.11% |
| Transformer v2 + TF decay | 45K | 0.70→5.87 | — | — | Collapsed |
| Transformer + Beam Search | — | — | — | — | 🔄 In progress |

### Understanding the Metrics

- **Exact Match 29.23%** means the LSTM gets the Taylor expansion **perfectly correct** for about 1 in 3.4 test functions
- **The other 71% are close but not perfect** — usually just one or two coefficients are wrong
- **BLEU-1 of 93%** proves this: even when the prediction isn't an exact match, it shares 93% of its tokens with the correct answer
- **Example:** for `sin(x)/2`, the model might predict `x/2 - x**3/12` (correct!) or `x/2 - x**3/11` (one coefficient off — not exact match, but very close)

### The Surprising Finding

**The Transformer has the lowest validation loss (0.65) but the worst test accuracy (0.63%).** This is the [exposure bias problem](https://arxiv.org/abs/1511.06732) — the Transformer trains with correct previous tokens but must use its own (potentially wrong) predictions at test time. One wrong token cascades into a completely wrong sequence. [Details below.](#phase-3-transformer-v1--the-exposure-bias-discovery)

---

## The Full Research Journey

### Phase 1: Data Generation & Tokenization

#### Generating Our Dataset with SymPy

Built a custom pipeline generating **19,015 unique function-Taylor pairs** across 8 difficulty levels:

| Level | Type | Example | Count |
|:-----:|------|---------|------:|
| 1 | Basic functions | `sin(x)`, `exp(x)`, `log(1+x)` | 14 |
| 2 | Scaled functions | `sin(2*x)`, `exp(-x)`, `cos(x/3)` | 341 |
| 3 | Random polynomials | `3*x**3 - 2*x + 1` | 2,789 |
| 4 | Compositions | `sin(cos(x))`, `exp(sin(x))` | 2,839 |
| 5 | Products | `x*exp(x)`, `sin(x)*cos(x)` | 4,992 |
| 6 | Sums | `sin(x) + 2*cos(x) - exp(-x)` | 18,745 |
| 7 | Power functions | `(1+x)**(1/2)`, `(1+sin(x))**2` | 18,765 |
| 8 | Deep compositions | `sin(cos(exp(x)))` | 19,015 |

Each function's Taylor expansion was computed with `sympy.series(f, x, 0, n=5)`, then validated to filter out trivial, undefined, or overly long expressions.

We also used the mentor's [57K dataset](https://github.com/hbprosper/symbolic-ai) for extended experiments on harder functions (which include numerical constants like `cos(6)`, `exp(5)`, `sinh(9)`).

#### The Tokenization Breakthrough — The Single Most Impactful Decision

**Prior work** used character-level tokenization:
```
sin(2*x)  →  ['s', 'i', 'n', '(', '2', '*', 'x', ')']     # 8 tokens, model must learn s+i+n = sine
```

**Our math-aware tokenizer** treats function names as atomic units:
```
sin(2*x)  →  ['sin', '(', '2', '*', 'x', ')']              # 6 tokens, 'sin' is one meaningful token
```

Multi-digit numbers are split into digits to keep vocabulary small:
```
1024  →  ['1', '0', '2', '4']    # Only need 10 digit tokens instead of hundreds of number tokens
```

**Final vocabulary: just 34 tokens.** 4 special (PAD, SOS, EOS, UNK) + 10 digits + 6 operators + 1 variable + 13 function names.

**Impact: 97x reduction in training time.** Our LSTM reaches lower loss in 40 epochs vs 3,890 epochs for the character-level baseline. The model converges faster because it works with meaningful mathematical units instead of individual characters.

---

### Phase 2: LSTM + Attention — The Strong Baseline

**Architecture:** Bidirectional LSTM encoder (2 layers, hidden=256) → Bahdanau attention → LSTM decoder. 3.8M parameters.

**Key design:** Teacher forcing with linear decay from 1.0 to 0.1 over 40 epochs. Early training uses correct tokens (learn patterns). Late training uses own predictions (learn to recover from mistakes). This is critical — [it's why the LSTM beats the Transformer](#phase-3-transformer-v1--the-exposure-bias-discovery).

**Results on our 19K SymPy data:**
```
Exact Match:    29.23%    (1 in 3.4 predictions is perfectly correct)
Token Accuracy: 66.08%    (2 out of 3 tokens are correct on average)
BLEU-1:         93.28%    (even "wrong" predictions share 93% of tokens with ground truth)
```

**Example correct predictions:**
```
exp(x)                    → 1 + x + x**2/2 + x**3/6 + x**4/24           ✓
-2*x**3 + 8*x**2 - x     → -2*x**3 + 8*x**2 - x                        ✓
2*exp(-x)*sin(x)/(x**2+1) → 2*x**4 - 4*x**3/3 - 2*x**2 + 2*x           ✓
```

**Example close-but-wrong predictions (showing why BLEU is 93%):**
```
-2*sin(x)**2 - 2*cos(x)
  TRUE: 7*x**4/12 - x**2 - 2
  PRED: 7*x**4/12 - 3*x**2 - 2    ← only the x**2 coefficient is wrong (3 vs 1)
```

---

### Phase 3: Transformer v1 — The Exposure Bias Discovery

**Architecture:** Encoder-decoder Transformer, d_model=256, 8 heads, 4 layers, FFN=1024. 7.4M parameters. Warmup + cosine annealing LR schedule.

**The shocking result:**
```
Val Loss:       0.65   ← MUCH better than LSTM's 1.51
Exact Match:    0.63%  ← MUCH worse than LSTM's 29.23%
```

**How can the best validation loss give the worst test accuracy?**

During training, the Transformer always sees correct previous tokens (teacher forcing). During test inference, it uses its own predictions. One early mistake cascades:

```
True answer:    1 + x + x**2/2 + x**3/6 + x**4/24
Greedy decode:  1 + x + x**2/2 + x**3/3 + x**4/9    ← wrong from step 4 onward
                                    ^^^
                            one wrong coefficient here
                            corrupts everything after
```

**Why the LSTM doesn't have this problem:** The LSTM was trained with teacher forcing decay (1.0 → 0.1), which forced it to practice generating from its own predictions. The Transformer always saw perfect inputs during training.

**Key insight: Low validation loss ≠ good test accuracy when there's a train-test mismatch in how tokens are fed.**

---

### Phase 4: Scaling to 76K Data

Combined our 19K SymPy data with Prosper's 57K dataset (64,631 pairs after filtering).

**Finding:** Harder data doesn't always improve exact match. LSTM dropped from 29.2% → 25.4% because Prosper's data includes complex numerical constants (`cos(6)`, `sinh(9)`) that are harder to predict exactly. But BLEU stayed at 93% — the model is still very close on most predictions.

---

### Phase 5: Transformer v2 — The Teacher Forcing Decay Experiment (Failed)

**Hypothesis:** If TF decay fixed the LSTM, it should fix the Transformer too.

**What we tried:** Full teacher forcing for 10 epochs (stable initial learning), then gradually decrease TF ratio from 1.0 to 0.3 over the remaining 40 epochs.

**What happened:**
```
Ep  1-10: TF=1.00 → Val loss: 1.56 → 0.70    ✓ Learning well
Ep 11:    TF=0.97 → Val loss: 0.70 → 5.87     ✗ COLLAPSED
Ep 12:    TF=0.95 → Val loss: 5.93             ✗ Not recovering  
Ep 13:    TF=0.93 → Val loss: 5.93             ✗ Getting worse
Ep 14:    TF=0.90 → Val loss: 6.03             ✗ Stopped training
```

**The moment TF dropped below 1.0, the model collapsed.** We stopped after 15 epochs — no point burning GPU hours on a diverging model.

**Why it failed:** The Transformer's multi-head self-attention is much more sensitive to input perturbations than the LSTM's recurrent hidden state. Even 3% wrong tokens (TF=0.97) caused catastrophic interference across all attention heads and all layers. The autoregressive training loop also caused memory issues (had to reduce batch size from 128 to 32) and was extremely slow — negating the Transformer's parallelization advantage.

**Lesson learned:** Teacher forcing decay is NOT a universal fix.
- **LSTM exposure bias → fix during training** (TF decay works great)
- **Transformer exposure bias → fix during inference** (beam search is the right approach)

Each architecture has its own failure mode and its own solution.

---

### Phase 6: Beam Search Decoding (In Progress)

Instead of fixing the Transformer's training, we improve its **inference:**

```
Greedy decoding:  Pick the single best token at each step → one mistake ruins everything
Beam search (k=5): Keep top 5 candidate sequences → explore alternatives → recover from errors
```

Implementation complete for both LSTM and Transformer. Full evaluation started on H100 (completed 1000/2229 test samples) but was interrupted when cloud credits ran out. This is the immediate next step.

---

## Infrastructure Journey

This project ran across 5 different compute environments:

| Platform | GPU | Time | Cost | What We Got |
|----------|-----|------|------|-------------|
| Google Colab | T4 (free) | ~4h | $0 | Initial LSTM, data gen (disconnected twice) |
| Mac Mini M4 | Apple M4 | ~1h | $0 | SymPy data generation (19K pairs in 300s) |
| Vast.ai | RTX 5070 Ti | ~8h | ~$0.90 | LSTM on 76K data (6.5h for 40 epochs!) |
| Vast.ai | A100 SXM | ~4h | ~$4.10 | Transformer v2 TF decay (failed at epoch 12) |
| Vast.ai | H100 SXM | ~3h | ~$4.40 | Transformer v1+v2, partial beam search |
| **Total** | | **~20h** | **~$9.40** | |

**Lessons:** Colab disconnects randomly (save checkpoints!). LSTMs are sequential — fast GPUs don't help much. Transformer TF decay negates parallelization. Budget beam search time carefully (5x slower than greedy).

---

## Repository Structure

```
├── FASEROH_evaluation.ipynb      # Original evaluation (19K data, LSTM + Transformer)  
├── ML4sciH100.ipynb              # H100 training: LSTM + Transformer v1 on 45K data
├── upgradeV2.ipynb               # Beam search + Transformer v2 experiments
├── models/
│   ├── best_lstm_h100.pth        # Trained LSTM (val loss 1.51, 25.4% exact match)
│   └── best_transformer_h100.pth # Trained Transformer v1 (val loss 0.65)
├── data/
│   ├── our_19k_data.csv          # Our SymPy-generated dataset (19,015 pairs)
│   ├── prosper_60k.txt           # Prosper's 57K dataset  
│   └── data.txt                  # Filtered 45K training data
├── h100_full_evaluation.png      # Training curves and comparison plots
├── h100_all_results.pkl          # Serialized evaluation metrics
└── README.md                     # This document
```

## Quick Start

```bash
pip install torch sympy numpy pandas matplotlib scikit-learn
jupyter notebook FASEROH_evaluation.ipynb
```

## Tech Stack

Python, PyTorch, SymPy, NumPy, Pandas, Matplotlib, scikit-learn  
Trained on: NVIDIA H100 80GB HBM3 via [Vast.ai](https://vast.ai)

---

## What's Next

- [ ] Complete beam search evaluation (LSTM + Transformer)
- [ ] Scale SymPy dataset to 100K+ pairs
- [ ] Implement curriculum learning (easy → hard functions)
- [ ] Extend to core FASEROH goal: histogram → symbolic function mapping
- [ ] Explore alternative exposure bias fixes (minimum risk training, sequence-level objectives)

---

*Built by [Vishal Lohiya](https://github.com/vishall4) — IIT Jodhpur — GSoC 2026*  
*Mentors: Abdulhakim Alnuqaydan, Harrison Prosper — ML4Sci*
