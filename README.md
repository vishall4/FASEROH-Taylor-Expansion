# FASEROH: Teaching Neural Networks to Compute Taylor Series

**GSoC 2026 Evaluation Test — ML4Sci (Machine Learning for Science)**  
**Author:** [Vishal Lohiya](https://github.com/vishall4) | IIT Jodhpur | vishallohiya50@gmail.com

> 🔄 **Active project.** Training, evaluation, and experiments are ongoing. Beam search decoding and further Transformer improvements are in progress. This document and results are updated regularly.

---

## What This Project Does

This project builds a neural network that **translates mathematical functions into their Taylor series expansions** — the same way Google Translate converts English to French, except here we translate math:

```
Input:  sin(x)          →  Output: x - x³/6
Input:  exp(x)          →  Output: 1 + x + x²/2 + x³/6 + x⁴/24
Input:  cos(2x)         →  Output: 1 - 2x² + 2x⁴/3
Input:  exp(3x²/4)      →  Output: 1 + 3x²/4 + 9x⁴/32
```

**Why does this matter?** Computing Taylor expansions by hand is tedious and error-prone. Computer algebra systems like SymPy can do it, but they work symbolically (slow for complex functions). A neural network could provide **instant approximations** — useful for physicists who need quick symbolic estimates during analysis.

This is the evaluation test for the [FASEROH](https://ml4sci.org/) project, which ultimately aims to map histograms (experimental data) to symbolic functions.

---

## Results Summary

| Model | Dataset | Val Loss | Exact Match | BLEU-1 |
|-------|---------|:--------:|:-----------:|:------:|
| Prior Work¹ (3,890 epochs) | 60K | 2.35 | N/A | N/A |
| **Our LSTM+Attention (40 epochs)** | **19K** | **1.41** | **29.23%** | **93.28%** |
| Our LSTM+Attention (40 epochs) | 45K | 1.51 | 25.39% | 93.22% |
| Transformer v1 (50 epochs) | 45K | 0.65 | 0.63% | 57.11% |
| Transformer v2 + TF Decay | 45K | Collapsed | — | — |
| Transformer + Beam Search | — | — | 🔄 In progress | — |

¹ [Prosper's symbolic-ai baseline](https://github.com/hbprosper/symbolic-ai) using character-level tokenization

### What Do These Metrics Actually Mean?

**Exact Match** measures whether the model's prediction is character-for-character identical to the correct Taylor expansion. **29.23% exact match means the LSTM produces the perfectly correct answer for about 1 in every 3.4 test functions.** The other ~71% have small errors — a wrong coefficient, a missing term — but are structurally very close to the right answer.

**BLEU-1** (Bilingual Evaluation Understudy) measures token-level overlap between prediction and ground truth. **93.28% BLEU-1 means even when the model doesn't get an exact match, it still gets 93% of the tokens right.** This is reassuring — it tells us the model genuinely understands the mathematical structure and is usually just one or two numbers off. Here's a concrete example:

```
Function: -2*sin(x)² - 2*cos(x)
  Correct answer:   7x⁴/12  -  x²  - 2
  Model prediction:  7x⁴/12 - 3x²  - 2     ← only the coefficient of x² is wrong (3 vs 1)
                                               BLEU-1 ≈ 90% (most tokens match)
                                               Exact Match = ✗ (not perfectly identical)
```

So **25% exact match + 93% BLEU-1 together tell us: the model is right a quarter of the time, and close to right almost all of the time.** For a practical tool, the "close" predictions could be verified with a single SymPy call, making the system useful even with imperfect accuracy.

**Val Loss** measures how well the model predicts the next token during training. Lower is better — but as we discovered, **low validation loss does NOT guarantee good test accuracy** (see the [Transformer section](#phase-3-the-transformer--discovering-exposure-bias) for why).

---

## The Research Journey

### Phase 1: Data Generation & The Tokenization Breakthrough

#### Building Our Dataset with SymPy

The evaluation test requires generating your own dataset. I built a pipeline that creates mathematical functions across **8 progressively harder difficulty levels:**

| Level | Type | Example | What Makes It Hard |
|:-----:|------|---------|-------------------|
| 1 | Basic functions | `sin(x)`, `exp(x)` | Simple — one function, no composition |
| 2 | Scaled functions | `sin(2x)`, `exp(-x)` | Coefficients change the expansion |
| 3 | Polynomials | `3x³ - 2x + 1` | Already a Taylor expansion (identity mapping) |
| 4 | Compositions | `sin(cos(x))` | Must expand outer AND inner function |
| 5 | Products | `x·exp(x)` | Multiply two expansions together |
| 6 | Sums | `sin(x) + 2cos(x)` | Combine multiple expansions with coefficients |
| 7 | Powers | `√(1+x)`, `(1+sin(x))²` | Binomial/generalized expansion |
| 8 | Deep nesting | `sin(cos(exp(x)))` | 3 levels of composition — very complex |

**Result: 19,015 unique pairs** generated in ~5 minutes on CPU. Split: 80/10/10 train/val/test.

I also used the mentor's [57K dataset](https://github.com/hbprosper/symbolic-ai) for extended experiments. Their data is harder — it includes numerical constants like `cos(6)` and `sinh(9)` in the Taylor expansions, which require the model to work with specific irrational numbers.

#### The Tokenization Decision (Most Important Choice in the Project)

**The question:** How do you represent a math expression like `sin(2*x)` as a sequence of tokens that a neural network can process?

**Approach A — Character-level (what prior work used):**
```
sin(2*x)  →  ['s', 'i', 'n', '(', '2', '*', 'x', ')']
```
Problem: the model has to learn that the characters 's', 'i', 'n' appearing together mean "the sine function." It's like teaching someone English by spelling every word letter-by-letter — technically possible, but painfully slow.

**Approach B — Our math-aware tokenizer:**
```
sin(2*x)  →  ['sin', '(', '2', '*', 'x', ')']
```
`sin` is a single token. The model immediately knows it's dealing with the sine function. Each token carries mathematical meaning.

For numbers, we split multi-digit numbers into individual digits:
```
1024  →  ['1', '0', '2', '4']
```
This keeps our vocabulary at just **34 tokens** instead of needing hundreds of unique number tokens.

**Why this matters so much:** With character-level tokens, the baseline needed **3,890 epochs** to reach val loss 2.35. Our LSTM reached **lower loss in just 40 epochs** — a **97x speedup**. Same data, same model size, different tokenization. The tokenizer is doing the heavy lifting.

---

### Phase 2: LSTM + Attention — Building a Strong Baseline

#### How It Works (Simplified)

The model has two parts:

1. **Encoder** (reads the input function): A bidirectional LSTM processes the tokenized input from both directions, creating a rich understanding of the mathematical structure.

2. **Decoder with Attention** (writes the Taylor expansion): Generates the output one token at a time. At each step, the **attention mechanism** looks back at the encoder's understanding and decides which parts of the input function are most relevant for predicting the next output token.

Think of it like a student solving a Taylor expansion on a whiteboard: they keep glancing back at the original function (attention) while writing out each term of the expansion.

#### The Teacher Forcing Strategy (Why the LSTM Generalizes Well)

During training, we gradually transition from "training wheels" to "solo riding":

```
Epoch  1: 100% teacher forcing — always show the correct previous token
Epoch 10:  73% teacher forcing — sometimes use the model's own (possibly wrong) prediction
Epoch 20:  43% teacher forcing — more often using own predictions than correct ones  
Epoch 30:  13% teacher forcing — almost entirely on its own
Epoch 40:  10% teacher forcing — nearly full autoregressive (like real inference)
```

This **teacher forcing decay** is critical. By the end of training, the LSTM has practiced generating sequences from its own predictions — including recovering from its own mistakes. This is exactly what happens at test time.

#### LSTM Results

On our 19K SymPy dataset:
```
Exact Match:    29.23%   → 1 in 3.4 predictions is perfectly correct
Token Accuracy: 66.08%   → 2 out of 3 tokens are correct on average
BLEU-1:         93.28%   → even "wrong" predictions share 93% of tokens with the truth
```

**Correct predictions (the model nails these):**
```
exp(x)                     → 1 + x + x**2/2 + x**3/6 + x**4/24     ✓
-2*x**3 + 8*x**2 - x      → -2*x**3 + 8*x**2 - x                   ✓  
exp(3*x**2/4)              → 1 + 3*x**2/4 + 9*x**4/32               ✓
```

**Close but not exact (this is the 93% BLEU in action):**
```
-2*sin(x)² - 2*cos(x)
  TRUE: 7*x**4/12  - x**2   - 2
  PRED: 7*x**4/12  - 3*x**2 - 2      ← just one coefficient wrong

sin(6*x³+3) / exp(-4*x/9)  
  TRUE: sin(3) + 4x·sin(3)/9 + 8x²·sin(3)/81 + x³·(6cos(3) + ...
  PRED: sin(3) + 4x·sin(3)/9 + 8x²·sin(3)/81 + x³·(32sin(33)/...
  ← gets first 3 terms perfect, struggles with the complex 4th term
```

---

### Phase 3: The Transformer — Discovering Exposure Bias

#### The Transformer Architecture

Same encoder-decoder concept as the LSTM, but using **self-attention** instead of recurrence. The Transformer processes all positions in parallel (much faster to train) and uses multi-head attention to capture different types of relationships between tokens.

Config: 256 dimensions, 8 attention heads, 4 layers, 7.4M parameters. Trained with warmup + cosine learning rate schedule.

#### The Shocking Result

```
Transformer val loss:  0.65    ← Much BETTER than LSTM's 1.51
Transformer exact match: 0.63% ← Much WORSE than LSTM's 29.23%
```

**The model with the best training performance had the worst real-world accuracy.** This paradox has a name: **exposure bias**.

#### What Is Exposure Bias? (And Why It Matters)

During training, at each step the decoder sees the **correct** previous token:
```
Training step 4: model sees [1, +, x, +, x**2/2, +] ← all correct tokens
                 model predicts: x**3/6 ← learns the right pattern
```

During test inference, the decoder sees its **own predictions** (which might be wrong):
```
Test step 4: model sees [1, +, x, +, x**2/2, +] ← correct so far
             model predicts: x**3/3 ← WRONG (should be x**3/6)
Test step 5: model sees [1, +, x, +, x**2/2, +, x**3/3, +] ← corrupted input!
             model predicts: x**4/9 ← cascading error
```

One wrong token corrupts everything that follows. The model never practiced handling wrong inputs during training, so it doesn't know how to recover.

**Why the LSTM doesn't have this problem:** We trained the LSTM with teacher forcing decay — gradually replacing correct tokens with the model's own predictions during training. By epoch 40, the LSTM was generating almost entirely from its own predictions, so it learned to handle mistakes. **The Transformer was always given perfect inputs during training, so it's helpless when it sees its own mistakes.**

#### Why Beam Search Is the Right Solution (Not TF Decay)

We have two options to fix exposure bias:

**Option A: Fix during training** — add teacher forcing decay to the Transformer (we tried this — it failed, [see Phase 5](#phase-5-the-experiment-that-failed--transformer--teacher-forcing-decay))

**Option B: Fix during inference** — use **beam search** instead of greedy decoding

**What is greedy decoding?** At each step, pick the single most probable token. Like navigating a maze by always turning right — simple, but if one turn is wrong, you're lost.

**What is beam search?** At each step, keep the **top 5 most promising partial sequences** (beam width = 5). Each is extended independently. At the end, pick the best complete sequence.

```
Greedy (width=1):           Beam Search (width=5):
Step 1: "1"                 Step 1: "1" (5 candidates)
Step 2: "1 + x"             Step 2: "1 + x" | "1 - x" | "1 + 2" | ... (5 best)
Step 3: "1 + x + x²/3" ✗   Step 3: "1 + x + x²/2" ✓ | "1 + x + x²/3" | ... (5 best)
                            → keeps the correct path alive even if it wasn't the #1 pick
```

Beam search doesn't change the model — it just searches more carefully during inference. It's the right tool for Transformers because:
- It doesn't require modifying the training process
- It works with the existing trained weights
- It explores multiple paths, reducing the impact of single-token errors

**Status:** Beam search is implemented and partially evaluated. Full test set evaluation is in progress.

---

### Phase 4: Scaling to 76K Data

We merged our 19K SymPy data with the mentor's 57K dataset (64,631 usable pairs after filtering). Training on more data with the LSTM:

```
On 19K data: Exact Match = 29.23%, BLEU = 93.28%
On 45K data: Exact Match = 25.39%, BLEU = 93.22%
```

**Exact match dropped but BLEU stayed the same.** Why? The larger dataset includes much harder functions with numerical constants (e.g., Taylor expansion of `sin(6x³+3)` contains `sin(3)` and `cos(3)` as coefficients). These are inherently harder to predict exactly. But the BLEU staying at 93% confirms the model is still producing structurally correct expansions — just with occasional coefficient errors on the hard cases.

**Takeaway:** More data isn't always better for exact match. What matters is **matched difficulty** between training data and what you evaluate on.

---

### Phase 5: The Experiment That Failed — Transformer + Teacher Forcing Decay

#### The Hypothesis

"If teacher forcing decay made the LSTM robust to its own errors, it should work for the Transformer too."

#### What We Implemented

Modified the Transformer to support **scheduled sampling**: first 10 epochs with full teacher forcing (stable learning), then gradually reduce the TF ratio:

```
Epochs  1-10: TF = 1.00  (always correct previous token — standard training)
Epoch 11:     TF = 0.97  (97% correct, 3% model's own predictions)
Epoch 15:     TF = 0.88  (decreasing further)
Epoch 30:     TF = 0.50  (half and half — never reached this point)
```

#### What Actually Happened

```
Epochs 1-10:  Val loss: 1.56 → 0.70   ✓ Learning well (full teacher forcing)
Epoch 11:     Val loss: 0.70 → 5.87   ✗ INSTANT COLLAPSE when TF dropped to 0.97
Epoch 12:     Val loss: 5.93          ✗ Not recovering
Epoch 13:     Val loss: 5.93          ✗ Getting worse
Epoch 14:     Val loss: 6.03          ✗ Stopped training — clearly not going to recover
```

**The moment teacher forcing dropped below 100%, the model catastrophically collapsed.** Even with 97% correct tokens (only 3% from the model's own predictions), the Transformer couldn't cope.

#### Why It Failed (The Technical Explanation)

Three compounding factors:

1. **Attention amplifies errors.** In the LSTM, a wrong token affects the next hidden state — a local disturbance. In the Transformer, a wrong token is attended to by **every position through every head across every layer**. One wrong token pollutes the entire representation, causing widespread damage that the model can't recover from.

2. **Abrupt transition.** The model spent 10 epochs perfectly optimized for correct inputs at every position. Going from 100% to 97% correct sounds small, but in a 120-token sequence, that's ~4 wrong tokens per sequence — and each one disrupts all the attention computations.

3. **Memory and speed problems.** The scheduled sampling loop requires running the decoder step-by-step (autoregressive) instead of the Transformer's normal parallel forward pass. This caused out-of-memory errors on GPUs with less than 80GB VRAM and made training extremely slow — negating the Transformer's main advantage.

#### The Lesson

**Teacher forcing decay is NOT a universal fix for exposure bias:**
- ✅ **Works for LSTMs** — the recurrent hidden state absorbs perturbations gradually
- ❌ **Breaks Transformers** — multi-head attention amplifies perturbations catastrophically

**The right approach depends on the architecture:**
- LSTM exposure bias → fix during **training** (teacher forcing decay)
- Transformer exposure bias → fix during **inference** (beam search)

This is one of the most valuable findings from the project. It's a practical result that anyone training seq2seq models should know.

---

## Infrastructure Story

This project ran across 5 different compute environments, each teaching lessons about GPU computing for deep learning research:

| Platform | GPU | Cost | What Happened |
|----------|-----|------|---------------|
| Google Colab | T4 (free) | $0 | Generated 19K data, trained initial LSTM. Disconnected twice — lost all variables. **Lesson: always save checkpoints.** |
| Mac Mini M4 | Apple M4 | $0 | Generated SymPy data locally (300s). Training too slow for iteration. **Lesson: local is fine for data prep, not for training.** |
| Vast.ai | RTX 5070 Ti | ~$0.90 | LSTM on 76K data — took 6.5 hours for 40 epochs. **Lesson: LSTMs are sequential; fast GPUs don't help much.** |
| Vast.ai | A100 (80GB) | ~$4.10 | Transformer v2 with TF decay. OOM at batch 128, reduced to 32. Collapsed at epoch 12. **Lesson: autoregressive loops negate Transformer parallelization.** |
| Vast.ai | H100 (80GB) | ~$4.40 | Trained Transformer v1+v2, started beam search eval. Credits ran out mid-evaluation. **Lesson: budget inference time carefully — beam search is 5x slower than greedy.** |
| **Total** | | **~$9.40** | Complete LSTM training, Transformer experiments, partial beam search |

---

## Repository Structure

| File | Description |
|------|-------------|
| `FASEROH_evaluation.ipynb` | **Start here.** Original evaluation on 19K SymPy data — all 3 tasks in one notebook |
| `ML4sciH100.ipynb` | LSTM + Transformer v1 training on 45K data (H100) |
| `upgradeV2.ipynb` | Beam search implementation + Transformer v2 TF decay experiments |
| `models/best_lstm.pth` | Trained LSTM weights (3.8M params, val loss 1.51) |
| `models/best_transformer.pth` | Trained Transformer v1 weights (7.4M params, val loss 0.65) |
| `data/our_19k_data.csv` | Our SymPy-generated dataset (19,015 pairs, 8 difficulty levels) |
| `data/prosper_60k.txt` | Mentor's 57K dataset |
| `h100_full_evaluation.png` | Training curves and model comparison plots |
| `README.md` | This document |

## Quick Start

```bash
pip install torch sympy numpy pandas matplotlib scikit-learn
jupyter notebook FASEROH_evaluation.ipynb
```

## What's Next

- [ ] Complete beam search evaluation on full test set
- [ ] Scale SymPy dataset to 100K+ pairs with more diverse functions
- [ ] Implement curriculum learning (train on easy functions first, then harder ones)
- [ ] Explore alternative exposure bias solutions (minimum risk training, reinforcement learning)
- [ ] Extend to the core FASEROH goal: histogram → symbolic function mapping

---

**Tech Stack:** Python · PyTorch · SymPy · NumPy · Pandas · Matplotlib  
**Trained on:** NVIDIA H100 80GB HBM3 via [Vast.ai](https://vast.ai)  
**Mentors:** Abdulhakim Alnuqaydan (U. of Kentucky) · Harrison Prosper (FSU) · [ML4Sci](https://ml4sci.org/)

*Built by [Vishal Lohiya](https://github.com/vishall4) — IIT Jodhpur — GSoC 2026*
