# FASEROH: Fast Accurate Symbolic Empirical Representation Of Histograms

**GSoC 2026 Evaluation Test — ML4Sci**

Seq2seq models that translate mathematical functions into their Taylor series expansions, framed as a neural machine translation problem.

## Problem

Given a mathematical function like `sin(x)`, predict its Taylor expansion up to 4th order:

```
sin(x)      →  x - x**3/6
exp(x)      →  1 + x + x**2/2 + x**3/6 + x**4/24
cos(2*x)    →  1 - 2*x**2 + 2*x**4/3
x**2*exp(x) →  x**4/2 + x**3 + x**2
```

## Approach

### Data Generation (Task 1)
- **76K+ unique function–Taylor pairs** from two sources:
  - 57K pairs from [Prosper's symbolic-ai dataset](https://github.com/hbprosper/symbolic-ai)
  - 19K pairs generated with SymPy across 8 difficulty levels (basic → deep compositions)
- **Math-aware tokenizer** that treats `sin`, `cos`, `exp`, `**` as single tokens (34-token vocabulary) instead of character-level tokenization (~50+ tokens)

### LSTM + Attention (Task 2)
- Bidirectional LSTM encoder with Bahdanau attention
- Teacher forcing with linear decay schedule
- 3.8M parameters

### Transformer (Task 3)
- Encoder-decoder Transformer with positional encoding
- Warmup + cosine annealing learning rate schedule
- 256 dim, 8 heads, 4 layers, 1024 FFN — 8.5M parameters

## Results

| Metric | Prosper Baseline | Our LSTM | Our Transformer |
|--------|:---:|:---:|:---:|
| Tokenization | Character-level | Math-aware | Math-aware |
| Data | 60K | 76K | 76K |
| Epochs | 3,890 | 40 | 50 |
| Val Loss | 2.35 | **1.65** | **TBD** |

> **Key insight**: Math-aware tokenization (treating `sin`, `cos`, `exp` as single tokens) dramatically improves convergence — our LSTM reaches lower loss in 40 epochs vs 3,890 epochs with character-level tokenization.

## Repository Structure

```
├── FASEROH_evaluation.ipynb    # Complete evaluation notebook (all 3 tasks)
├── data/
│   ├── prosper_60k.txt         # Prosper's 57K dataset
│   └── our_19k_data.csv        # Our SymPy-generated 19K dataset
├── models/
│   ├── best_lstm.pth           # Trained LSTM weights
│   └── best_transformer.pth    # Trained Transformer weights
├── plots/
│   └── full_evaluation.png     # Training curves + comparison plots
└── README.md
```

## Quick Start

```bash
pip install torch sympy numpy pandas matplotlib scikit-learn

# Run the full evaluation notebook
jupyter notebook FASEROH_evaluation.ipynb
```

## Tech Stack

- **PyTorch** — model implementation and training
- **SymPy** — symbolic math for data generation
- **HuggingFace-style architecture** — encoder-decoder with attention
- **Trained on**: NVIDIA H100 NVL (100GB VRAM)

## Author

**Vishal Lohiya**
- GitHub: [vishall4](https://github.com/vishall4)
- IIT Jodhpur — B.Tech Data Science & AI
- GSoC 2026 Contributor

## Acknowledgments

- [Harrison Prosper](https://github.com/hbprosper) (FSU) — FASEROH mentor, original symbolic-ai dataset
- [ML4Sci](https://ml4sci.org/) — GSoC umbrella organization

## License

MIT
