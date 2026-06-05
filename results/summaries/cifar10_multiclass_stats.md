# Multi-Class Unlearning — CIFAR10

Per-class Retain / Forget accuracy, demonstrating robustness to forget-class choice.
Format: mean ± std across seeds.

| Method \ Forget class | 3 (R) / 3 (F) | 5 (R) / 5 (F) | 7 (R) / 7 (F) |
|:---|:---:|:---:|:---:|
| **FT** | 94.28% / 3.67% | 93.86% / 3.60% | 92.31% / 36.70% |
| **NegGrad+** | 44.46% / 0.00% | 51.59% / 0.00% | 62.62% / 0.00% |
| **Random Labels** | 93.44% / 0.00% | 92.77% / 0.00% | 91.40% / 0.00% |
| **Bad-Teaching** | 86.24% / 24.23% | 79.45% / 26.63% | 83.82% / 51.60% |
| **SCRUB** | 88.50% / 28.57% | 87.87% / 1.37% | 86.23% / 21.17% |
| **MASUC (Ours)** | 86.76% / 11.10% | 83.08% / 13.50% | 83.60% / 18.67% |
| **Retrain (Gold)** | 85.06% / 0.00% | 85.53% / 0.00% | 82.51% / 0.00% |