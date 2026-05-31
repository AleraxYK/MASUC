# Multi-Class Unlearning — CIFAR10

Per-class Retain / Forget accuracy, demonstrating robustness to forget-class choice.
Format: mean ± std across seeds.

| Method \ Forget class | 3 (R) / 3 (F) |
|:---|:---:|
| **FT** | 94.28% / 3.67% |
| **NegGrad+** | 44.46% / 0.00% |
| **Random Labels** | 93.44% / 0.00% |
| **Bad-Teaching** | 86.24% / 24.23% |
| **SCRUB** | 88.50% / 28.57% |
| **MASUC (Ours)** | 86.76% / 11.10% |
| **Retrain (Gold)** | 85.06% / 0.00% |