# Unlearning Methods Comparison — CIFAR10

Format: mean ± std over multiple seeds (where available).

| Method | n | Retain Acc | Forget Acc |
|:---|:---:|:---:|:---:|
| **Fine-Tuning** | 3 | 94.28% ± 0.08% | 3.67% ± 0.40% |
| **NegGrad+** | 3 | 44.46% ± 4.31% | 0.00% ± 0.00% |
| **Random Labels** | 3 | 93.44% ± 0.20% | 0.00% ± 0.00% |
| **Bad-Teaching** | 3 | 86.24% ± 0.59% | 24.23% ± 10.81% |
| **SCRUB** | 3 | 88.50% ± 0.18% | 28.57% ± 1.10% |
| **MASUC (Ours)** | 3 | 86.76% ± 0.19% | 11.10% ± 0.56% |
| **Retrain (Gold Std.)** | 3 | 85.06% ± 1.28% | 0.00% ± 0.00% |

### Objective
- **Retain Accuracy** should be as high as possible (close to or higher than 'Before' / 'Retrain').
- **Forget Accuracy** should be as low as possible (ideally close to 1/N, where N is the number of classes).