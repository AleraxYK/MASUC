# Unlearning Methods Comparison — CIFAR10

Format: mean ± std over multiple seeds (where available).

| Method | n | Retain Acc | Forget Acc |
|:---|:---:|:---:|:---:|
| **Before Unlearning** | — | 82.79% | 82.00% |
| **Fine-Tuning** | 3 | 94.19% ± 0.15% | 5.43% ± 0.61% |
| **NegGrad+** | 3 | 38.90% ± 2.84% | 0.00% ± 0.00% |
| **Random Labels** | 3 | 93.49% ± 0.12% | 0.00% ± 0.00% |
| **Bad-Teaching** | 3 | 87.19% ± 0.91% | 28.10% ± 15.70% |
| **SCRUB** | 3 | 88.59% ± 0.09% | 38.93% ± 1.16% |
| **MASUC (Ours)** | 3 | 83.60% ± 0.85% | 9.57% ± 2.30% |
| **Retrain (Gold Std.)** | 3 | 87.06% ± 1.09% | 0.00% ± 0.00% |

### Objective
- **Retain Accuracy** should be as high as possible (close to or higher than 'Before' / 'Retrain').
- **Forget Accuracy** should be as low as possible (ideally close to 1/N, where N is the number of classes).