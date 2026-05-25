# MASUC Ablation Study — CIFAR10

Results over 3 seed(s). Format: mean ± std (sample std, ddof=1).

∆ Forget = forget_acc relative to Full MASUC (positive = worse forgetting, negative = better).

| Configuration | n | Retain Acc | Forget Acc | ∆ Forget |
|:---|:---:|:---:|:---:|:---:|
| **Full MASUC** | 3 | 85.01% ± 0.81% | 11.87% ± 3.74% | — |
| **w/o Energy Alignment** | 3 | 82.60% ± 1.00% | 8.60% ± 2.02% | -3.27% |
| **w/o Knowledge Distillation** | 3 | 83.60% ± 0.85% | 9.57% ± 2.30% | -2.30% |
| **w/o Reciprocal Altruism** | 3 | 84.11% ± 0.63% | 8.60% ± 2.01% | -3.27% |
| **w/o Erasure Loss** | 3 | 83.68% ± 0.76% | 13.60% ± 3.28% | +1.73% |