# MASUC Ablation Study — CIFAR10

Results over 3 seed(s). Format: mean ± std (sample std, ddof=1).

∆ Forget = forget_acc relative to Full MASUC (positive = worse forgetting, negative = better).

| Configuration | n | Retain Acc | Forget Acc | ∆ Forget |
|:---|:---:|:---:|:---:|:---:|
| **Full MASUC** | 3 | 86.76% ± 0.19% | 11.10% ± 0.56% | — |
| **w/o Energy Alignment** | 3 | 86.62% ± 0.24% | 12.90% ± 0.60% | +1.80% |
| **w/o Knowledge Distillation** | 3 | 84.60% ± 0.12% | 8.90% ± 0.35% | -2.20% |
| **w/o Reciprocal Altruism** | 3 | 87.32% ± 0.16% | 60.77% ± 0.85% | +49.67% |
| **w/o Erasure Loss** | 3 | 86.87% ± 0.17% | 13.67% ± 0.81% | +2.57% |