# Compute Wall-Clock Accounting — CIFAR10

All times measured on the same hardware. Reports per-method unlearning cost (excluding pre-training).

**Offline pre-training cost (amortizable):**
- Student (one-shot): see `cifar10_train_curve.json` or per-epoch logs
- Teacher society (K=5): total 3003.0s (50.1 min) — sum of independent runs

## Per-method unlearning wall-clock

| Method | n seeds | Mean elapsed (s) | Mean elapsed (min) |
|:---|:---:|:---:|:---:|
| **Fine-Tuning** | 3 | 406.6 ± 1.1 | 6.78 ± 0.02 |
| **NegGrad+** | 3 | 1281.3 ± 9.5 | 21.35 ± 0.16 |
| **Random Labels** | 3 | 4387.4 ± 111.5 | 73.12 ± 1.86 |
| **Bad-Teaching** | 3 | 1535.0 ± 15.6 | 25.58 ± 0.26 |
| **SCRUB** | 3 | 549.3 ± 0.6 | 9.15 ± 0.01 |
| **MASUC (Ours)** | 3 | 379.0 ± 3.2 | 6.32 ± 0.05 |
| **MASUC w/o EA** | 3 | 254.2 ± 1.7 | 4.24 ± 0.03 |
| **MASUC w/o KD** | 3 | 253.3 ± 0.6 | 4.22 ± 0.01 |
| **MASUC w/o RA** | 3 | 110.7 ± 0.3 | 1.85 ± 0.01 |
| **MASUC w/o Erasure** | 3 | 252.0 ± 0.9 | 4.20 ± 0.02 |
| **Retrain (Gold)** | 3 | 1641.8 ± 4.0 | 27.36 ± 0.07 |

## Notes
- *Per-method* numbers exclude offline pre-training (student + teachers).
- *Amortized* MASUC cost over N forget requests:
  `cost(N) = teacher_pretrain + N * unlearning_phase`
- Speed-up vs Retrain is computed against the *Retrain (Gold)* row.