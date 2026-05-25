# Membership Inference Attack — CIFAR10 (forget class 3)

Members = forget_train (in original training set). Non-members = forget_test (held out, never seen).

AUC ≈ 0.5 → no recoverable membership signal (ideal privacy). AUC > 0.5 → leakage.

Format: mean ± std over multiple seeds.

| Method | n | AUC (loss) | AUC (conf) | AUC (entropy) | Best-thr Acc (loss) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Before Unlearning** | 1 | 0.5315 | 0.5134 | 0.5131 | 0.5344 |
| **Fine-Tuning** | 3 | 0.5232 ± 0.0013 | 0.5072 ± 0.0016 | 0.5038 ± 0.0002 | 0.5271 ± 0.0025 |
| **NegGrad+** | 3 | 0.5031 ± 0.0003 | 0.5018 ± 0.0054 | 0.5021 ± 0.0053 | 0.5117 ± 0.0014 |
| **Random Labels** | 3 | 0.5534 ± 0.0013 | 0.4412 ± 0.0030 | 0.4367 ± 0.0011 | 0.5464 ± 0.0014 |
| **Bad-Teaching** | 3 | 0.5403 ± 0.0014 | 0.4733 ± 0.0076 | 0.4723 ± 0.0058 | 0.5374 ± 0.0017 |
| **SCRUB** | 3 | 0.5416 ± 0.0009 | 0.4966 ± 0.0036 | 0.5038 ± 0.0034 | 0.5343 ± 0.0014 |
| **MASUC (Ours)** | 3 | 0.5477 ± 0.0014 | 0.4692 ± 0.0142 | 0.4669 ± 0.0097 | 0.5400 ± 0.0018 |
| **MASUC w/o EA** | 3 | 0.5486 ± 0.0009 | 0.4716 ± 0.0080 | 0.4706 ± 0.0053 | 0.5394 ± 0.0037 |
| **MASUC w/o KD** | 3 | 0.5476 ± 0.0030 | 0.4695 ± 0.0109 | 0.4682 ± 0.0073 | 0.5397 ± 0.0009 |
| **MASUC w/o RA** | 3 | 0.5471 ± 0.0024 | 0.4697 ± 0.0015 | 0.4684 ± 0.0020 | 0.5357 ± 0.0026 |
| **MASUC w/o Erasure** | 3 | 0.5497 ± 0.0016 | 0.4694 ± 0.0124 | 0.4674 ± 0.0098 | 0.5404 ± 0.0014 |
| **Retrain (Gold)** | 3 | 0.5059 ± 0.0092 | 0.5005 ± 0.0107 | 0.4999 ± 0.0073 | 0.5105 ± 0.0020 |