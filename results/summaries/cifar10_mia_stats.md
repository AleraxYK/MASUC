# Membership Inference Attack — CIFAR10 (forget class 3)

Members = forget_train (in original training set). Non-members = forget_test (held out, never seen).

AUC ≈ 0.5 → no recoverable membership signal (ideal privacy). AUC > 0.5 → leakage.

Format: mean ± std over multiple seeds.

| Method | n | AUC (loss) | AUC (conf) | AUC (entropy) | Best-thr Acc (loss) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Before Unlearning** | 1 | 0.5363 | 0.5055 | 0.5069 | 0.5342 |
| **Fine-Tuning** | 3 | 0.5333 ± 0.0009 | 0.4978 ± 0.0055 | 0.4951 ± 0.0041 | 0.5352 ± 0.0034 |
| **NegGrad+** | 3 | 0.5163 ± 0.0010 | 0.5020 ± 0.0088 | 0.5022 ± 0.0086 | 0.5163 ± 0.0010 |
| **Random Labels** | 3 | 0.5501 ± 0.0024 | 0.4344 ± 0.0040 | 0.4348 ± 0.0019 | 0.5477 ± 0.0040 |
| **Bad-Teaching** | 3 | 0.5463 ± 0.0024 | 0.4761 ± 0.0035 | 0.4741 ± 0.0063 | 0.5381 ± 0.0004 |
| **SCRUB** | 3 | 0.5522 ± 0.0016 | 0.4821 ± 0.0031 | 0.4883 ± 0.0029 | 0.5441 ± 0.0047 |
| **MASUC (Ours)** | 3 | 0.5550 ± 0.0018 | 0.4586 ± 0.0047 | 0.4577 ± 0.0041 | 0.5444 ± 0.0012 |
| **MASUC w/o EA** | 3 | 0.5544 ± 0.0016 | 0.4585 ± 0.0009 | 0.4581 ± 0.0013 | 0.5428 ± 0.0012 |
| **MASUC w/o KD** | 3 | 0.5610 ± 0.0023 | 0.4612 ± 0.0013 | 0.4597 ± 0.0022 | 0.5506 ± 0.0009 |
| **MASUC w/o RA** | 3 | 0.5316 ± 0.0030 | 0.4909 ± 0.0036 | 0.4929 ± 0.0034 | 0.5345 ± 0.0039 |
| **MASUC w/o Erasure** | 3 | 0.5550 ± 0.0011 | 0.4612 ± 0.0019 | 0.4597 ± 0.0015 | 0.5421 ± 0.0013 |
| **Retrain (Gold)** | 3 | 0.5037 ± 0.0122 | 0.5000 ± 0.0070 | 0.5004 ± 0.0104 | 0.5133 ± 0.0070 |