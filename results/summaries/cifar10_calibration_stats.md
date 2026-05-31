# Calibration Analysis — CIFAR10 (forget class 3)

Lower ECE = better-calibrated retain predictions.
Lower conf_forget = more uncertain (better) predictions on forget class.
High conf_forget on a method with low forget accuracy indicates the
model confidently MISpredicts the forget class — a pathological
side-effect of aggressive over-specialization.

| Method | n | ECE (retain) | Conf (retain) | Conf (forget) | Max-other-class conf (forget) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Before Unlearning** | 1 | 0.0355 | 0.8751 | 0.7938 | 0.2502 |
| **Fine-Tuning** | 3 | 0.0052 ± 0.0012 | 0.9431 ± 0.0003 | 0.7447 ± 0.0064 | 0.7294 ± 0.0074 |
| **NegGrad+** | 3 | 0.4542 ± 0.0450 | 0.8986 ± 0.0043 | 0.9139 ± 0.0189 | 0.9139 ± 0.0189 |
| **Random Labels** | 3 | 0.0568 ± 0.0026 | 0.8775 ± 0.0026 | 0.2201 ± 0.0027 | 0.2201 ± 0.0027 |
| **Bad-Teaching** | 3 | 0.0979 ± 0.0088 | 0.7646 ± 0.0076 | 0.3657 ± 0.0245 | 0.3158 ± 0.0258 |
| **SCRUB** | 3 | 0.0140 ± 0.0019 | 0.8793 ± 0.0005 | 0.5933 ± 0.0014 | 0.4838 ± 0.0059 |
| **MASUC (Ours)** | 3 | 0.0358 ± 0.0015 | 0.9032 ± 0.0011 | 0.6275 ± 0.0004 | 0.5758 ± 0.0033 |
| **MASUC w/o EA** | 3 | 0.0423 ± 0.0016 | 0.9085 ± 0.0011 | 0.6485 ± 0.0010 | 0.5836 ± 0.0037 |
| **MASUC w/o KD** | 3 | 0.0496 ± 0.0013 | 0.8954 ± 0.0004 | 0.6406 ± 0.0008 | 0.6016 ± 0.0026 |
| **MASUC w/o RA** | 3 | 0.0143 ± 0.0010 | 0.8621 ± 0.0008 | 0.6532 ± 0.0047 | 0.3283 ± 0.0049 |
| **MASUC w/o Erasure** | 3 | 0.0344 ± 0.0014 | 0.9027 ± 0.0011 | 0.6246 ± 0.0004 | 0.5622 ± 0.0038 |
| **Retrain (Gold)** | 3 | 0.0215 ± 0.0099 | 0.8710 ± 0.0017 | 0.6727 ± 0.0424 | 0.6727 ± 0.0424 |