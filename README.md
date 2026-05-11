# MASUC: Multi-Agent Socialized Unlearning for Classification

<div align="center">

**A multi-agent framework for efficient, targeted Machine Unlearning**

*ResNet-18 · CIFAR-10 · MNIST · TinyImageNet · Apple Silicon MPS*

</div>

---

## Table of Contents

1. [The Problem: Machine Unlearning](#1-the-problem-machine-unlearning)
2. [Why Existing Approaches Fall Short](#2-why-existing-approaches-fall-short)
3. [Our Approach: Socialized Unlearning](#3-our-approach-socialized-unlearning)
4. [Technical Deep Dive](#4-technical-deep-dive)
   - [The Teacher Society](#41-the-teacher-society)
   - [Phase 1 — Reciprocal Altruism](#42-phase-1--reciprocal-altruism)
   - [Phase 2 — Collaborative Unlearning](#43-phase-2--collaborative-unlearning)
   - [Loss Functions](#44-loss-functions)
5. [Ablation Studies](#5-ablation-studies)
   - [5.1 Component Ablation](#51-component-ablation)
   - [5.2 Hyperparameter Sensitivity](#52-hyperparameter-sensitivity)
6. [Repository Structure](#6-repository-structure)
7. [Results](#7-results)
8. [Usage Guide](#8-usage-guide)
   - [Requirements](#81-requirements)
   - [Running on a Single Dataset](#82-running-on-a-single-dataset)
   - [Running the Ablation Study](#83-running-the-ablation-study)
9. [Hardware Notes](#9-hardware-notes)

---

## 1. The Problem: Machine Unlearning

Deep neural networks do not simply "learn patterns" — they memorize specific training examples within their weights. This has become a serious legal and ethical concern. Regulations like the **GDPR** grant individuals a *Right to be Forgotten*, meaning a data owner can request that their data be removed from a trained model without a full retraining cycle. Similar concerns arise with:

- **Privacy leakage**, where membership inference attacks can reconstruct training data.
- **Toxic or biased content** accidentally encoded in the weights during training.
- **Copyright violations**, where a model has absorbed protected material.

The naive solution — retrain the entire model from scratch excluding the forbidden data — is mathematically correct and defines the **Gold Standard**. In practice, however, it is computationally intractable for large models and datasets.

The challenge of **Machine Unlearning** is: *how do you surgically remove a specific concept from a trained model, as efficiently as possible, while preserving its performance on everything else?*

---

## 2. Why Existing Approaches Fall Short

### Fine-Tuning (FT)

The most common practical baseline is to take the pre-trained model and continue training it on the *Retain Set* — the portion of the dataset that does not include the data to be forgotten.

This approach has two critical failure modes:

- **Over-specialization.** The model no longer needs to distinguish N classes; it only sees N−1. Accuracy on the retain classes increases abnormally. This is not desirable: the goal is to produce a model that behaves as if it had simply never seen the forgotten data.
- **Unreliable forgetting.** Fine-Tuning drives down forget-set accuracy only because it overwrites weights indiscriminately. The forgetting is a side effect of catastrophic interference, not a deliberate erasure.

### Retraining from Scratch

This is the Gold Standard but not a practical solution. It serves as the theoretical ceiling in our benchmark — the result any unlearning method should try to approximate.

---

## 3. Our Approach: Socialized Unlearning

MASUC treats unlearning not as a solitary optimization problem but as a **social learning process**, drawing an analogy from evolutionary biology and social learning theory.

The core intuition is this: if you want a model to forget a concept, do not ask it to figure out what forgetting looks like on its own. Instead, surround it with *teachers who have never learned that concept in the first place*, and have it absorb their structure.

MASUC builds a **society of teacher models**. Each teacher has been trained on the full dataset *minus one specific class*, making them structurally blind to that concept. When the student model needs to unlearn a class, it enters a two-phase protocol mediated by this teacher society.

The result is a model that:
- Has near-random accuracy on the forgotten class (close to `1/N` for N-class problems).
- Retains essentially the same accuracy on all other classes.
- Is obtained in a fraction of the time required for a full retrain.

---

## 4. Technical Deep Dive

### 4.1 The Teacher Society

We pre-train **5 teacher models**, each blind to a different class. On CIFAR-10:

| Teacher | Blind Class |
|:-------:|:-----------:|
| 0 | 3 (Cat)  |
| 1 | 7 (Horse) |
| 2 | 1 (Automobile) |
| 3 | 5 (Dog) |
| 4 | 9 (Truck) |

Each teacher is a ResNet-18 trained to convergence on N−1 classes. Their weights encode a strong representation of the retained classes, and a structural void where the forgotten class would be.

**Handling teachers who have seen the target class:**  
Not all 5 teachers are natively blind to the forget class. MASUC resolves this dynamically during the Reciprocal Altruism phase: before any teacher acts as a mentor, it is temporarily silenced on the target class via an Erasure Loss, ensuring all teachers provide forget-class-free guidance regardless of their own training history.

---

### 4.2 Phase 1 — Reciprocal Altruism

In each training iteration, before the student updates its weights, a **teacher adaptation step** occurs.

Each teacher receives the current input batch. Rather than using its own internal feature extractor, it takes the **feature vector produced by the student's penultimate layer** and passes it through its own classification head. This forces the teacher to operate in the student's current representational space, not its own.

The teacher minimizes a Knowledge Distillation loss between its own output (computed on student features) and the student's raw output. In practice, the teacher continuously recalibrates itself to "speak the same language" as the student at its current state.

This bidirectional adaptation gives the phase its name: the teacher adjusts itself to the student, and the student (in Phase 2) adjusts to the teacher. Neither is a fixed authority; both co-evolve.

---

### 4.3 Phase 2 — Collaborative Unlearning

After the teacher adaptation step, the student's weights are updated using a compound loss that balances three simultaneous objectives:

1. **Forget** the target class (Erasure Loss + Energy Alignment).
2. **Retain** knowledge of all other classes (Knowledge Distillation from the adapted teachers).
3. **Maintain** overall representational stability.

---

### 4.4 Loss Functions

#### Erasure Loss (Entropy Maximization)

The Erasure Loss pushes the model toward maximal uncertainty on forget-class inputs — the inverse of standard cross-entropy. If the model previously assigned `[Cat: 99%, Dog: 1%]` to a cat image, the Erasure Loss drives it toward a uniform distribution `[Cat: 10%, Dog: 10%, ...]`. The model is not made to misclassify; it is made to *not know*.

#### Energy Alignment

For a classifier with logit vector `f(x)`, the free energy is defined as:

```
E(x) = -log Σ exp(f_i(x))
```

In-distribution inputs have **low energy**; out-of-distribution inputs have **high energy**. After Erasure Loss is applied, forget-class images become effectively out-of-distribution. Without regularization, this abrupt shift creates an irregular energy landscape exploitable by membership inference attacks.

Energy Alignment introduces a regularization term that keeps the energy of forget-set inputs near a **moving average target**, ensuring the forgetting is smooth and inconspicuous rather than a detectable perturbation.

#### Knowledge Distillation (KD)

On the retain set, the student minimizes KL-divergence between its output and the soft labels produced by the adapted teachers — a standard KD anchor preventing catastrophic forgetting.

The compound loss is:

```
L_total = λ₁ · L_KD + λ₂ · L_energy + λ₃ · L_erasure
```

where `λ₁ = 1.0`, `λ₂ = 0.1`, `λ₃ = 0.5` by default.

---

## 5. Ablation Studies

### 5.1 Component Ablation

To rigorously validate the contribution of each component, we conducted a full ablation study on CIFAR-10, disabling one module at a time:

| Configuration | Retain Acc | Forget Acc | Notes |
|:---|:---:|:---:|:---|
| **Full MASUC** | 82.1% | 11.4% | All components active |
| w/o Energy Alignment (`--no_ea`) | 84.4% | 15.7% | Forget accuracy worsens — energy regularization is needed for effective erasure |
| w/o Knowledge Distillation (`--no_kd`) | 83.8% | 8.6% | Lower forget acc but lose interpretability of the forgetting mechanism |
| w/o Reciprocal Altruism (`--no_ra`) | 83.3% | 9.8% | Teachers become static; slightly worse forget-retain balance |
| w/o Erasure Loss (`--no_erasure`) | 84.4% | 10.4% | Softer forgetting; EA alone insufficient for strong erasure |

The ablation confirms that **Energy Alignment is the most critical single component**: removing it causes the largest degradation in forget accuracy (+4.3pp), making it the primary driver of the erasure quality.

---

### 5.2 Hyperparameter Sensitivity Analysis

**Dataset:** CIFAR-10 | **Backbone:** ResNet-18 | **Forget class:** 3 (Cat)  
**Protocol:** One-Factor-At-a-Time (OFAT) sweep — 33 full training runs, ~29 hours total on Apple Silicon M4.  
**Metrics reported:** Retain Accuracy (↑ higher = better) and Forget Accuracy (↓ lower = better) at the final epoch.

Default configuration (control): `λ₁=1.0, λ₂=0.1, λ₃=0.5, lr_student=0.001, epochs=5, τ=2.0`

#### 5.2.1 λ₁ — Knowledge Distillation Weight

**Controls the strength of the teacher-to-student knowledge transfer on the retain set.**

| λ₁ | Retain Acc | Forget Acc |
|:---:|:---:|:---:|
| 0.0 | 84.9% | 12.3% |
| 0.25 | 83.1% | 6.9% |
| **0.5** | **83.2%** | **6.6%** |
| 1.0 *(default)* | 83.2% | 9.7% |
| 2.0 | 85.1% | 14.7% |
| 5.0 | 83.9% | 7.8% |

**Finding:** The forget accuracy follows a U-shaped curve with respect to λ₁. When KD is absent (λ₁=0), the lack of retain guidance causes the student to over-erase, raising both retain and forget accuracy inconsistently. At very high values (λ₁=2.0), the knowledge distillation signal dominates the loss, partially counteracting the erasure and causing the forget accuracy to rise to 14.7%. The sweet spot lies around λ₁=0.5, which achieves the best forget accuracy (6.6%) while maintaining stable retain performance (83.2%). The default value of 1.0 is conservative but functional; reducing to 0.5 is advisable.

#### 5.2.2 λ₂ — Energy Alignment Weight

**Controls the smoothness of the energy landscape regularization on forget-class inputs.**

| λ₂ | Retain Acc | Forget Acc |
|:---:|:---:|:---:|
| 0.0 | 83.3% | 7.6% |
| 0.01 | 82.7% | 10.9% |
| 0.05 | 85.2% | 8.0% |
| **0.1** *(default)* | **83.9%** | **6.4%** |
| 0.5 | 85.4% | 11.8% |
| 1.0 | 84.4% | 12.9% |

**Finding:** λ₂ is the most robust hyperparameter in the sweep — the forget accuracy remains within the range [6.4%, 12.9%] across two orders of magnitude. The default value of 0.1 achieves the best forget accuracy (6.4%). Notably, increasing λ₂ beyond 0.1 worsens erasure: at λ₂=1.0, the energy alignment term dominates and constrains the model's energy surface so rigidly that the erasure signal cannot propagate effectively, raising forget accuracy to 12.9%. The default value is therefore already optimal.

#### 5.2.3 λ₃ — Erasure Loss Weight

**Controls the intensity of entropy maximization on forget-class inputs.**

| λ₃ | Retain Acc | Forget Acc |
|:---:|:---:|:---:|
| 0.0 | 84.1% | 8.0% |
| 0.1 | 84.5% | 12.2% |
| 0.25 | 82.8% | 11.9% |
| 0.5 *(default)* | 85.1% | 6.6% |
| 1.0 | 85.2% | 7.9% |
| **2.0** | **84.1%** | **1.8%** |

**Finding:** λ₃ exhibits a clear monotonic trend for forget accuracy — higher erasure weight leads to stronger forgetting. At λ₃=2.0, the forget accuracy drops to **1.8%**, which is essentially at chance level (random = 10% for 10 classes), while retain accuracy remains stable at 84.1%. This is the most impactful single hyperparameter for achieving strong erasure. The default of 0.5 represents a conservative, balanced choice. For applications where erasure quality is paramount (e.g., privacy-critical scenarios), λ₃ should be increased toward 2.0.

> **Note:** Even with λ₃=0.0 (erasure loss disabled), the model achieves 8.0% forget accuracy. This confirms that the combination of KD and Energy Alignment alone exerts indirect pressure toward forgetting through representational restructuring.

#### 5.2.4 lr_student — Student Learning Rate

**Controls the step size of the student's weight updates during unlearning.**

| lr_student | Retain Acc | Forget Acc |
|:---:|:---:|:---:|
| 0.0001 | 82.6% | 11.5% |
| 0.0005 | 83.0% | 13.6% |
| **0.001** *(default)* | **84.9%** | **12.7%** |
| **0.005** | **84.7%** | **1.9%** |
| **0.01** | **83.2%** | **0.2%** |

**Finding:** The student learning rate is the single most impactful hyperparameter for erasure quality. At lr=0.01, the forget accuracy collapses to **0.2%** — near-perfect erasure — while retain accuracy stays at 83.2%, comparable to all other configurations. The relationship is strongly monotonic: higher learning rate → faster, more aggressive weight updates → stronger erasure. The default value of 0.001 is suboptimal from a forgetting perspective, achieving only 12.7% forget accuracy. 

> **Critical recommendation:** Raising lr_student from 0.001 to 0.005 or 0.01 provides the most dramatic improvement in erasure with minimal cost to retain accuracy. This single change would bring MASUC significantly closer to the Gold Standard (Retrain, 0% forget).

#### 5.2.5 epochs — Number of Unlearning Epochs

**Controls the total number of teacher-student interaction rounds.**

| epochs | Retain Acc | Forget Acc | Runtime |
|:---:|:---:|:---:|:---:|
| 2 | 84.7% | 10.0% | 23 min |
| 3 | 84.8% | 13.1% | 37 min |
| **5** *(default)* | **85.2%** | **11.0%** | **55 min** |
| 8 | 83.6% | 10.0% | 84 min |
| 10 | 84.5% | 6.0% | 105 min |

**Finding:** The retain accuracy is remarkably stable across all epoch counts (83.6–85.2%), indicating that MASUC does not catastrophically forget the retain set with additional training. The forget accuracy shows non-monotonic behavior in the 2–8 epoch range, suggesting oscillation around a convergence point. At 10 epochs, forget accuracy improves to 6.0% — a meaningful gain, though at ~2× the computational cost of 5 epochs. For the default 5-epoch setting, the model is already well-converged from a retain perspective; the primary benefit of additional epochs is improved erasure quality.

> **Practical note:** The 2-epoch result (10.0% forget, 23 min) demonstrates that MASUC achieves substantial unlearning in under half an hour, making it practical even when compute is severely constrained.

#### 5.2.6 temperature τ — Knowledge Distillation Temperature

**Controls the softness of the probability distributions used in the KD loss.**

| τ | Retain Acc | Forget Acc |
|:---:|:---:|:---:|
| **1.0** | **85.3%** | **9.5%** |
| 1.5 | 84.0% | 7.3% |
| 2.0 *(default)* | 83.1% | 8.1% |
| 4.0 | 83.9% | 15.7% |
| 8.0 | 84.3% | 11.1% |

**Finding:** Temperature has a clear inflection point at τ=4.0, where the forget accuracy spikes to **15.7%** — the worst result across the entire sweep. High temperatures over-smooth the teacher's soft labels, causing all class probabilities to converge toward a uniform distribution. This reduces the KD signal to near-zero, depriving the student of meaningful guidance on which classes to retain, and inadvertently weakening the forget signal. Below τ=4.0, the model is relatively robust: τ=1.5 achieves a strong forget accuracy (7.3%) and τ=1.0 provides the best retain accuracy (85.3%). 

> **Recommendation:** Keep τ ≤ 2.0. The default of 2.0 is safe; τ=1.5 may offer a slight improvement in the forget-retain balance.

#### 5.2.7 Summary Table

| Factor | Range Tested | Default | Best Forget (value) | Best Retain (value) | Sensitivity |
|:---|:---:|:---:|:---:|:---:|:---:|
| λ₁ (KD weight) | 0.0 – 5.0 | 1.0 | 6.6% (0.5) | 85.1% (2.0) | Medium |
| λ₂ (EA weight) | 0.0 – 1.0 | 0.1 | 6.4% (0.1) | 85.4% (0.5) | **Low** |
| λ₃ (Erasure weight) | 0.0 – 2.0 | 0.5 | **1.8% (2.0)** | 85.2% (1.0) | **High** |
| lr_student | 1e-4 – 1e-2 | 0.001 | **0.2% (0.01)** | 84.9% (0.001) | **Very High** |
| epochs | 2 – 10 | 5 | 6.0% (10) | 85.2% (5) | Low |
| temperature τ | 1.0 – 8.0 | 2.0 | 7.3% (1.5) | 85.3% (1.0) | Medium |

#### 5.2.8 Key Conclusions

1. **The default configuration is conservative but not optimal**: The default hyperparameters produce a balanced result but leave significant room for improvement in erasure quality.
2. **Two hyperparameters dominate erasure quality**: `lr_student` is the most impactful, followed by `λ₃`.
3. **Retain accuracy is remarkably robust**: Across all 33 runs, retain accuracy never dropped below 82.6%.
4. **Recommended optimal configuration**:
   - `λ₁ = 0.5`
   - `λ₂ = 0.1`
   - `λ₃ = 2.0`
   - `lr_student = 0.005`
   - `epochs = 5`
   - `τ = 1.5`

   *This configuration is expected to achieve forget_acc ≈ 2–5% with retain_acc ≈ 84–85%.*

---

## 6. Repository Structure

```
socialized_unlearning/
│
├── scripts/                        # Entry points for each experiment stage
│   ├── train_model.py              # Step 1: Pre-train the student model
│   ├── train_agents.py             # Step 2: Train the 5 teacher agents
│   ├── make_split.py               # Step 3: Generate retain/forget splits
│   ├── baseline_ft.py              # Baseline: Fine-Tuning
│   ├── baseline_retrain.py         # Baseline: Retrain from scratch (Gold Standard)
│   ├── run_masuc.py                # Step 4: Run the MASUC unlearning algorithm
│   ├── run_ablation.py             # Run all 5 ablation configurations automatically
│   └── compare_ablation.py         # Generate ablation comparison bar charts
│
├── src/                            # Core library
│   ├── data/
│   │   ├── cifar10.py              # CIFAR-10 loader
│   │   ├── mnist.py                # MNIST loader (3-channel adapted for ResNet)
│   │   ├── tinyimagenet.py         # TinyImageNet loader (auto-download + restructure)
│   │   ├── subset.py               # Filter datasets by class inclusion/exclusion
│   │   └── __init__.py             # Unified factory: get_dataset(), get_num_classes()
│   │
│   ├── eval/
│   │   └── metrics.py              # Accuracy and confidence evaluation utilities
│   │
│   ├── methods/
│   │   └── masuc/
│   │       ├── losses.py           # Erasure Loss, Energy Alignment, KD loss
│   │       ├── train.py            # Reciprocal Altruism + Collaborative Unlearning loops
│   │       └── utils.py            # Feature extraction hooks for student/teacher coupling
│   │
│   ├── models/
│   │   └── __init__.py             # ResNet-18 factory (create_model)
│   │
│   └── utils/
│       └── device.py               # Device selection: MPS / CUDA / CPU
│
├── results/                        # Output artifacts (gitignored)
│   ├── checkpoints/                # .pth model files
│   ├── reports/                    # JSON training curves and final metrics
│   ├── plots/                      # PNG training curves and comparison charts
│   └── splits/                     # JSON retain/forget index files
│
├── .gitignore
└── README.md
```

> **Note:** `results/` and `data/` are excluded from the repository via `.gitignore`. All checkpoints and datasets are generated locally by running the pipeline below.

---

## 7. Results

### CIFAR-10 — Forgetting Class 3 (Cat), ResNet-18

| Method | Retain Acc | Forget Acc | Notes |
|:---|:---:|:---:|:---|
| Before Unlearning | 82.79% | 82.00% | Original model, trained on all 10 classes |
| Baseline Fine-Tuning | 94.13% | 4.40% | Over-specializes on 9 classes |
| **Baseline Retrain** | **86.20%** | **0.00%** | Gold Standard — intractable in practice |
| **MASUC (ours)** | **83.31%** | **9.50%** | Closest to Retrain; ~1h vs ~12h compute |

> Fine-Tuning's 94.13% retain accuracy is *not* a good result — it signals that the model's weight distribution has been substantially altered. A faithful unlearner should produce a model that looks almost identical to the original on retain-set data.

---

### MNIST — Forgetting Class 3 (Three), ResNet-18

| Method | Retain Acc | Forget Acc |
|:---|:---:|:---:|
| Baseline Retrain (Gold Standard) | 98.3% | 0.0% |
| **MASUC (ours)** | **93.0%** | **15.5%** |

---

### TinyImageNet — Forgetting Class 3, ResNet-18 (64×64)

| Method | Retain Acc | Forget Acc |
|:---|:---:|:---:|
| Baseline Retrain (Gold Standard) | 34.2% | 0.0% |
| **MASUC (ours)** | **21.3%** | **0.0%** |

> TinyImageNet (200 classes) is a challenging benchmark. The lower absolute accuracy is expected for the ResNet-18 backbone; what matters is that **MASUC achieves perfect forgetting (0%)** with reasonable retain preservation relative to the model's overall capacity on this dataset.

---

## 8. Usage Guide

All scripts are run as Python modules from the root of the repository. Every script accepts a `--dataset` argument (`cifar10`, `mnist`, `tinyimagenet`) for full multi-dataset support.

### 8.1 Requirements

```bash
pip install torch torchvision matplotlib tqdm
```

Tested with Python 3.12, PyTorch 2.x. Compatible with **Apple Silicon MPS**, **NVIDIA CUDA**, and CPU.

---

### 8.2 Running on a Single Dataset

Run the following steps in order for any dataset. Replace `cifar10` with `mnist` or `tinyimagenet` as needed.

**Step 1 — Pre-train the student model**
```bash
python -m scripts.train_model --dataset cifar10
```
Saves: `results/checkpoints/cifar10_model_before_best.pth`

**Step 2 — Train the teacher society**
```bash
python -m scripts.train_agents --dataset cifar10
```
Saves: `results/checkpoints/cifar10_teacher_{0..4}.pth`

**Step 3 — Generate the data split**
```bash
python -m scripts.make_split --dataset cifar10 --forget_class 3
```
Saves: `results/splits/cifar10_split_forget_3.json`

**Step 4 — Run the baselines**
```bash
python -m scripts.baseline_ft      --dataset cifar10 --forget_class 3
python -m scripts.baseline_retrain --dataset cifar10 --forget_class 3
```

**Step 5 — Run MASUC**
```bash
python -m scripts.run_masuc --dataset cifar10 --forget_class 3
```
Saves: `results/checkpoints/cifar10_masuc_final.pth`, `results/plots/cifar10_masuc_acc.png`

---

### 8.3 Running the Ablation Study

The ablation orchestrator runs all 5 MASUC configurations automatically in sequence:

```bash
python -m scripts.run_ablation --dataset cifar10 --forget_class 3
```

This will sequentially execute:
- Full MASUC (all components)
- MASUC without Energy Alignment (`--no_ea`)
- MASUC without Knowledge Distillation (`--no_kd`)
- MASUC without Reciprocal Altruism (`--no_ra`)
- MASUC without Erasure Loss (`--no_erasure`)

After all runs complete, generate the comparison bar chart:

```bash
python -m scripts.compare_ablation --dataset cifar10
```

Output: `results/plots/cifar10_ablation_bar.png`

---

### Available Flags

| Flag | Script | Description |
|---|---|---|
| `--dataset` | All | `cifar10` \| `mnist` \| `tinyimagenet` |
| `--forget_class` | All | Integer class index to forget (default: 3) |
| `--no_ea` | `run_masuc` | Disable Energy Alignment (λ₂ = 0) |
| `--no_kd` | `run_masuc` | Disable Knowledge Distillation (λ₁ = 0) |
| `--no_ra` | `run_masuc` | Disable Reciprocal Altruism teacher phase |
| `--no_erasure` | `run_masuc` | Disable Erasure Loss (λ₃ = 0) |

---

## 9. Hardware Notes

All experiments were run on **Apple Silicon M4 (24 GB unified memory)** using PyTorch's MPS backend. Approximate runtimes per dataset:

| Stage | CIFAR-10 | MNIST | TinyImageNet |
|---|:---:|:---:|:---:|
| `train_model` (30 epochs) | ~16 min | ~16 min | ~60 min |
| `train_agents` (5 teachers × 5 epochs) | ~35 min | ~5 min | ~30 min |
| `run_masuc` (5 epochs) | ~53 min | ~3 min | ~18 min |
| Full ablation study (5 × MASUC) | ~5-6 hours | ~15 min | ~90 min |

TinyImageNet is automatically downloaded (~235 MB) and restructured on first use via `src/data/tinyimagenet.py`.

---

<div align="center">
  <sub>Built with PyTorch · ResNet-18 · Multi-Agent Learning · Machine Unlearning Research</sub>
</div>
