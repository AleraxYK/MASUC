"""
Calibration metrics for unlearned models.

Used to quantify the "over-specialization pathology" of Fine-Tuning-style
unlearning baselines: they inflate retain accuracy but produce poorly
calibrated predictions (and saturated confidences on the forget class).

Key metrics:
  * Expected Calibration Error (ECE) on retain set
  * Mean max-softmax confidence on retain set
  * Mean max-softmax confidence on forget set (low = good unlearning,
    high = confident misprediction = pathological)
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (probs, labels) numpy arrays for a whole loader."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        probs = F.softmax(model(x), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (Guo et al., 2017).

    Bins predictions by max-softmax confidence and measures the discrepancy
    between average confidence and average accuracy within each bin.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Returns ECE in [0, 1]. Lower = better calibrated.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_weight = in_bin.sum() / n
        bin_acc    = accuracies[in_bin].mean()
        bin_conf   = confidences[in_bin].mean()
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Worst-bin gap |acc - conf| (Maximum Calibration Error)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    gaps = []
    for lo, hi in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        gaps.append(abs(accuracies[in_bin].mean() - confidences[in_bin].mean()))
    return float(max(gaps)) if gaps else 0.0


def calibration_report(model, retain_loader, forget_loader, device, n_bins: int = 15) -> dict:
    """
    Full calibration report for a model post-unlearning.

    Returns:
        ece_retain      : ECE on the retain test set (lower = better)
        mce_retain      : Maximum CE on retain
        conf_retain     : mean max-softmax confidence on retain
        acc_retain      : top-1 accuracy on retain (sanity check)
        conf_forget     : mean max-softmax confidence on forget set
                          (low = good unlearning; high = pathological
                          confident misclassification)
        max_other_class_conf_forget : mean of max probability ACROSS non-forget
                          classes on forget inputs — quantifies whether the
                          model confidently re-routes to a specific wrong class
    """
    r_probs, r_labels = collect_predictions(model, retain_loader, device)
    f_probs, f_labels = collect_predictions(model, forget_loader, device)

    ece_retain = expected_calibration_error(r_probs, r_labels, n_bins)
    mce_retain = maximum_calibration_error(r_probs, r_labels, n_bins)
    conf_retain = float(r_probs.max(axis=1).mean())
    acc_retain  = float((r_probs.argmax(axis=1) == r_labels).mean())
    conf_forget = float(f_probs.max(axis=1).mean())

    # Probability mass on non-forget classes when input is forget class
    if len(f_labels) > 0:
        forget_cls = int(f_labels[0])  # all the same for class-level unlearning
        mask = np.ones(f_probs.shape[1], dtype=bool)
        mask[forget_cls] = False
        max_other = f_probs[:, mask].max(axis=1).mean()
    else:
        max_other = float("nan")

    return {
        "ece_retain":                  ece_retain,
        "mce_retain":                  mce_retain,
        "conf_retain":                 conf_retain,
        "acc_retain":                  acc_retain,
        "conf_forget":                 conf_forget,
        "max_other_class_conf_forget": float(max_other),
    }
