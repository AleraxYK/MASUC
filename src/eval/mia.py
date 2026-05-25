import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def compute_per_sample_scores(model, loader, device: torch.device) -> dict:
    """
    Per-sample MIA signals: cross-entropy loss, max-softmax confidence, and entropy.

    All scores are computed under the model's TRUE-LABEL view (loss uses the
    dataset label y). For forget-class samples y == forget_class for every
    sample, so this is equivalent to the negative log-probability the model
    assigns to the forget class.
    """
    model.eval()
    losses, confs, entrs = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        log_p  = F.log_softmax(logits, dim=1)
        probs  = log_p.exp()

        ce   = F.nll_loss(log_p, y, reduction="none")
        conf = probs.max(dim=1).values
        ent  = -(probs * log_p).sum(dim=1)

        losses.append(ce.detach().cpu().numpy())
        confs.append(conf.detach().cpu().numpy())
        entrs.append(ent.detach().cpu().numpy())

    return {
        "loss":    np.concatenate(losses),
        "conf":    np.concatenate(confs),
        "entropy": np.concatenate(entrs),
    }


def _attack_score(member_vals: np.ndarray, nonmember_vals: np.ndarray, attack: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (y_true, y_score) arrays for ROC-AUC.
    Membership intuition (member > nonmember in score):
      - loss:    members have lower loss     → score = -loss
      - conf:    members have higher conf    → score = +conf
      - entropy: members have lower entropy  → score = -entropy
    """
    if attack == "loss":
        m, nm = -member_vals, -nonmember_vals
    elif attack == "conf":
        m, nm = member_vals, nonmember_vals
    elif attack == "entropy":
        m, nm = -member_vals, -nonmember_vals
    else:
        raise ValueError(f"Unknown attack: {attack}")

    y_true  = np.concatenate([np.ones_like(m), np.zeros_like(nm)])
    y_score = np.concatenate([m, nm])
    return y_true, y_score


def mia_auc(member_vals: np.ndarray, nonmember_vals: np.ndarray, attack: str) -> float:
    """
    ROC-AUC of distinguishing members from non-members using a single signal.
    0.5 = no leak (ideal privacy); >0.5 = recoverable membership signal.
    """
    y_true, y_score = _attack_score(member_vals, nonmember_vals, attack)
    return float(roc_auc_score(y_true, y_score))


def mia_threshold_accuracy(member_vals: np.ndarray, nonmember_vals: np.ndarray, attack: str) -> float:
    """
    Best-threshold balanced accuracy of the attack (sweeps the threshold).
    0.5 = no leak; higher = more leakage.
    """
    y_true, y_score = _attack_score(member_vals, nonmember_vals, attack)
    order  = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    cum_neg_below = np.cumsum(y_true == 0)
    cum_pos_below = np.cumsum(y_true == 1)
    tn = cum_neg_below
    fn = cum_pos_below
    fp = n_neg - tn
    tp = n_pos - fn

    tpr = tp / n_pos
    fpr = fp / n_neg
    bal = 0.5 * (tpr + (1.0 - fpr))
    return float(bal.max())


def mia_evaluate(model, members_loader, nonmembers_loader, device: torch.device) -> dict:
    """
    Run all three single-signal attacks and return AUC + best-threshold accuracy.
    """
    m  = compute_per_sample_scores(model, members_loader,    device)
    nm = compute_per_sample_scores(model, nonmembers_loader, device)

    out = {
        "n_members":     int(m["loss"].shape[0]),
        "n_nonmembers":  int(nm["loss"].shape[0]),
    }
    for attack in ("loss", "conf", "entropy"):
        out[f"auc_{attack}"] = mia_auc(m[attack], nm[attack], attack)
        out[f"adv_{attack}"] = mia_threshold_accuracy(m[attack], nm[attack], attack)

    out["mean_loss_member"]    = float(np.mean(m["loss"]))
    out["mean_loss_nonmember"] = float(np.mean(nm["loss"]))
    out["mean_conf_member"]    = float(np.mean(m["conf"]))
    out["mean_conf_nonmember"] = float(np.mean(nm["conf"]))

    return out
