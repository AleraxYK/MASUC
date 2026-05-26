import torch
import torch.nn as nn
import torch.nn.functional as F


def unlearning_energy_alignment_loss(predictions: torch.Tensor, energy_state) -> tuple[torch.Tensor, "object"]:
    """
    Energy Alignment loss: regularizes the per-sample free energy of the
    student's predictions toward a running mean target.

    Args:
        predictions: raw logits (B, N).
        energy_state: an ``EnergyRunningMean`` instance OR a legacy 1-D
            tensor (kept for backward compatibility — slow, do not use in
            new code).

    Returns:
        loss   : scalar tensor.
        energy_state: same object, updated in place.

    The math matches the paper (cumulative mean target within an epoch),
    but the running mean implementation is O(1) per call instead of the
    original O(N) torch.cat.
    """
    energy = -torch.logsumexp(predictions, dim=1)

    # Fast path: running-mean state object
    from .utils import EnergyRunningMean
    if isinstance(energy_state, EnergyRunningMean):
        delta = energy_state.update(energy)
        return ((energy - delta) ** 2).mean(), energy_state

    # Legacy path (tensor accumulator) — kept so old call sites still work
    if energy_state is None:
        energy_state = torch.tensor([], device=energy.device)
    energy_state = torch.cat((energy_state, energy.detach()))
    delta = energy_state.mean()
    return ((energy - delta) ** 2).mean(), energy_state


def unlearning_knowledge_distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    KL divergence between temperature-softened student and teacher logits.
    """
    soft_student = torch.log_softmax(student_output / temperature, dim=1)
    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)


def erasure_loss(student_output: torch.Tensor, forget_classes: list[int]) -> torch.Tensor:
    """
    Drives the student toward reduced confidence on forget-class inputs by
    minimizing the partial entropy contribution of the target class(es).

        L_eras = - (1/B) * Σ_x  Σ_{c in forget_classes}  p_c(x) * log(p_c(x) + ε)

    Note: the sign is correct as written. The gradient pushes p_c → 0 from
    the typical post-pretraining starting point (p_c ≈ 1), which is the
    desired unlearning behaviour. See paper §III-E for the derivation.
    """
    probabilities = F.softmax(student_output, dim=1)
    forget_probs  = probabilities[:, forget_classes]
    return -torch.mean(torch.sum(forget_probs * torch.log(forget_probs + 1e-8), dim=1))
