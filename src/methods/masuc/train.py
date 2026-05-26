import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from tqdm import tqdm

from .losses import unlearning_energy_alignment_loss, unlearning_knowledge_distillation_loss, erasure_loss
from .utils import forward_with_features, EnergyRunningMean
from src.eval.metrics import evaluate


def collaborative_unlearning(
    ep: int,
    num_epochs: int,
    student_model: nn.Module,
    teacher_models: dict[int, nn.Module],
    forget_train_loader: torch.utils.data.DataLoader,
    retain_test_loader: torch.utils.data.DataLoader,
    forget_test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    forget_class: int,
    initial_lambda_1: float = 1.0,
    lambda_2: float = 0.1,
    lambda_3: float = 0.5,
    temperature: float = 2.0,
    device: torch.device = torch.device("cpu"),
    training_energies_target=None,
    use_amp: bool = False,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> tuple[dict, "EnergyRunningMean"]:
    """
    Phase 2: student unlearning step with KD anchor + EA regularizer + Erasure.

    Optimizations vs the original implementation:
      * one student forward per batch (logits + features in a single pass)
      * teacher forwards under no_grad (gradients only flow through student head)
      * EnergyRunningMean replaces torch.cat-accumulator (O(1) per batch, no sync)
      * optional CUDA mixed-precision (AMP) when use_amp=True
    """
    student_model.train()
    for tm in teacher_models.values():
        tm.eval()

    if training_energies_target is None or not isinstance(training_energies_target, EnergyRunningMean):
        training_energies_target = EnergyRunningMean(device=device)

    running_loss, n = 0.0, 0
    t0 = time.time()
    forget_classes = [forget_class]

    autocast_ctx = (torch.cuda.amp.autocast if use_amp and device.type == "cuda" else None)

    pbar = tqdm(forget_train_loader, desc=f"[Student CU] epoch {ep:02d}/{num_epochs}", leave=False)
    for data, labels in pbar:
        data   = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        ctx = autocast_ctx() if autocast_ctx is not None else nullcontext()
        with ctx:
            # Single student forward — yields logits AND features.
            student_output, student_features = forward_with_features(student_model, data)

            # KD anchor: teacher's own head on student features vs teacher's own logits.
            loss_kd = 0
            for teacher_model in teacher_models.values():
                teacher_output1 = teacher_model.fc(student_features)
                with torch.no_grad():
                    teacher_output2, _ = forward_with_features(teacher_model, data)
                loss_kd = loss_kd + unlearning_knowledge_distillation_loss(
                    teacher_output1, teacher_output2, temperature=temperature
                )

            lambda_1 = initial_lambda_1 * (1 - (ep - 1) / num_epochs)

            loss_erasure = erasure_loss(student_output, forget_classes)
            loss_al, training_energies_target = unlearning_energy_alignment_loss(
                student_output, training_energies_target
            )

            loss = lambda_3 * loss_erasure + lambda_1 * loss_kd + lambda_2 * loss_al

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = data.size(0)
        running_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=running_loss / n)

    metrics = evaluate(student_model, {"retain": retain_test_loader, "forget": forget_test_loader}, device)
    metrics = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}

    elapsed = time.time() - t0
    print(
        f"[Student CU] epoch {ep:02d}/{num_epochs} "
        f"| loss {running_loss/n:.4f} "
        f"| retain_acc {metrics['retain_acc']:.4f} "
        f"| forget_acc {metrics['forget_acc']:.4f} "
        f"| elapsed {elapsed:.1f}s"
    )

    metrics["train_loss"] = running_loss / n
    return metrics, training_energies_target


def reciprocal_altruism(
    ep: int,
    num_epochs: int,
    teacher_id: int,
    teacher_model: nn.Module,
    student_model: nn.Module,
    forget_train_loader: torch.utils.data.DataLoader,
    forget_class: int,
    optimizer: torch.optim.Optimizer,
    initial_lambda_1: float = 1.0,
    lambda_2: float = 0.1,
    temperature: float = 2.0,
    device: torch.device = torch.device("cpu"),
    training_energies_target=None,
    use_amp: bool = False,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> "EnergyRunningMean":
    """
    Phase 1: each teacher adapts its own classification head to operate on
    the student's penultimate features (Reciprocal Altruism step).

    Optimizations:
      * student features computed under no_grad with a single forward
      * EnergyRunningMean replaces torch.cat accumulator
      * optional AMP on CUDA
    """
    teacher_model.train()
    student_model.eval()

    if training_energies_target is None or not isinstance(training_energies_target, EnergyRunningMean):
        training_energies_target = EnergyRunningMean(device=device)

    running_loss, n = 0.0, 0
    t0 = time.time()
    forget_classes = [forget_class]

    autocast_ctx = (torch.cuda.amp.autocast if use_amp and device.type == "cuda" else None)

    pbar = tqdm(forget_train_loader, desc=f"[Teacher {teacher_id} RA] epoch {ep:02d}/{num_epochs}", leave=False)
    for data, _labels in pbar:
        data = data.to(device, non_blocking=True)

        ctx = autocast_ctx() if autocast_ctx is not None else nullcontext()
        with ctx:
            teacher_output, _ = forward_with_features(teacher_model, data)
            loss_erasure = erasure_loss(teacher_output, forget_classes)

            with torch.no_grad():
                _, student_features = forward_with_features(student_model, data)
            teacher_output_after_student = teacher_model.fc(student_features)

            loss_kd = unlearning_knowledge_distillation_loss(
                teacher_output_after_student, teacher_output, temperature=temperature
            )

            loss_al, training_energies_target = unlearning_energy_alignment_loss(
                teacher_output, training_energies_target
            )

            lambda_1 = initial_lambda_1 * (1 - (ep - 1) / num_epochs)
            loss = loss_erasure + lambda_1 * loss_kd + lambda_2 * loss_al

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = data.size(0)
        running_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=running_loss / n)

    elapsed = time.time() - t0
    print(
        f"[Teacher {teacher_id} RA] epoch {ep:02d}/{num_epochs} "
        f"| loss {running_loss/n:.4f} "
        f"| elapsed {elapsed:.1f}s"
    )

    return training_energies_target
