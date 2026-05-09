import torch
import torch.nn as nn
from tqdm import tqdm
import time
from .losses import unlearning_energy_alignment_loss, unlearning_knowledge_distillation_loss, erasure_loss
from .utils import feature_extractor, classifier_extractor
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
    training_energies_target: torch.Tensor = None
) -> tuple[dict, torch.Tensor]:
    """
    Collaborative Unlearning Training (Student Phase).
    The student unlearns the target class while being constrained by the teachers.
    """
    student_model.train()
    running_loss, n = 0.0, 0
    t0 = time.time()

    if training_energies_target is None:
        training_energies_target = torch.tensor([], device=device)

    forget_classes = [forget_class]

    pbar = tqdm(forget_train_loader, desc=f"[Student CU] epoch {ep:02d}/{num_epochs}", leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)

        student_output = student_model(data)

        loss_kd = 0
        for tid, teacher_model in teacher_models.items():
            teacher_model.eval()
            with torch.no_grad():
                modified_input = feature_extractor(student_model, data)
                teacher_output1 = classifier_extractor(teacher_model, modified_input)
                teacher_output2 = teacher_model(data)

                loss_kd += unlearning_knowledge_distillation_loss(teacher_output1, teacher_output2, temperature=temperature)

        lambda_1 = initial_lambda_1 * (1 - (ep - 1) / num_epochs)
        lambda_loss_kd = lambda_1 * loss_kd

        loss_erasure = erasure_loss(student_output, forget_classes)
        lambda_loss_erasure = lambda_3 * loss_erasure

        loss_al, training_energies_target = unlearning_energy_alignment_loss(student_output, training_energies_target)
        lambda_loss_al = lambda_2 * loss_al

        loss = lambda_loss_erasure + lambda_loss_kd + lambda_loss_al
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        n += data.size(0)
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
    lambda_2: float = 0.3,
    temperature: float = 2.0,
    device: torch.device = torch.device("cpu"),
    training_energies_target: torch.Tensor = None
) -> torch.Tensor:
    """
    Reciprocal Altruism Training (Teacher Phase).
    Teachers adjust their knowledge given the student's state.
    """
    teacher_model.train()
    student_model.eval()
    running_loss, n = 0.0, 0
    t0 = time.time()

    if training_energies_target is None:
        training_energies_target = torch.tensor([], device=device)

    forget_classes = [forget_class]
    
    pbar = tqdm(forget_train_loader, desc=f"[Teacher {teacher_id} RA] epoch {ep:02d}/{num_epochs}", leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)


        teacher_output = teacher_model(data)


        loss_erasure = erasure_loss(teacher_output, forget_classes)

        with torch.no_grad():
            student_features = feature_extractor(student_model, data)
        teacher_output_after_student = classifier_extractor(teacher_model, student_features)
        loss_kd = unlearning_knowledge_distillation_loss(teacher_output_after_student, teacher_output, temperature=temperature)

        loss_al, training_energies_target = unlearning_energy_alignment_loss(teacher_output, training_energies_target)
        lambda_1 = initial_lambda_1 * (1 - (ep - 1) / num_epochs)

        loss = loss_erasure + lambda_1 * loss_kd + lambda_2 * loss_al

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        n += data.size(0)
        pbar.set_postfix(loss=running_loss / n)

    elapsed = time.time() - t0
    print(
        f"[Teacher {teacher_id} RA] epoch {ep:02d}/{num_epochs} "
        f"| loss {running_loss/n:.4f} "
        f"| elapsed {elapsed:.1f}s"
    )

    return training_energies_target
