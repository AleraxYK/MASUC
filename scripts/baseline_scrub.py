import os
import time
import argparse
import json as js
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data import get_dataset, get_num_classes
from src.models import create_model
from src.eval.metrics import evaluate
from src.utils.device import get_device
from src.utils.seed import set_seed


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL(student || teacher) on temperature-softened distributions."""
    s = F.log_softmax(student_logits / temperature, dim=1)
    t = F.softmax(teacher_logits / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(s, t) * (temperature ** 2)


def baseline_scrub():
    """
    SCRUB baseline (Kurmanji et al., NeurIPS 2023).
    Alternating min-max optimization w.r.t. a frozen teacher (= original model):
        - MAX step (forget): maximize KL(student(x_f), teacher(x_f))
        - MIN step (retain): minimize KL(student(x_r), teacher(x_r)) + gamma * CE(student(x_r), y_r)
    Max steps are run only for the first max_steps_epochs epochs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",          type=str,   default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class",     type=int,   default=3)
    parser.add_argument("--seed",             type=int,   default=None, help="Random seed for reproducibility")
    parser.add_argument("--epochs",           type=int,   default=5)
    parser.add_argument("--max_steps_epochs", type=int,   default=None, help="Epochs during which the max step is active (default: equal to --epochs, i.e. always on)")
    parser.add_argument("--lr",               type=float, default=0.0005)
    parser.add_argument("--temperature",      type=float, default=2.0)
    parser.add_argument("--gamma",            type=float, default=1.0,  help="CE task-loss weight in the min step (default: 1.0)")
    args = parser.parse_args()
    ds = args.dataset
    fc = args.forget_class

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Seed: {args.seed}")

    if args.max_steps_epochs is None:
        args.max_steps_epochs = args.epochs

    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/reports",     exist_ok=True)
    os.makedirs("results/plots",       exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | Dataset: {ds.upper()} | tau={args.temperature} | gamma={args.gamma} | max_epochs={args.max_steps_epochs}")

    hyperparams = {
        "backbone":         "resnet18",
        "epochs":           args.epochs,
        "max_steps_epochs": args.max_steps_epochs,
        "batch_size":       128,
        "lr":               args.lr,
        "momentum":         0.9,
        "weight_decay":     5e-4,
        "num_workers":      4,
        "temperature":      args.temperature,
        "gamma":            args.gamma,
        "seed":             args.seed,
    }

    train_ds = get_dataset(ds, "./data", train=True)
    test_ds  = get_dataset(ds, "./data", train=False)

    split_path = f"results/splits/{ds}_split_forget_{fc}.json"
    with open(split_path, "r") as f:
        split = js.load(f)

    retain_train_ds = Subset(train_ds, split["indices"]["retain_train"])
    forget_train_ds = Subset(train_ds, split["indices"]["forget_train"])
    forget_test_ds  = Subset(test_ds,  split["indices"]["forget_test"])
    retain_test_ds  = Subset(test_ds,  split["indices"]["retain_test"])

    retain_train_loader = DataLoader(retain_train_ds, batch_size=hyperparams["batch_size"], shuffle=True,  num_workers=hyperparams["num_workers"])
    forget_train_loader = DataLoader(forget_train_ds, batch_size=hyperparams["batch_size"], shuffle=True,  num_workers=hyperparams["num_workers"])
    forget_test_loader  = DataLoader(forget_test_ds,  batch_size=256,                       shuffle=False, num_workers=hyperparams["num_workers"])
    retain_test_loader  = DataLoader(retain_test_ds,  batch_size=256,                       shuffle=False, num_workers=hyperparams["num_workers"])

    num_classes = get_num_classes(ds)

    student = create_model(hyperparams["backbone"], num_classes=num_classes)
    student.load_state_dict(torch.load(f"results/checkpoints/{ds}_model_before_best.pth", weights_only=True, map_location=device))
    student = student.to(device)

    teacher = create_model(hyperparams["backbone"], num_classes=num_classes)
    teacher.load_state_dict(torch.load(f"results/checkpoints/{ds}_model_before_best.pth", weights_only=True, map_location=device))
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = optim.SGD(student.parameters(), lr=hyperparams["lr"], momentum=hyperparams["momentum"], weight_decay=hyperparams["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*60}\n\t=== Baseline SCRUB ({ds.upper()}) ===\n{'='*60}\n")
    t0 = time.time()
    history = {"epoch": [], "train_loss": [], "retain_acc": [], "forget_acc": []}

    for ep in range(1, hyperparams["epochs"] + 1):
        running_loss, n = 0.0, 0

        if ep <= hyperparams["max_steps_epochs"]:
            student.train()
            pbar = tqdm(forget_train_loader, desc=f"[SCRUB MAX] epoch {ep:02d}/{hyperparams['epochs']}", leave=False)
            for x_f, _ in pbar:
                x_f = x_f.to(device)
                with torch.no_grad():
                    t_logits = teacher(x_f)
                s_logits = student(x_f)
                loss_max = -kd_loss(s_logits, t_logits, hyperparams["temperature"])

                optimizer.zero_grad(set_to_none=True)
                loss_max.backward()
                optimizer.step()

                bs = x_f.size(0)
                running_loss += loss_max.item() * bs
                n += bs
                pbar.set_postfix(loss=running_loss / n)

        student.train()
        pbar = tqdm(retain_train_loader, desc=f"[SCRUB MIN] epoch {ep:02d}/{hyperparams['epochs']}", leave=False)
        for x_r, y_r in pbar:
            x_r, y_r = x_r.to(device), y_r.to(device)
            with torch.no_grad():
                t_logits = teacher(x_r)
            s_logits = student(x_r)
            loss_min = kd_loss(s_logits, t_logits, hyperparams["temperature"]) + hyperparams["gamma"] * criterion(s_logits, y_r)

            optimizer.zero_grad(set_to_none=True)
            loss_min.backward()
            optimizer.step()

            bs = x_r.size(0)
            running_loss += loss_min.item() * bs
            n += bs
            pbar.set_postfix(loss=running_loss / n)

        metrics = evaluate(student, {"retain": retain_test_loader, "forget": forget_test_loader}, device)
        metrics = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}
        history["epoch"].append(ep)
        history["train_loss"].append(running_loss / n)
        history["retain_acc"].append(metrics["retain_acc"])
        history["forget_acc"].append(metrics["forget_acc"])
        phase = "MAX+MIN" if ep <= hyperparams["max_steps_epochs"] else "MIN"
        print(f"[SCRUB {phase}] epoch {ep:02d}/{hyperparams['epochs']} | loss {running_loss/n:.4f} | retain_acc {metrics['retain_acc']:.4f} | forget_acc {metrics['forget_acc']:.4f} | elapsed {time.time()-t0:.1f}s")

    history["elapsed_sec"] = time.time() - t0
    history["hyperparams"] = hyperparams
    torch.save(student.state_dict(), f"results/checkpoints/{ds}_baseline_scrub{seed_suffix}.pth")
    with open(f"results/reports/{ds}_baseline_scrub{seed_suffix}_curve.json", "w") as f:
        js.dump(history, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["retain_acc"], label="retain_acc")
    plt.plot(history["epoch"], history["forget_acc"], label="forget_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"Baseline SCRUB ({ds.upper()}) — Retain vs Forget Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/plots/{ds}_baseline_scrub{seed_suffix}_acc.png", dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: results/plots/{ds}_baseline_scrub{seed_suffix}_acc.png")


if __name__ == "__main__":
    baseline_scrub()
