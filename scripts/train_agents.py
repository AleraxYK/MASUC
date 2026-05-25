import os
import json
import time
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import get_device
from src.models import create_model
from src.data import get_dataset, get_num_classes
from src.data.subset import filter_by_classes, filter_out_classes
from src.eval.metrics import evaluate


def train_single_teacher(teacher_id, blind, base_ckpt, train_ds, test_ds, hyperparams, device, num_classes: int = 10, ds: str = "cifar10") -> dict[str, float]:
    """
    Train one teacher starting from a base checkpoint, excluding its blind class.

    Args:
        teacher_id (int): The ID of the teacher.
        blind (int): The blind class of the teacher.
        base_ckpt (str): The path to the base checkpoint.
        train_ds (Dataset): The training dataset.
        test_ds (Dataset): The test dataset.
        hyperparams (dict): The hyperparameters for training.
        device (torch.device): The device to use for training.
    
    Returns:
        dict: A dictionary containing the training report.
    """

    train_seen = filter_out_classes(train_ds, [blind])
    test_seen  = filter_out_classes(test_ds, [blind])
    test_blind = filter_by_classes(test_ds, [blind])

    train_loader = DataLoader(
        train_seen,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=hyperparams["num_workers"],
    )
    test_seen_loader = DataLoader(test_seen, batch_size=256, shuffle=False, num_workers=hyperparams["num_workers"])
    test_blind_loader = DataLoader(test_blind, batch_size=256, shuffle=False, num_workers=hyperparams["num_workers"])

    model = create_model(backbone=hyperparams["backbone"], num_classes=num_classes).to(device)
    state = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=hyperparams["lr"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"],
    )

    start_time = time.time()
    history = {"epoch": [], "train_loss": [], "seen_acc": [], "blind_acc": []}

    for ep in range(1, hyperparams["epochs"] + 1):
        model.train()
        running_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"[teacher {teacher_id}] epoch {ep:02d}/{hyperparams['epochs']}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * y.size(0)
            n += y.size(0)
            pbar.set_postfix(loss=running_loss / n)

        metrics = evaluate(
            model,
            {
                "seen": test_seen_loader,
                "blind": test_blind_loader,
            },
            device,
        )
        metrics = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}

        history["epoch"].append(ep)
        history["train_loss"].append(running_loss / n)
        history["seen_acc"].append(metrics["seen_acc"])
        history["blind_acc"].append(metrics["blind_acc"])

        elapsed = time.time() - start_time
        print(
            f"[teacher {teacher_id}] epoch {ep:02d}/{hyperparams['epochs']} "
            f"| loss {running_loss/n:.4f} | seen_acc {metrics['seen_acc']:.4f} | blind_acc {metrics['blind_acc']:.4f} "
            f"| elapsed {elapsed:.1f}s"
        )

    elapsed = time.time() - start_time

    curve_path     = f"results/reports/{ds}_teacher_{teacher_id}_curve.json"
    loss_plot_path = f"results/plots/{ds}_teacher_{teacher_id}_loss.png"
    acc_plot_path  = f"results/plots/{ds}_teacher_{teacher_id}_acc.png"
    ckpt_path      = f"results/checkpoints/{ds}_teacher_{teacher_id}.pth"
    with open(curve_path, "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(history["epoch"], history["train_loss"])
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=400, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(history["epoch"], history["seen_acc"], label="seen_acc")
    plt.plot(history["epoch"], history["blind_acc"], label="blind_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=400, bbox_inches="tight"); plt.close()

    torch.save(model.state_dict(), ckpt_path)

    report = {
        "teacher_id": teacher_id,
        "blind_class": int(blind),
        "hyperparams": {
            "backbone": hyperparams["backbone"],
            "epochs": hyperparams["epochs"],
            "batch_size": hyperparams["batch_size"],
            "lr": hyperparams["lr"],
            "momentum": hyperparams["momentum"],
            "weight_decay": hyperparams["weight_decay"],
        },
        "metrics_final": metrics,
        "elapsed_sec": elapsed,
        "ckpt_path": ckpt_path,
        "curve_path": curve_path,
        "loss_plot_path": loss_plot_path,
        "acc_plot_path": acc_plot_path,
    }

    report_path = f"results/reports/{ds}_teacher_{teacher_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved teacher checkpoint: {ckpt_path}")
    print(f"Saved teacher report: {report_path}")

    return report



def train_multiple_agent(smoke_test: bool = False) -> dict[int, dict]:
    """
    Train multiple agents on CIFAR-10.

    Args:
        smoke_test (bool): Whether to run a smoke test.
    
    Returns:
        dict: A dictionary containing the training reports.
    """

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--epochs",  type=int, default=None, help="Override default epochs (default: 5)")
    args = parser.parse_args()
    ds = args.dataset

    device = get_device()
    print("Using device:", device)

    teacher_blind = {
    0: 3,
    1: 7,
    2: 1,
    3: 5,
    4: 9,
    }

    backbone = "resnet18"
    epochs = 5
    batch_size = 64
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    num_workers = 4

    train_ds = get_dataset(ds, "./data", train=True)
    test_ds  = get_dataset(ds, "./data", train=False)

    base_ckpt = f"results/checkpoints/{ds}_model_before_best.pth"
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Missing base checkpoint: {base_ckpt}")
    print("Base checkpoint:", base_ckpt)

    for tid, blind in teacher_blind.items():
        train_seen = filter_out_classes(train_ds, [blind])
        test_seen  = filter_out_classes(test_ds, [blind])
        test_blind = filter_by_classes(test_ds, [blind])

        print(
            f"teacher {tid} blind={blind} | "
            f"train_seen={len(train_seen)} | test_seen={len(test_seen)} | test_blind={len(test_blind)}"
        )
    
    hyperparams = {
    "backbone": backbone,
    "epochs": args.epochs if args.epochs is not None else (2 if smoke_test else 5),
    "batch_size": batch_size,
    "lr": lr,
    "momentum": momentum,
    "weight_decay": weight_decay,
    "num_workers": num_workers,
    }

    for tid, blind in teacher_blind.items():
        print(f"\n{'='*60}\n\t=== training teacher {tid} (blind={blind}) ===\n{'='*60}\n")
        train_single_teacher(
            teacher_id=tid,
            blind=blind,
            base_ckpt=base_ckpt,
            train_ds=train_ds,
            test_ds=test_ds,
            hyperparams=hyperparams,
            device=device,
            num_classes=get_num_classes(ds),
            ds=ds,
        )



if __name__ == "__main__":
    train_multiple_agent()