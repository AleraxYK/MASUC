import os
import time
import json
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import get_device
from src.models import create_model
from src.data import get_dataset, get_num_classes
from src.eval.metrics import evaluate



def train_model():
    """
    Train a ResNet model on CIFAR-10 (before unlearning), save checkpoints, reports and plots.

    Saves:
        - results/checkpoints/model_before_latest.pth (every epoch)
        - results/checkpoints/model_before_best.pth   (best test accuracy)
        - results/checkpoints/model_trained_before_unlearning.pth (final)
        - results/reports/train_curve.json
        - results/reports/before.json
        - results/reports/before_final.json
        - results/plots/train_loss_before.png
        - results/plots/test_acc_before.png
    """

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None, help="Override default epochs")
    args = parser.parse_args()
    ds = args.dataset

    backbone    = "resnet18"
    epoch       = args.epochs if args.epochs is not None else (50 if ds == "tinyimagenet" else 30)
    batch_size  = 64
    lr          = 0.1
    momentum    = 0.9
    weight_decay= 5e-4
    num_workers = 2
    history     = {"epoch": [], "train_loss": [], "test_acc": []}
    best_acc = -1.0

    device = get_device()

    print(f"{device} in using")

    train_ds = get_dataset(ds, "./data", train=True)
    test_ds  = get_dataset(ds, "./data", train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=num_workers)

    model = create_model(backbone=backbone, num_classes=get_num_classes(ds)).to(device)
    print(f"{backbone} model")

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    before = evaluate(model, {"test_all": test_loader}, device)

    before = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in before.items()}
    print(f"Before: {before}")

    start_time = time.time()

    for ep in range(1, epoch + 1):
        model.train()
        running_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"[train model] epoch {ep:02d} / {epoch}", leave=False)

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

        after = evaluate(model, {"test_all": test_loader}, device)
        after = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in after.items()}

        elapsed = time.time() - start_time
        print(f"[train_before] epoch {ep:02d}/{epoch} | loss {running_loss/n:.4f} | {after} | elapsed {elapsed:.1f}s")

        history["epoch"].append(ep)
        history["train_loss"].append(running_loss / n)
        history["test_acc"].append(after["test_all_acc"])

        with open(f"results/reports/{ds}_train_curve.json", "w") as f:
            json.dump(history, f, indent=2)

        torch.save(model.state_dict(), f"results/checkpoints/{ds}_model_before_latest.pth")

        if after["test_all_acc"] > best_acc:
            best_acc = after["test_all_acc"]
            torch.save(model.state_dict(), f"results/checkpoints/{ds}_model_before_best.pth")


    ckpt_path = f"results/checkpoints/{ds}_model_trained_before_unlearning.pth"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint:", ckpt_path)            


    report = {
        "backbone": backbone,
        "epoch": epoch,
        "batch_size": batch_size,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "device": str(device),
        "final": after,
    }

    report_path = "results/reports/before.json"
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    history_path = "results/reports/train_curve.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print("Saved history:", history_path)

    report["best_test_acc"] = best_acc
    with open("results/reports/before_final.json", "w") as f:
        json.dump(report, f, indent=2)


    plt.figure(figsize=(6.5, 4.0))
    plt.plot(history["epoch"], history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/train_loss_before.png", dpi=400, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(history["epoch"], history["test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/test_acc_before.png", dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved report: {report_path}\nSaved plots in results/plots")
    



if __name__ == "__main__":
    train_model()