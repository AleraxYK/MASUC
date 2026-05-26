"""
Parallel MASUC variant — CUDA-only.

The 5 teacher Reciprocal-Altruism updates are independent (each teacher has
its own parameters, optimizer, and energy buffer) so on a CUDA GPU we can
launch them on separate CUDA Streams and let the scheduler overlap the
kernels.

IMPORTANT:
  * No-op on MPS / CPU (no streams) — the script falls back to sequential.
  * Each teacher must own its own DataLoader iterator (num_workers=0) so
    that threads do not share a multi-worker iterator (unsafe).
  * Each teacher owns its own EnergyRunningMean state (separate buffers).
  * The student is in eval mode during RA, so concurrent student forwards
    are read-only — safe to share.

Use only when you've verified the sequential version (run_masuc.py) is
already optimized; the parallel variant is for benchmarking the wall-clock
speedup of stream overlap, NOT for primary multi-seed runs.
"""

import os
import time
import argparse
import json as js
import contextlib
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data import get_dataset, get_num_classes
from src.models import create_model
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.methods.masuc.train import collaborative_unlearning, reciprocal_altruism
from src.methods.masuc.utils import EnergyRunningMean


def run_masuc_parallel():
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/reports",     exist_ok=True)
    os.makedirs("results/plots",       exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class",type=int, default=3)
    parser.add_argument("--no_ea",      action="store_true")
    parser.add_argument("--no_kd",      action="store_true")
    parser.add_argument("--no_ra",      action="store_true")
    parser.add_argument("--no_erasure", action="store_true")
    parser.add_argument("--lambda_1",   type=float, default=None)
    parser.add_argument("--lambda_2",   type=float, default=None)
    parser.add_argument("--lambda_3",   type=float, default=None)
    parser.add_argument("--lr_student", type=float, default=None)
    parser.add_argument("--lr_teacher", type=float, default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--temperature",type=float, default=None)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--run_id",     type=str,   default=None)
    parser.add_argument("--amp",        action="store_true")
    args = parser.parse_args()
    ds = args.dataset

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Seed: {args.seed}")

    torch.backends.cudnn.benchmark = True

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"{ds}_masuc_parallel"
        if args.no_kd:      run_id += "_no_kd"
        if args.no_ea:      run_id += "_no_ea"
        if args.no_ra:      run_id += "_no_ra"
        if args.no_erasure: run_id += "_no_erasure"
        if args.seed is not None: run_id += f"_seed{args.seed}"
    print(f"Run ID: {run_id}")

    device = get_device()
    print(f"Using device: {device} | Dataset: {ds.upper()}")

    if device.type != "cuda":
        print("[!] Parallel variant only benefits CUDA. Falling back to sequential teacher loop.")

    hyperparams = {
        "backbone":     "resnet18",
        "epochs":       args.epochs      if args.epochs      is not None else 5,
        "batch_size":   128,
        "lr_student":   args.lr_student  if args.lr_student  is not None else 0.001,
        "lr_teacher":   args.lr_teacher  if args.lr_teacher  is not None else 0.0001,
        "momentum":     0.9,
        "weight_decay": 5e-4,
        "lambda_1":     (args.lambda_1   if args.lambda_1    is not None else (0.0 if args.no_kd      else 1.0)),
        "lambda_2":     (args.lambda_2   if args.lambda_2    is not None else (0.0 if args.no_ea      else 0.1)),
        "lambda_3":     (args.lambda_3   if args.lambda_3    is not None else (0.0 if args.no_erasure else 0.5)),
        "temperature":  args.temperature if args.temperature is not None else 2.0,
        "forget_class": args.forget_class,
        "seed":         args.seed,
        "amp":          args.amp,
    }

    train_ds = get_dataset(ds, "./data", train=True)
    test_ds  = get_dataset(ds, "./data", train=False)

    split_path = f"results/splits/{ds}_split_forget_{hyperparams['forget_class']}.json"
    with open(split_path) as f:
        split = js.load(f)

    forget_train_ds = Subset(train_ds, split["indices"]["forget_train"])
    forget_test_ds  = Subset(test_ds,  split["indices"]["forget_test"])
    retain_test_ds  = Subset(test_ds,  split["indices"]["retain_test"])

    pin = (device.type == "cuda")

    forget_test_loader = DataLoader(forget_test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=pin)
    retain_test_loader = DataLoader(retain_test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=pin)

    # One DataLoader per teacher with num_workers=0 for thread-safe iteration
    # (you cannot share a multi-worker DataLoader across Python threads).
    teacher_ra_loaders = [
        DataLoader(
            forget_train_ds,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
        )
        for _ in range(5)
    ]
    # Student CU uses a regular multi-worker loader (no thread sharing here).
    forget_train_loader_main = DataLoader(
        forget_train_ds,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=pin,
        persistent_workers=True,
    )

    print("Loading models...")
    num_classes = get_num_classes(ds)
    student = create_model(hyperparams["backbone"], num_classes=num_classes)
    student.load_state_dict(torch.load(f"results/checkpoints/{ds}_model_before_best.pth", weights_only=True, map_location=device))
    student = student.to(device)

    teachers, teacher_opts, teacher_streams, teacher_energy = {}, {}, {}, {}
    for i in range(5):
        t = create_model(hyperparams["backbone"], num_classes=num_classes)
        t.load_state_dict(torch.load(f"results/checkpoints/{ds}_teacher_{i}.pth", weights_only=True, map_location=device))
        t = t.to(device)
        teachers[i] = t
        teacher_opts[i] = optim.SGD(t.parameters(), lr=hyperparams["lr_teacher"], momentum=hyperparams["momentum"], weight_decay=hyperparams["weight_decay"])
        teacher_streams[i] = torch.cuda.Stream() if device.type == "cuda" else None
        teacher_energy[i] = EnergyRunningMean(device=device)

    student_optimizer = optim.SGD(
        student.parameters(),
        lr=hyperparams["lr_student"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"],
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("AMP (mixed precision) ENABLED")

    history = {"epoch": [], "train_loss": [], "retain_acc": [], "forget_acc": []}

    print(f"\n{'='*60}\n\t=== MASUC Parallel Unlearning ===\n{'='*60}\n")
    t0_total = time.time()

    for ep in range(1, hyperparams["epochs"] + 1):
        print(f"\n--- Epoch {ep}/{hyperparams['epochs']} ---")

        if not args.no_ra:
            def _run_one_teacher(tid: int) -> None:
                stream = teacher_streams[tid]
                ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with ctx:
                    teacher_energy[tid] = reciprocal_altruism(
                        ep=ep,
                        num_epochs=hyperparams["epochs"],
                        teacher_id=tid,
                        teacher_model=teachers[tid],
                        student_model=student,
                        forget_train_loader=teacher_ra_loaders[tid],
                        forget_class=hyperparams["forget_class"],
                        optimizer=teacher_opts[tid],
                        initial_lambda_1=hyperparams["lambda_1"],
                        lambda_2=hyperparams["lambda_2"],
                        temperature=hyperparams["temperature"],
                        device=device,
                        training_energies_target=teacher_energy[tid],
                        use_amp=use_amp,
                        scaler=scaler,
                    )

            if device.type == "cuda":
                with ThreadPoolExecutor(max_workers=len(teachers)) as pool:
                    futures = [pool.submit(_run_one_teacher, tid) for tid in teachers]
                    for f in futures:
                        f.result()
                torch.cuda.synchronize()
            else:
                # No streams on MPS/CPU → sequential fallback (same as run_masuc.py)
                for tid in teachers:
                    _run_one_teacher(tid)

        metrics, _ = collaborative_unlearning(
            ep=ep,
            num_epochs=hyperparams["epochs"],
            student_model=student,
            teacher_models=teachers,
            forget_train_loader=forget_train_loader_main,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            optimizer=student_optimizer,
            forget_class=hyperparams["forget_class"],
            initial_lambda_1=hyperparams["lambda_1"],
            lambda_2=hyperparams["lambda_2"],
            lambda_3=hyperparams["lambda_3"],
            temperature=hyperparams["temperature"],
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )

        history["epoch"].append(ep)
        history["train_loss"].append(metrics["train_loss"])
        history["retain_acc"].append(metrics["retain_acc"])
        history["forget_acc"].append(metrics["forget_acc"])

    total_elapsed = time.time() - t0_total
    print(f"\nMASUC PARALLEL Finished! Total elapsed: {total_elapsed:.1f}s")

    history["elapsed_sec"] = total_elapsed
    history["hyperparams"] = hyperparams

    torch.save(student.state_dict(), f"results/checkpoints/{run_id}_final.pth")
    with open(f"results/reports/{run_id}_curve.json", "w") as f:
        js.dump(history, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["retain_acc"], label="retain_acc")
    plt.plot(history["epoch"], history["forget_acc"], label="forget_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"{run_id.upper()} — Retain vs Forget Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"results/plots/{run_id}_acc.png", dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: results/plots/{run_id}_acc.png")


if __name__ == "__main__":
    run_masuc_parallel()
