import os
import re
import csv
import glob
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.data import get_dataset, get_num_classes
from src.models import create_model
from src.eval.mia import mia_evaluate
from src.utils.device import get_device


# Method registry: filename templates relative to results/checkpoints/
# Placeholders: {ds}, {suffix}  (suffix is "" or "_seed{N}")
METHODS = [
    {"key": "before",            "label": "Before Unlearning",  "template": "{ds}_model_before_best",            "has_seed": False},
    {"key": "ft",                "label": "Fine-Tuning",        "template": "{ds}_baseline_ft{suffix}",          "has_seed": True},
    {"key": "neggradplus",       "label": "NegGrad+",           "template": "{ds}_baseline_neggradplus{suffix}", "has_seed": True},
    {"key": "random_labels",     "label": "Random Labels",      "template": "{ds}_baseline_random_labels{suffix}","has_seed": True},
    {"key": "bad_teaching",      "label": "Bad-Teaching",       "template": "{ds}_baseline_bad_teaching{suffix}","has_seed": True},
    {"key": "scrub",             "label": "SCRUB",              "template": "{ds}_baseline_scrub{suffix}",       "has_seed": True},
    {"key": "masuc",             "label": "MASUC (Ours)",       "template": "{ds}_masuc{suffix}_final",          "has_seed": True},
    {"key": "masuc_no_ea",       "label": "MASUC w/o EA",       "template": "{ds}_masuc_no_ea{suffix}_final",    "has_seed": True},
    {"key": "masuc_no_kd",       "label": "MASUC w/o KD",       "template": "{ds}_masuc_no_kd{suffix}_final",    "has_seed": True},
    {"key": "masuc_no_ra",       "label": "MASUC w/o RA",       "template": "{ds}_masuc_no_ra{suffix}_final",    "has_seed": True},
    {"key": "masuc_no_erasure",  "label": "MASUC w/o Erasure",  "template": "{ds}_masuc_no_erasure{suffix}_final","has_seed": True},
    {"key": "retrain",           "label": "Retrain (Gold)",     "template": "{ds}_baseline_retrain{suffix}",     "has_seed": True},
]


def find_checkpoints(method: dict, ds: str, seeds: list[int] | None) -> list[tuple[int | None, str]]:
    """Return [(seed, ckpt_path)] for this method. Falls back to legacy single-file if no seed files found."""
    if not method["has_seed"]:
        path = f"results/checkpoints/{method['template'].format(ds=ds, suffix='')}.pth"
        return [(None, path)] if os.path.exists(path) else []

    found = []
    if seeds:
        for s in seeds:
            p = f"results/checkpoints/{method['template'].format(ds=ds, suffix=f'_seed{s}')}.pth"
            if os.path.exists(p):
                found.append((s, p))
    else:
        glob_pat = f"results/checkpoints/{method['template'].format(ds=ds, suffix='_seed*')}.pth"
        for p in sorted(glob.glob(glob_pat)):
            m = re.search(r"_seed(\d+)", os.path.basename(p))
            seed = int(m.group(1)) if m else None
            found.append((seed, p))

    if not found:
        legacy = f"results/checkpoints/{method['template'].format(ds=ds, suffix='')}.pth"
        if os.path.exists(legacy):
            found.append((None, legacy))

    return found


def stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1))


def eval_mia():
    """
    For every method × seed checkpoint found under results/checkpoints/, run a
    Membership Inference Attack distinguishing forget_train (members, were in
    the original training set) from forget_test (non-members, never seen).

    Reports AUC for loss-/confidence-/entropy-based attacks. AUC ≈ 0.5 means no
    membership signal remains (good privacy); AUC > 0.5 indicates leakage.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str,   default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int,   default=3)
    parser.add_argument("--seeds",        type=int,   nargs="+", default=None,
                        help="Seeds to evaluate (default: auto-detect from filenames)")
    parser.add_argument("--methods",      type=str,   nargs="+", default=None,
                        help="Subset of methods to evaluate (default: all). Choices: " + ", ".join(m["key"] for m in METHODS))
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--num_workers",  type=int,   default=4)
    args = parser.parse_args()

    ds = args.dataset
    fc = args.forget_class

    os.makedirs("results/summaries", exist_ok=True)
    os.makedirs("results/plots",     exist_ok=True)
    os.makedirs("results/reports",   exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | Dataset: {ds.upper()} | Forget class: {fc}")

    train_ds = get_dataset(ds, "./data", train=True)
    test_ds  = get_dataset(ds, "./data", train=False)

    split_path = f"results/splits/{ds}_split_forget_{fc}.json"
    with open(split_path) as f:
        split = json.load(f)

    members_ds    = Subset(train_ds, split["indices"]["forget_train"])
    nonmembers_ds = Subset(test_ds,  split["indices"]["forget_test"])

    members_loader    = DataLoader(members_ds,    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    nonmembers_loader = DataLoader(nonmembers_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"  Members (forget_train):     {len(members_ds)}")
    print(f"  Non-members (forget_test):  {len(nonmembers_ds)}")

    num_classes = get_num_classes(ds)
    methods = [m for m in METHODS if (args.methods is None or m["key"] in args.methods)]

    rows = []
    for method in methods:
        ckpts = find_checkpoints(method, ds, args.seeds)
        if not ckpts:
            print(f"  [!] No checkpoint for '{method['label']}' — skipping.")
            continue

        print(f"\n=== {method['label']} ({len(ckpts)} ckpt) ===")
        per_seed = []
        for seed, ckpt_path in ckpts:
            model = create_model("resnet18", num_classes=num_classes)
            model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
            model = model.to(device)
            result = mia_evaluate(model, members_loader, nonmembers_loader, device)
            result["seed"]      = seed
            result["ckpt_path"] = ckpt_path
            per_seed.append(result)
            tag = f"seed={seed}" if seed is not None else "single"
            print(f"  [{tag}] AUC loss={result['auc_loss']:.4f} | conf={result['auc_conf']:.4f} | entropy={result['auc_entropy']:.4f}")

        auc_loss_m, auc_loss_s = stats([r["auc_loss"]    for r in per_seed])
        auc_conf_m, auc_conf_s = stats([r["auc_conf"]    for r in per_seed])
        auc_ent_m,  auc_ent_s  = stats([r["auc_entropy"] for r in per_seed])
        adv_loss_m, adv_loss_s = stats([r["adv_loss"]    for r in per_seed])

        rows.append({
            "label":        method["label"],
            "key":          method["key"],
            "n_seeds":      len(per_seed),
            "auc_loss_m":   auc_loss_m, "auc_loss_s":   auc_loss_s,
            "auc_conf_m":   auc_conf_m, "auc_conf_s":   auc_conf_s,
            "auc_ent_m":    auc_ent_m,  "auc_ent_s":    auc_ent_s,
            "adv_loss_m":   adv_loss_m, "adv_loss_s":   adv_loss_s,
            "per_seed":     per_seed,
        })

    if not rows:
        print("\nNo methods evaluated. Make sure checkpoints exist.")
        return

    # --- JSON ---
    json_path = f"results/summaries/{ds}_mia_stats.json"
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "per_seed"} for r in rows], f, indent=2)
    print(f"\nJSON  → {json_path}")

    # --- CSV ---
    csv_path = f"results/summaries/{ds}_mia_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "key", "n_seeds",
            "auc_loss_m", "auc_loss_s",
            "auc_conf_m", "auc_conf_s",
            "auc_ent_m",  "auc_ent_s",
            "adv_loss_m", "adv_loss_s",
        ])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"CSV   → {csv_path}")

    # --- Markdown ---
    any_multi = any(r["n_seeds"] > 1 for r in rows)
    md = [
        f"# Membership Inference Attack — {ds.upper()} (forget class {fc})\n",
        "Members = forget_train (in original training set). "
        "Non-members = forget_test (held out, never seen).\n",
        "AUC ≈ 0.5 → no recoverable membership signal (ideal privacy). "
        "AUC > 0.5 → leakage.\n",
        "Format: mean ± std over multiple seeds.\n" if any_multi else "",
        "| Method | n | AUC (loss) | AUC (conf) | AUC (entropy) | Best-thr Acc (loss) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        def fmt(m, s):
            return f"{m:.4f} ± {s:.4f}" if (any_multi and s > 0) else f"{m:.4f}"
        md.append(
            f"| **{r['label']}** | {r['n_seeds']} "
            f"| {fmt(r['auc_loss_m'], r['auc_loss_s'])} "
            f"| {fmt(r['auc_conf_m'], r['auc_conf_s'])} "
            f"| {fmt(r['auc_ent_m'],  r['auc_ent_s'])} "
            f"| {fmt(r['adv_loss_m'], r['adv_loss_s'])} |"
        )
    md_text = "\n".join(md)
    md_path = f"results/summaries/{ds}_mia_stats.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"MD    → {md_path}")
    print(f"\n{md_text}\n")

    # --- Bar plot: AUC (loss) per method ---
    labels = [r["label"] for r in rows]
    means  = np.array([r["auc_loss_m"] for r in rows])
    stds   = np.array([r["auc_loss_s"] for r in rows])

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(labels)), 5))
    bars = ax.bar(np.arange(len(labels)), means, yerr=stds if any_multi else None,
                  capsize=4, color="#4C72B0", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Ideal (no leak, AUC=0.5)")
    for bar, m, s in zip(bars, means, stds):
        label_str = f"{m:.3f}\n±{s:.3f}" if (any_multi and s > 0) else f"{m:.3f}"
        ax.annotate(label_str,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("MIA AUC (loss-based)")
    ax.set_title(f"Membership Inference Attack — {ds.upper()} (forget class {fc})")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=14, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"results/plots/{ds}_mia_auc.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Plot  → {out}")


if __name__ == "__main__":
    eval_mia()
