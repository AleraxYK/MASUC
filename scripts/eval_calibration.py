"""
Run calibration analysis across all available unlearned checkpoints.

For every method × seed pair found under results/checkpoints/, compute:
  * ECE on retain test set        (lower = better calibration)
  * Max-softmax confidence on forget set (lower = good unlearning;
    high = confident misclassification = pathological FT-style over-spec)
  * Max non-forget class confidence on forget inputs (quantifies whether
    the model confidently re-routes to a specific wrong class)

Outputs:
  results/summaries/{ds}_calibration_stats.{json,csv,md}
  results/plots/{ds}_calibration_bar.png
"""

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
from src.eval.calibration import calibration_report
from src.utils.device import get_device


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
        pat = f"results/checkpoints/{method['template'].format(ds=ds, suffix='_seed*')}.pth"
        for p in sorted(glob.glob(pat)):
            m = re.search(r"_seed(\d+)", os.path.basename(p))
            seed = int(m.group(1)) if m else None
            found.append((seed, p))

    if not found:
        legacy = f"results/checkpoints/{method['template'].format(ds=ds, suffix='')}.pth"
        if os.path.exists(legacy):
            found.append((None, legacy))
    return found


def stats(vals):
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int, default=3)
    parser.add_argument("--seeds",        type=int, nargs="+", default=None)
    parser.add_argument("--methods",      type=str, nargs="+", default=None)
    parser.add_argument("--batch_size",   type=int, default=256)
    args = parser.parse_args()

    ds = args.dataset
    fc = args.forget_class

    os.makedirs("results/summaries", exist_ok=True)
    os.makedirs("results/plots",     exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | Dataset: {ds.upper()} | Forget class: {fc}")

    test_ds  = get_dataset(ds, "./data", train=False)
    split_path = f"results/splits/{ds}_split_forget_{fc}.json"
    with open(split_path) as f:
        split = json.load(f)

    retain_ds = Subset(test_ds, split["indices"]["retain_test"])
    forget_ds = Subset(test_ds, split["indices"]["forget_test"])
    retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = get_num_classes(ds)
    methods = [m for m in METHODS if (args.methods is None or m["key"] in args.methods)]

    rows = []
    for method in methods:
        ckpts = find_checkpoints(method, ds, args.seeds)
        if not ckpts:
            print(f"  [skip] {method['label']}: no checkpoint")
            continue

        print(f"\n=== {method['label']} ({len(ckpts)} ckpt) ===")
        per_seed = []
        for seed, ckpt_path in ckpts:
            model = create_model("resnet18", num_classes=num_classes)
            model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
            model = model.to(device)
            r = calibration_report(model, retain_loader, forget_loader, device)
            r["seed"] = seed
            per_seed.append(r)
            tag = f"seed={seed}" if seed is not None else "single"
            print(f"  [{tag}] ECE={r['ece_retain']:.4f} | conf_retain={r['conf_retain']:.4f} | conf_forget={r['conf_forget']:.4f}")

        ece_m, ece_s   = stats([r["ece_retain"]  for r in per_seed])
        cret_m, cret_s = stats([r["conf_retain"] for r in per_seed])
        cfor_m, cfor_s = stats([r["conf_forget"] for r in per_seed])
        cothm, coths   = stats([r["max_other_class_conf_forget"] for r in per_seed])

        rows.append({
            "label":   method["label"],
            "key":     method["key"],
            "n_seeds": len(per_seed),
            "ece_m":   ece_m,   "ece_s":   ece_s,
            "conf_retain_m": cret_m, "conf_retain_s": cret_s,
            "conf_forget_m": cfor_m, "conf_forget_s": cfor_s,
            "max_other_m":   cothm,  "max_other_s":   coths,
        })

    if not rows:
        print("\nNo checkpoints evaluated.")
        return

    # JSON
    json_path = f"results/summaries/{ds}_calibration_stats.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nJSON  → {json_path}")

    # CSV
    csv_path = f"results/summaries/{ds}_calibration_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV   → {csv_path}")

    # MD
    any_multi = any(r["n_seeds"] > 1 for r in rows)
    md = [
        f"# Calibration Analysis — {ds.upper()} (forget class {fc})\n",
        "Lower ECE = better-calibrated retain predictions.",
        "Lower conf_forget = more uncertain (better) predictions on forget class.",
        "High conf_forget on a method with low forget accuracy indicates the",
        "model confidently MISpredicts the forget class — a pathological",
        "side-effect of aggressive over-specialization.\n",
        "| Method | n | ECE (retain) | Conf (retain) | Conf (forget) | Max-other-class conf (forget) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        def fmt(m, s):
            return f"{m:.4f} ± {s:.4f}" if (any_multi and s > 0) else f"{m:.4f}"
        md.append(
            f"| **{r['label']}** | {r['n_seeds']} "
            f"| {fmt(r['ece_m'], r['ece_s'])} "
            f"| {fmt(r['conf_retain_m'], r['conf_retain_s'])} "
            f"| {fmt(r['conf_forget_m'], r['conf_forget_s'])} "
            f"| {fmt(r['max_other_m'], r['max_other_s'])} |"
        )
    md_text = "\n".join(md)
    md_path = f"results/summaries/{ds}_calibration_stats.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"MD    → {md_path}")
    print(f"\n{md_text}\n")

    # Bar plot
    labels = [r["label"] for r in rows]
    ece_m = np.array([r["ece_m"] for r in rows])
    ece_s = np.array([r["ece_s"] for r in rows])
    cfor_m = np.array([r["conf_forget_m"] for r in rows])
    cfor_s = np.array([r["conf_forget_s"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(max(12, 1.3 * len(labels)), 5))

    ax[0].bar(range(len(labels)), ece_m, yerr=ece_s if any_multi else None, capsize=4, color="#3F51B5", alpha=0.85)
    ax[0].set_ylabel("ECE (retain)")
    ax[0].set_title("Expected Calibration Error (lower = better)")
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, rotation=18, ha="right")
    ax[0].grid(True, axis="y", alpha=0.3)

    ax[1].bar(range(len(labels)), cfor_m, yerr=cfor_s if any_multi else None, capsize=4, color="#E91E63", alpha=0.85)
    ax[1].set_ylabel("Max-softmax confidence on forget set")
    ax[1].set_title("Forget-set confidence (lower = healthier unlearning)")
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, rotation=18, ha="right")
    ax[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = f"results/plots/{ds}_calibration_bar.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Plot  → {out}")


if __name__ == "__main__":
    main()
