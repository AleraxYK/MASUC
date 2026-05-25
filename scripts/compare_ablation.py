import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


ABLATIONS = [
    {"suffix": "",            "label": "Full MASUC"},
    {"suffix": "_no_ea",      "label": "w/o Energy Alignment"},
    {"suffix": "_no_kd",      "label": "w/o Knowledge Distillation"},
    {"suffix": "_no_ra",      "label": "w/o Reciprocal Altruism"},
    {"suffix": "_no_erasure", "label": "w/o Erasure Loss"},
]


def load_config_results(ds: str, suffix: str, seeds: list[int] | None) -> dict:
    """
    Collect final-epoch retain_acc / forget_acc across seeds for one config.

    If seeds is provided, looks for exact seed files and warns on missing ones.
    If seeds is None, auto-detects via glob.
    Falls back to a single legacy file (no seed in name) if nothing is found.
    """
    retain_vals, forget_vals, found_seeds = [], [], []

    if seeds:
        for seed in seeds:
            path = f"results/reports/{ds}_masuc{suffix}_seed{seed}_curve.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                retain_vals.append(data["retain_acc"][-1])
                forget_vals.append(data["forget_acc"][-1])
                found_seeds.append(seed)
            else:
                print(f"  [!] Missing: {path}")
    else:
        for path in sorted(glob.glob(f"results/reports/{ds}_masuc{suffix}_seed*_curve.json")):
            with open(path) as f:
                data = json.load(f)
            retain_vals.append(data["retain_acc"][-1])
            forget_vals.append(data["forget_acc"][-1])
            found_seeds.append(os.path.basename(path))

    # Legacy fallback: single-seed file with no seed suffix
    if not retain_vals:
        path = f"results/reports/{ds}_masuc{suffix}_curve.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            retain_vals.append(data["retain_acc"][-1])
            forget_vals.append(data["forget_acc"][-1])
            found_seeds.append("legacy")

    return {"retain_acc": retain_vals, "forget_acc": forget_vals, "seeds": found_seeds}


def compute_stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1))


def compare_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--seeds",   type=int, nargs="+", default=None,
                        help="Seeds to aggregate (default: auto-detect from files)")
    args = parser.parse_args()
    ds = args.dataset

    os.makedirs("results/plots",     exist_ok=True)
    os.makedirs("results/summaries", exist_ok=True)

    results = []
    for ab in ABLATIONS:
        raw = load_config_results(ds, ab["suffix"], args.seeds)
        if not raw["retain_acc"]:
            print(f"  [!] No results for '{ab['label']}' — skipping.")
            continue
        r_mean, r_std = compute_stats(raw["retain_acc"])
        f_mean, f_std = compute_stats(raw["forget_acc"])
        results.append({
            "label":       ab["label"],
            "retain_mean": r_mean,
            "retain_std":  r_std,
            "forget_mean": f_mean,
            "forget_std":  f_std,
            "n_seeds":     len(raw["retain_acc"]),
            "raw":         raw,
        })

    if not results:
        print("No ablation results found. Run scripts.run_ablation first.")
        return

    # --- JSON summary ---
    summary = [
        {k: v for k, v in r.items() if k != "raw"}
        for r in results
    ]
    json_path = f"results/summaries/{ds}_ablation_stats.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary → {json_path}")

    # --- Markdown table ---
    n_seeds    = results[0]["n_seeds"]
    multi_seed = n_seeds > 1
    ref_forget = results[0]["forget_mean"]

    md_lines = [
        f"# MASUC Ablation Study — {ds.upper()}\n",
        f"Results over {n_seeds} seed(s). Format: mean ± std (sample std, ddof=1).\n"
        if multi_seed else
        f"Results from a single run (no multi-seed stats available).\n",
        "∆ Forget = forget_acc relative to Full MASUC "
        "(positive = worse forgetting, negative = better).\n",
        "| Configuration | n | Retain Acc | Forget Acc | ∆ Forget |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]

    for r in results:
        if multi_seed:
            retain_str = f"{r['retain_mean']:.2%} ± {r['retain_std']:.2%}"
            forget_str = f"{r['forget_mean']:.2%} ± {r['forget_std']:.2%}"
        else:
            retain_str = f"{r['retain_mean']:.2%}"
            forget_str = f"{r['forget_mean']:.2%}"

        delta     = r["forget_mean"] - ref_forget
        delta_str = "—" if r["label"] == "Full MASUC" else f"{delta:+.2%}"

        md_lines.append(
            f"| **{r['label']}** | {r['n_seeds']} "
            f"| {retain_str} | {forget_str} | {delta_str} |"
        )

    md_text = "\n".join(md_lines)
    md_path = f"results/summaries/{ds}_ablation_stats.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"Markdown table → {md_path}")
    print(f"\n{md_text}\n")

    # --- Bar plot with error bars ---
    labels       = [r["label"] for r in results]
    retain_means = np.array([r["retain_mean"] for r in results])
    retain_stds  = np.array([r["retain_std"]  for r in results])
    forget_means = np.array([r["forget_mean"] for r in results])
    forget_stds  = np.array([r["forget_std"]  for r in results])

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, retain_means, width,
                   yerr=retain_stds if multi_seed else None,
                   capsize=4, label="Retain Acc", color="teal", alpha=0.85)
    bars2 = ax.bar(x + width / 2, forget_means, width,
                   yerr=forget_stds if multi_seed else None,
                   capsize=4, label="Forget Acc", color="crimson", alpha=0.85)

    all_bars   = list(bars1) + list(bars2)
    all_means  = list(retain_means) + list(forget_means)
    all_stds   = list(retain_stds)  + list(forget_stds)
    for bar, mean, std in zip(all_bars, all_means, all_stds):
        label_str = f"{mean:.1%}\n±{std:.1%}" if (multi_seed and std > 0) else f"{mean:.1%}"
        ax.annotate(label_str,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    n_label = f"(n={n_seeds} seeds)" if multi_seed else "(single run)"
    ax.set_ylabel("Accuracy")
    ax.set_title(f"MASUC Ablation Study — {ds.upper()} {n_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"results/plots/{ds}_ablation_bar.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Plot → {out}")


if __name__ == "__main__":
    compare_ablation()
