import os
import glob
import json
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt


METHODS = [
    {"key": "before",         "label": "Before Unlearning",     "report": None},
    {"key": "baseline_ft",            "label": "Fine-Tuning",          "report": "{ds}_baseline_ft"},
    {"key": "baseline_neggradplus",   "label": "NegGrad+",             "report": "{ds}_baseline_neggradplus"},
    {"key": "baseline_random_labels", "label": "Random Labels",        "report": "{ds}_baseline_random_labels"},
    {"key": "baseline_bad_teaching",  "label": "Bad-Teaching",         "report": "{ds}_baseline_bad_teaching"},
    {"key": "baseline_scrub",         "label": "SCRUB",                "report": "{ds}_baseline_scrub"},
    {"key": "masuc",                  "label": "MASUC (Ours)",         "report": "{ds}_masuc"},
    {"key": "baseline_retrain",       "label": "Retrain (Gold Std.)",  "report": "{ds}_baseline_retrain"},
]


def load_method_results(ds: str, report_prefix: str, seeds: list[int] | None) -> dict:
    """
    Collect final-epoch retain_acc / forget_acc across seeds for one method.
    Falls back to a legacy single-seed file if no multi-seed files are found.
    """
    if report_prefix is None:
        return {"retain_acc": [], "forget_acc": [], "seeds": []}

    base = report_prefix.format(ds=ds)
    retain_vals, forget_vals, found = [], [], []

    if seeds:
        for seed in seeds:
            path = f"results/reports/{base}_seed{seed}_curve.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                retain_vals.append(data["retain_acc"][-1])
                forget_vals.append(data["forget_acc"][-1])
                found.append(seed)
    else:
        for path in sorted(glob.glob(f"results/reports/{base}_seed*_curve.json")):
            with open(path) as f:
                data = json.load(f)
            retain_vals.append(data["retain_acc"][-1])
            forget_vals.append(data["forget_acc"][-1])
            found.append(os.path.basename(path))

    if not retain_vals:
        path = f"results/reports/{base}_curve.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            retain_vals.append(data["retain_acc"][-1])
            forget_vals.append(data["forget_acc"][-1])
            found.append("legacy")

    return {"retain_acc": retain_vals, "forget_acc": forget_vals, "seeds": found}


def compute_stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1))


def load_before_metrics(ds: str) -> tuple[float, float] | None:
    """Read 'Before Unlearning' retain/forget accuracy from baseline_ft epoch-0 if available."""
    candidates = sorted(glob.glob(f"results/reports/{ds}_baseline_ft_seed*_curve.json")) \
        or [f"results/reports/{ds}_baseline_ft_curve.json"]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if data.get("retain_acc") and data.get("forget_acc"):
            return None
    return None


def compare_methods():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--seeds",   type=int, nargs="+", default=None,
                        help="Seeds to aggregate (default: auto-detect from files)")
    parser.add_argument("--before_retain", type=float, default=None,
                        help="Hard-coded Retain Acc of the original model (e.g. 0.8279)")
    parser.add_argument("--before_forget", type=float, default=None,
                        help="Hard-coded Forget Acc of the original model (e.g. 0.8200)")
    args = parser.parse_args()
    ds = args.dataset

    os.makedirs("results/summaries", exist_ok=True)
    os.makedirs("results/plots",     exist_ok=True)

    results = []
    for m in METHODS:
        if m["key"] == "before":
            if args.before_retain is not None and args.before_forget is not None:
                results.append({
                    "label":       m["label"],
                    "retain_mean": args.before_retain,
                    "retain_std":  0.0,
                    "forget_mean": args.before_forget,
                    "forget_std":  0.0,
                    "n_seeds":     0,
                })
            continue

        raw = load_method_results(ds, m["report"], args.seeds)
        if not raw["retain_acc"]:
            print(f"  [!] No results for '{m['label']}' — skipping.")
            continue
        r_mean, r_std = compute_stats(raw["retain_acc"])
        f_mean, f_std = compute_stats(raw["forget_acc"])
        results.append({
            "label":       m["label"],
            "retain_mean": r_mean,
            "retain_std":  r_std,
            "forget_mean": f_mean,
            "forget_std":  f_std,
            "n_seeds":     len(raw["retain_acc"]),
        })

    if not results:
        print("No method results found.")
        return

    # --- JSON & CSV ---
    json_path = f"results/summaries/{ds}_methods_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON  → {json_path}")

    csv_path = f"results/summaries/{ds}_methods_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "n_seeds", "retain_mean", "retain_std", "forget_mean", "forget_std"])
        w.writeheader()
        w.writerows(results)
    print(f"CSV   → {csv_path}")

    # --- Markdown table ---
    any_multi = any(r["n_seeds"] > 1 for r in results)
    md_lines = [
        f"# Unlearning Methods Comparison — {ds.upper()}\n",
        f"Format: mean ± std over multiple seeds (where available).\n" if any_multi else "",
        "| Method | n | Retain Acc | Forget Acc |",
        "|:---|:---:|:---:|:---:|",
    ]
    for r in results:
        if r["n_seeds"] > 1:
            retain_str = f"{r['retain_mean']:.2%} ± {r['retain_std']:.2%}"
            forget_str = f"{r['forget_mean']:.2%} ± {r['forget_std']:.2%}"
        else:
            retain_str = f"{r['retain_mean']:.2%}"
            forget_str = f"{r['forget_mean']:.2%}"
        n_disp = r["n_seeds"] if r["n_seeds"] > 0 else "—"
        md_lines.append(f"| **{r['label']}** | {n_disp} | {retain_str} | {forget_str} |")

    md_lines += [
        "",
        "### Objective",
        "- **Retain Accuracy** should be as high as possible (close to or higher than 'Before' / 'Retrain').",
        "- **Forget Accuracy** should be as low as possible (ideally close to 1/N, where N is the number of classes).",
    ]

    md_path = f"results/summaries/{ds}_methods_comparison.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"MD    → {md_path}")
    print("\n" + "\n".join(md_lines) + "\n")

    # --- Bar plot ---
    labels       = [r["label"] for r in results]
    retain_means = np.array([r["retain_mean"] for r in results])
    retain_stds  = np.array([r["retain_std"]  for r in results])
    forget_means = np.array([r["forget_mean"] for r in results])
    forget_stds  = np.array([r["forget_std"]  for r in results])

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(labels)), 5))
    bars1 = ax.bar(x - width / 2, retain_means, width,
                   yerr=retain_stds if any_multi else None,
                   capsize=4, label="Retain Acc", color="teal", alpha=0.85)
    bars2 = ax.bar(x + width / 2, forget_means, width,
                   yerr=forget_stds if any_multi else None,
                   capsize=4, label="Forget Acc", color="crimson", alpha=0.85)

    for bar, mean, std in zip(list(bars1) + list(bars2),
                              list(retain_means) + list(forget_means),
                              list(retain_stds)  + list(forget_stds)):
        label_str = f"{mean:.1%}\n±{std:.1%}" if std > 0 else f"{mean:.1%}"
        ax.annotate(label_str,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Accuracy")
    ax.set_title(f"Unlearning Methods Comparison — {ds.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=14, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"results/plots/{ds}_methods_comparison.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Plot  → {out}")


if __name__ == "__main__":
    compare_methods()
