"""
Visualise the results of the MASUC hyperparameter ablation study.

Reads results/reports/{ds}_hp_ablation_summary.json and produces one
two-panel subplot per swept factor:
  - Left panel:  retain_acc vs. hyperparameter value
  - Right panel: forget_acc vs. hyperparameter value

A vertical dashed line marks the default value for each factor.

Output: results/plots/{ds}_hp_ablation_{factor}.png  (one per factor)
        results/plots/{ds}_hp_ablation_all.png        (combined grid)

Usage:
    python -m scripts.compare_hp_ablation --dataset mnist
"""

import os
import json
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DEFAULTS = {
    "lambda_1":    1.0,
    "lambda_2":    0.1,
    "lambda_3":    0.5,
    "lr_student":  0.001,
    "epochs":      5,
    "temperature": 2.0,
}

LABELS = {
    "lambda_1":    r"$\lambda_1$ (KD weight)",
    "lambda_2":    r"$\lambda_2$ (Energy Alignment weight)",
    "lambda_3":    r"$\lambda_3$ (Erasure weight)",
    "lr_student":  "Student LR",
    "epochs":      "Epochs",
    "temperature": r"Temperature $\tau$",
}

COLORS = {
    "retain_acc": "#2196F3",   # blue
    "forget_acc": "#F44336",   # red
}


def load_summary(ds: str) -> dict:
    path = f"results/reports/{ds}_hp_ablation_summary.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Summary not found: {path}\n"
            f"Run first: python -m scripts.run_hp_ablation --dataset {ds}"
        )
    with open(path) as f:
        rows = json.load(f)

    # Group by factor
    grouped: dict[str, dict] = {}
    for row in rows:
        factor = row["factor"]
        if factor not in grouped:
            grouped[factor] = {"values": [], "retain_acc": [], "forget_acc": []}
        grouped[factor]["values"].append(row["value"])
        grouped[factor]["retain_acc"].append(row.get("retain_acc"))
        grouped[factor]["forget_acc"].append(row.get("forget_acc"))
    return grouped


def plot_factor(factor: str, data: dict, ds: str, save: bool = True):
    """Two-panel plot for a single factor sweep."""
    values      = data["values"]
    retain_accs = data["retain_acc"]
    forget_accs = data["forget_acc"]
    default_val = DEFAULTS[factor]
    label       = LABELS[factor]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    fig.suptitle(f"HP Ablation — {label}  [{ds.upper()}]", fontsize=13, fontweight="bold")

    for ax, metric, title, color in zip(
        axes,
        [retain_accs, forget_accs],
        ["Retain Accuracy ↑ (higher = better)", "Forget Accuracy ↓ (lower = better)"],
        [COLORS["retain_acc"], COLORS["forget_acc"]],
    ):
        valid = [(v, m) for v, m in zip(values, metric) if m is not None]
        if not valid:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        xs, ys = zip(*valid)

        ax.plot(xs, ys, marker="o", color=color, linewidth=2, markersize=7, zorder=3)
        ax.fill_between(xs, ys, alpha=0.08, color=color)

        # Mark the default value
        if default_val in xs:
            idx = xs.index(default_val)
            ax.axvline(default_val, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.annotate(
                f"default\n({default_val})",
                xy=(default_val, ys[idx]),
                xytext=(8, -20),
                textcoords="offset points",
                fontsize=8,
                color="gray",
            )

        # Log scale for lr
        if factor == "lr_student":
            ax.set_xscale("log")

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if save:
        os.makedirs("results/plots", exist_ok=True)
        out = f"results/plots/{ds}_hp_ablation_{factor}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")
    else:
        return fig


def plot_all(grouped: dict, ds: str):
    """Combined grid: one row per factor."""
    factors = list(grouped.keys())
    n = len(factors)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4 * n))
    fig.suptitle(f"MASUC — Full Hyperparameter Ablation [{ds.upper()}]",
                 fontsize=14, fontweight="bold", y=1.01)

    for row_idx, factor in enumerate(factors):
        data        = grouped[factor]
        values      = data["values"]
        default_val = DEFAULTS[factor]
        label       = LABELS[factor]

        for col_idx, (metric_key, title, color) in enumerate([
            ("retain_acc", "Retain Acc ↑", COLORS["retain_acc"]),
            ("forget_acc", "Forget Acc ↓", COLORS["forget_acc"]),
        ]):
            ax = axes[row_idx, col_idx]
            metric = data[metric_key]
            valid  = [(v, m) for v, m in zip(values, metric) if m is not None]
            if not valid:
                continue
            xs, ys = zip(*valid)

            ax.plot(xs, ys, marker="o", color=color, linewidth=2, markersize=6, zorder=3)
            ax.fill_between(xs, ys, alpha=0.08, color=color)
            ax.axvline(default_val, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                       label=f"default ({default_val})")

            if factor == "lr_student":
                ax.set_xscale("log")

            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_title(f"{label} — {title}", fontsize=9)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = f"results/plots/{ds}_hp_ablation_all.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Combined grid saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Compare MASUC HP Ablation results")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--factors", type=str, nargs="+", default=None,
                        help="Subset of factors to plot (default: all)")
    args = parser.parse_args()

    ds      = args.dataset
    grouped = load_summary(ds)
    factors = args.factors if args.factors else list(grouped.keys())

    print(f"\nPlotting HP ablation for {ds.upper()} — factors: {factors}\n")

    for factor in factors:
        if factor not in grouped:
            print(f"  ⚠️  Factor '{factor}' not found in summary, skipping.")
            continue
        plot_factor(factor, grouped[factor], ds)

    # Combined grid (only if all factors present)
    if set(factors) == set(grouped.keys()):
        plot_all({f: grouped[f] for f in factors}, ds)


if __name__ == "__main__":
    main()
