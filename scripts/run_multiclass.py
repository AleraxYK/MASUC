"""
Multi-class forget orchestrator (addresses reviewer C7).

For a list of forget classes, runs:
  * split generation (if missing)
  * MASUC × seeds
  * all baselines × seeds
  * compare_methods aggregation per class

Then produces a cross-class table that shows the method's robustness to
forget-class choice.

Default classes for CIFAR-10: 3 (Cat), 5 (Dog), 7 (Horse) — chosen because
each is natively covered by a blind teacher in the K=5 society
(blind = {3, 7, 1, 5, 9}).

Usage:
  python -m scripts.run_multiclass --dataset cifar10 --classes 3 5 7 --seeds 42 123 7
"""

import os
import csv
import json
import argparse
import subprocess
import sys

import numpy as np


def run(cmd: list[str]) -> int:
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=os.getcwd()).returncode


def aggregate_per_class(ds: str, classes: list[int], seeds: list[int]) -> list[dict]:
    """Read per-class results and produce a flat table."""
    rows = []
    for fc in classes:
        for method_key, method_label, prefix in [
            ("ft",            "FT",             f"{ds}_baseline_ft"),
            ("neggradplus",   "NegGrad+",       f"{ds}_baseline_neggradplus"),
            ("random_labels", "Random Labels",  f"{ds}_baseline_random_labels"),
            ("bad_teaching",  "Bad-Teaching",   f"{ds}_baseline_bad_teaching"),
            ("scrub",         "SCRUB",          f"{ds}_baseline_scrub"),
            ("masuc",         "MASUC (Ours)",   f"{ds}_masuc"),
            ("retrain",       "Retrain (Gold)", f"{ds}_baseline_retrain"),
        ]:
            retain_vals, forget_vals = [], []
            for seed in seeds:
                # Reports are class-agnostic in filename (overwrites per forget_class).
                # We rely on the *_curve.json named identically; class is encoded in
                # the split file used at training time. To distinguish per-class
                # results we expect this script to have moved the prior results
                # before re-running with a different class.
                #
                # The cross-class aggregator therefore reads results from a
                # per-class results directory: results_class{fc}/reports/...
                p = f"results_class{fc}/reports/{prefix}_seed{seed}_curve.json"
                if not os.path.exists(p):
                    # Try the standard location too (single-class runs without
                    # the directory snapshotting trick below).
                    p_alt = f"results/reports/{prefix}_seed{seed}_curve.json"
                    p = p_alt if os.path.exists(p_alt) else p
                if not os.path.exists(p):
                    continue
                with open(p) as f:
                    d = json.load(f)
                retain_vals.append(d["retain_acc"][-1])
                forget_vals.append(d["forget_acc"][-1])

            if not retain_vals:
                continue
            arr_r, arr_f = np.array(retain_vals), np.array(forget_vals)
            rows.append({
                "forget_class": fc,
                "method":       method_label,
                "n_seeds":      len(retain_vals),
                "retain_mean":  float(arr_r.mean()),
                "retain_std":   float(arr_r.std(ddof=1)) if len(arr_r) > 1 else 0.0,
                "forget_mean":  float(arr_f.mean()),
                "forget_std":   float(arr_f.std(ddof=1)) if len(arr_f) > 1 else 0.0,
            })
    return rows


def write_summary(ds: str, rows: list[dict]):
    if not rows:
        print("No per-class results found.")
        return

    os.makedirs("results/summaries", exist_ok=True)

    json_path = f"results/summaries/{ds}_multiclass_stats.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON  → {json_path}")

    csv_path = f"results/summaries/{ds}_multiclass_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV   → {csv_path}")

    # Group rows by method
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    md = [
        f"# Multi-Class Unlearning — {ds.upper()}\n",
        "Per-class Retain / Forget accuracy, demonstrating robustness to forget-class choice.",
        "Format: mean ± std across seeds.\n",
        "| Method \\ Forget class | " + " | ".join(f"{r['forget_class']} (R)" + " / " + f"{r['forget_class']} (F)" for r in by_method[next(iter(by_method))]) + " |",
        "|:---|" + "|".join([":---:"] * len(by_method[next(iter(by_method))])) + "|",
    ]
    for method, mrows in by_method.items():
        cells = []
        for r in mrows:
            cells.append(f"{r['retain_mean']:.2%} / {r['forget_mean']:.2%}")
        md.append(f"| **{method}** | " + " | ".join(cells) + " |")

    md_text = "\n".join(md)
    md_path = f"results/summaries/{ds}_multiclass_stats.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"MD    → {md_path}")
    print(f"\n{md_text}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--classes", type=int, nargs="+", default=[3, 5, 7],
                        help="Forget classes to evaluate (default: 3 5 7)")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--skip_run",  action="store_true",
                        help="Only aggregate existing results — do not launch training")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Run MASUC only (skip baselines)")
    args = parser.parse_args()

    ds = args.dataset
    classes = args.classes
    seeds   = args.seeds

    print(f"\n{'='*60}")
    print(f"  Multi-class unlearning | Dataset: {ds.upper()} | Classes: {classes}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}\n")

    if not args.skip_run:
        for fc in classes:
            print(f"\n--- Forget class {fc} ---")

            # Split
            sp = f"results/splits/{ds}_split_forget_{fc}.json"
            if not os.path.exists(sp):
                run([sys.executable, "-m", "scripts.make_split", "--dataset", ds, "--forget_class", str(fc)])

            # Move previous per-class results aside to avoid overwrite
            class_dir = f"results_class{fc}"
            os.makedirs(f"{class_dir}/reports", exist_ok=True)
            os.makedirs(f"{class_dir}/checkpoints", exist_ok=True)

            for seed in seeds:
                # MASUC
                run([sys.executable, "-m", "scripts.run_masuc",
                     "--dataset", ds, "--forget_class", str(fc), "--seed", str(seed)])

                if not args.no_baselines:
                    for script in [
                        "scripts.baseline_ft",
                        "scripts.baseline_neggradplus",
                        "scripts.baseline_random_labels",
                        "scripts.baseline_bad_teaching",
                        "scripts.baseline_scrub",
                        "scripts.baseline_retrain",
                    ]:
                        run([sys.executable, "-m", script,
                             "--dataset", ds, "--forget_class", str(fc), "--seed", str(seed)])

            # Snapshot results for this class (move so next iteration won't overwrite)
            import shutil
            for pattern_dir in ["reports", "checkpoints", "plots"]:
                src = f"results/{pattern_dir}"
                dst = f"{class_dir}/{pattern_dir}"
                os.makedirs(dst, exist_ok=True)
                # Move only files matching this dataset's prefix
                for f in os.listdir(src):
                    if f.startswith(f"{ds}_") and (f"_seed" in f or "_final" in f or "_curve" in f):
                        shutil.move(os.path.join(src, f), os.path.join(dst, f))

            print(f"Class {fc} done. Results stashed in {class_dir}/")

    # Aggregate
    print("\n--- Aggregating cross-class results ---")
    rows = aggregate_per_class(ds, classes, seeds)
    write_summary(ds, rows)


if __name__ == "__main__":
    main()
