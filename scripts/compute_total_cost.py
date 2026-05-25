import os
import glob
import json
import argparse

import numpy as np


def stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1))


def load_elapsed(pattern: str) -> list[float]:
    vals = []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            if "elapsed_sec" in d:
                vals.append(float(d["elapsed_sec"]))
        except Exception:
            pass
    return vals


def load_pretrain_elapsed(ds: str) -> tuple[float, float]:
    """Pre-training: student + teacher society."""
    student_t = 0.0
    teacher_total = 0.0

    teacher_files = sorted(glob.glob(f"results/reports/{ds}_teacher_*.json")) + sorted(glob.glob(f"results/reports/teacher_*.json"))
    for p in teacher_files:
        if "_curve" in p:
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            teacher_total += float(d.get("elapsed_sec", 0))
        except Exception:
            pass

    return student_t, teacher_total


def compute_total_cost():
    """
    Aggregate wall-clock elapsed across all reports for one dataset.
    Reports pre-training + per-method unlearning cost, with mean ± std over seeds.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    args = parser.parse_args()
    ds = args.dataset

    os.makedirs("results/summaries", exist_ok=True)

    student_t, teacher_total = load_pretrain_elapsed(ds)

    methods = [
        ("Fine-Tuning",          f"results/reports/{ds}_baseline_ft_seed*_curve.json"),
        ("NegGrad+",             f"results/reports/{ds}_baseline_neggradplus_seed*_curve.json"),
        ("Random Labels",        f"results/reports/{ds}_baseline_random_labels_seed*_curve.json"),
        ("Bad-Teaching",         f"results/reports/{ds}_baseline_bad_teaching_seed*_curve.json"),
        ("SCRUB",                f"results/reports/{ds}_baseline_scrub_seed*_curve.json"),
        ("MASUC (Ours)",         f"results/reports/{ds}_masuc_seed*_curve.json"),
        ("MASUC w/o EA",         f"results/reports/{ds}_masuc_no_ea_seed*_curve.json"),
        ("MASUC w/o KD",         f"results/reports/{ds}_masuc_no_kd_seed*_curve.json"),
        ("MASUC w/o RA",         f"results/reports/{ds}_masuc_no_ra_seed*_curve.json"),
        ("MASUC w/o Erasure",    f"results/reports/{ds}_masuc_no_erasure_seed*_curve.json"),
        ("Retrain (Gold)",       f"results/reports/{ds}_baseline_retrain_seed*_curve.json"),
    ]

    rows = []
    for label, pat in methods:
        vals = load_elapsed(pat)
        m, s = stats(vals)
        rows.append({"label": label, "elapsed_mean_s": m, "elapsed_std_s": s, "n": len(vals)})

    md = [
        f"# Compute Wall-Clock Accounting — {ds.upper()}\n",
        f"All times measured on the same hardware. Reports per-method unlearning cost (excluding pre-training).\n",
        f"**Offline pre-training cost (amortizable):**",
        f"- Student (one-shot): see `{ds}_train_curve.json` or per-epoch logs",
        f"- Teacher society (K=5): total {teacher_total:.1f}s ({teacher_total/60:.1f} min) — sum of independent runs\n",
        "## Per-method unlearning wall-clock",
        "",
        "| Method | n seeds | Mean elapsed (s) | Mean elapsed (min) |",
        "|:---|:---:|:---:|:---:|",
    ]
    for r in rows:
        if r["n"] == 0:
            md.append(f"| **{r['label']}** | 0 | — | — |")
        else:
            mean_s = r["elapsed_mean_s"]
            std_s  = r["elapsed_std_s"]
            mean_min = mean_s / 60.0
            std_min  = std_s / 60.0
            std_str_s   = f" ± {std_s:.1f}"   if r["n"] > 1 else ""
            std_str_min = f" ± {std_min:.2f}" if r["n"] > 1 else ""
            md.append(
                f"| **{r['label']}** | {r['n']} "
                f"| {mean_s:.1f}{std_str_s} "
                f"| {mean_min:.2f}{std_str_min} |"
            )

    md += [
        "",
        "## Notes",
        "- *Per-method* numbers exclude offline pre-training (student + teachers).",
        "- *Amortized* MASUC cost over N forget requests:",
        f"  `cost(N) = teacher_pretrain + N * unlearning_phase`",
        "- Speed-up vs Retrain is computed against the *Retrain (Gold)* row.",
    ]

    md_text = "\n".join(md)
    md_path = f"results/summaries/{ds}_compute_cost.md"
    with open(md_path, "w") as f:
        f.write(md_text)

    json_path = f"results/summaries/{ds}_compute_cost.json"
    with open(json_path, "w") as f:
        json.dump({
            "pretrain": {"teacher_total_sec": teacher_total},
            "methods":  rows,
        }, f, indent=2)

    print(md_text)
    print(f"\nMD   → {md_path}")
    print(f"JSON → {json_path}")


if __name__ == "__main__":
    compute_total_cost()
