"""
Auto-generates LaTeX-ready tables for the MASUC paper, from the aggregated
JSON/CSV files under results/summaries/.

Produces files in paper_tables/{ds}/*.tex that can be \\input{}'d directly:
  * tab_methods.tex       — main results vs SOTA baselines (retain / forget)
  * tab_ablation.tex      — ablation study (multi-seed mean ± std)
  * tab_mia.tex           — membership inference attack AUCs
  * tab_calibration.tex   — ECE + confidence (FT pathology quantified)
  * tab_compute.tex       — wall-clock per method
  * tab_multiclass.tex    — per-class breakdown (if available)

Usage:
  python -m scripts.generate_paper_tables --dataset cifar10
"""

import os
import json
import argparse


def fmt_mean_std(m, s, pct=True, prec=2):
    if pct:
        return f"${m*100:.{prec}f} \\pm {s*100:.{prec}f}$" if s > 0 else f"${m*100:.{prec}f}$"
    return f"${m:.{prec}f} \\pm {s:.{prec}f}$"


def safe_load_json(path):
    if not os.path.exists(path):
        print(f"  [skip] missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def write_tex(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Methods comparison
# ─────────────────────────────────────────────────────────────────────────────
def gen_methods_table(ds: str, outdir: str):
    data = safe_load_json(f"results/summaries/{ds}_methods_comparison.json")
    if not data:
        return

    rows = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Class-level unlearning on " + ds.upper() + " (forget class 3, ResNet-18). "
        "Format: mean $\\pm$ std over multiple seeds.}",
        "\\label{tab:methods_" + ds + "}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & $n$ & Retain Acc. ($\\uparrow$) & Forget Acc. ($\\downarrow$) \\\\",
        "\\midrule",
    ]
    for r in data:
        rmean = r.get("retain_mean", float("nan"))
        rstd  = r.get("retain_std",  0.0)
        fmean = r.get("forget_mean", float("nan"))
        fstd  = r.get("forget_std",  0.0)
        n     = r.get("n_seeds", 0)
        label = r.get("label", "?")
        emph = label == "MASUC (Ours)"
        prefix = "\\textbf{" + label + "}" if emph else label
        rows.append(
            f"{prefix} & {n if n>0 else '---'} "
            f"& {fmt_mean_std(rmean, rstd)}\\% & {fmt_mean_std(fmean, fstd)}\\% \\\\"
        )
    rows += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    write_tex(f"{outdir}/tab_methods.tex", "\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ablation
# ─────────────────────────────────────────────────────────────────────────────
def gen_ablation_table(ds: str, outdir: str):
    data = safe_load_json(f"results/summaries/{ds}_ablation_stats.json")
    if not data:
        return

    full_forget = next((r["forget_mean"] for r in data if r["label"] == "Full MASUC"), None)

    rows = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Component ablation on " + ds.upper() + ". Multi-seed mean $\\pm$ std. "
        "$\\Delta$Forget is the change in forget accuracy relative to Full MASUC "
        "(positive $=$ worse forgetting).}",
        "\\label{tab:ablation_" + ds + "}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Configuration & $n$ & Retain Acc. & Forget Acc. & $\\Delta$Forget \\\\",
        "\\midrule",
    ]
    for r in data:
        rmean = r["retain_mean"]; rstd = r["retain_std"]
        fmean = r["forget_mean"]; fstd = r["forget_std"]
        n     = r["n_seeds"]
        label = r["label"]
        if full_forget is None or label == "Full MASUC":
            dstr = "---"
        else:
            delta = (fmean - full_forget) * 100
            dstr  = f"${delta:+.2f}$ pp"
        emph = label == "Full MASUC"
        prefix = "\\textbf{" + label + "}" if emph else label
        rows.append(
            f"{prefix} & {n} & "
            f"{fmt_mean_std(rmean, rstd)}\\% & {fmt_mean_std(fmean, fstd)}\\% & {dstr} \\\\"
        )
    rows += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    write_tex(f"{outdir}/tab_ablation.tex", "\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# 3. MIA
# ─────────────────────────────────────────────────────────────────────────────
def gen_mia_table(ds: str, outdir: str):
    data = safe_load_json(f"results/summaries/{ds}_mia_stats.json")
    if not data:
        return

    rows = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Membership-inference attack on " + ds.upper() + ". Members are "
        "forget-train samples (originally in the training set); non-members are "
        "forget-test samples (never seen). AUC $\\approx 0.5$ indicates no recoverable "
        "membership signal (ideal privacy).}",
        "\\label{tab:mia_" + ds + "}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & $n$ & AUC (loss) & AUC (conf) & AUC (entropy) \\\\",
        "\\midrule",
    ]
    for r in data:
        n = r["n_seeds"]
        rows.append(
            f"{r['label']} & {n} & "
            f"{fmt_mean_std(r['auc_loss_m'], r['auc_loss_s'], pct=False, prec=4)} & "
            f"{fmt_mean_std(r['auc_conf_m'], r['auc_conf_s'], pct=False, prec=4)} & "
            f"{fmt_mean_std(r['auc_ent_m'],  r['auc_ent_s'],  pct=False, prec=4)} \\\\"
        )
    rows += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    write_tex(f"{outdir}/tab_mia.tex", "\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibration
# ─────────────────────────────────────────────────────────────────────────────
def gen_calibration_table(ds: str, outdir: str):
    data = safe_load_json(f"results/summaries/{ds}_calibration_stats.json")
    if not data:
        return

    rows = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Calibration analysis on " + ds.upper() + ". ECE on the retain "
        "set ($\\downarrow$): lower is better-calibrated. Conf(forget) is the mean "
        "max-softmax confidence on forget-set inputs ($\\downarrow$): low values "
        "indicate healthy unlearning; high values combined with low forget accuracy "
        "indicate confident misprediction (over-specialization pathology).}",
        "\\label{tab:calibration_" + ds + "}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & $n$ & ECE (retain) & Conf (retain) & Conf (forget) \\\\",
        "\\midrule",
    ]
    for r in data:
        rows.append(
            f"{r['label']} & {r['n_seeds']} & "
            f"{fmt_mean_std(r['ece_m'], r['ece_s'], pct=False, prec=4)} & "
            f"{fmt_mean_std(r['conf_retain_m'], r['conf_retain_s'], pct=False, prec=4)} & "
            f"{fmt_mean_std(r['conf_forget_m'], r['conf_forget_s'], pct=False, prec=4)} \\\\"
        )
    rows += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    write_tex(f"{outdir}/tab_calibration.tex", "\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compute
# ─────────────────────────────────────────────────────────────────────────────
def gen_compute_table(ds: str, outdir: str):
    data = safe_load_json(f"results/summaries/{ds}_compute_cost.json")
    if not data:
        return
    methods = data.get("methods", [])
    teacher_total = data.get("pretrain", {}).get("teacher_total_sec", 0.0)

    rows = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Wall-clock unlearning cost on " + ds.upper() + " (same hardware "
        "across methods). Teacher-society offline pre-training: "
        f"{teacher_total:.0f}\\,s "
        f"({teacher_total/60:.1f}\\,min), amortized across forget requests.}}",
        "\\label{tab:compute_" + ds + "}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & $n$ seeds & Mean time (s) & Mean time (min) \\\\",
        "\\midrule",
    ]
    for r in methods:
        if r["n"] == 0:
            continue
        m_s = r["elapsed_mean_s"]
        s_s = r["elapsed_std_s"]
        m_m = m_s / 60.0
        s_m = s_s / 60.0
        rows.append(
            f"{r['label']} & {r['n']} & "
            f"{fmt_mean_std(m_s, s_s, pct=False, prec=1)} & "
            f"{fmt_mean_std(m_m, s_m, pct=False, prec=2)} \\\\"
        )
    rows += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    write_tex(f"{outdir}/tab_compute.tex", "\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-class
# ─────────────────────────────────────────────────────────────────────────────
def gen_multiclass_table(ds: str, outdir: str):
    data = safe_load_json(f"results/summaries/{ds}_multiclass_stats.json")
    if not data:
        return

    classes = sorted({r["forget_class"] for r in data})
    methods = list(dict.fromkeys(r["method"] for r in data))

    header = " & ".join([f"\\multicolumn{{2}}{{c}}{{Class {c}}}" for c in classes])
    subhdr = " & ".join(["R & F"] * len(classes))

    rows = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Multi-class unlearning robustness on " + ds.upper() + ". "
        "R = retain accuracy ($\\uparrow$), F = forget accuracy ($\\downarrow$). "
        "Mean across seeds. Demonstrates that MASUC's trade-off is stable across "
        "forget-class choices, not specific to a single class.}",
        "\\label{tab:multiclass_" + ds + "}",
        "\\begin{tabular}{l" + "cc" * len(classes) + "}",
        "\\toprule",
        " & " + header + " \\\\",
        " & " + subhdr + " \\\\",
        "\\midrule",
    ]
    for m in methods:
        cells = []
        for c in classes:
            entry = next((r for r in data if r["method"] == m and r["forget_class"] == c), None)
            if entry is None:
                cells += ["---", "---"]
            else:
                cells.append(f"{entry['retain_mean']*100:.2f}\\%")
                cells.append(f"{entry['forget_mean']*100:.2f}\\%")
        emph = m == "MASUC (Ours)"
        prefix = "\\textbf{" + m + "}" if emph else m
        rows.append(f"{prefix} & " + " & ".join(cells) + " \\\\")
    rows += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    write_tex(f"{outdir}/tab_multiclass.tex", "\n".join(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    args = parser.parse_args()
    ds = args.dataset

    outdir = f"paper_tables/{ds}"
    os.makedirs(outdir, exist_ok=True)

    print(f"Generating LaTeX tables for {ds.upper()} → {outdir}/\n")
    gen_methods_table(ds,     outdir)
    gen_ablation_table(ds,    outdir)
    gen_mia_table(ds,         outdir)
    gen_calibration_table(ds, outdir)
    gen_compute_table(ds,     outdir)
    gen_multiclass_table(ds,  outdir)

    print(f"\nDone. \\input{{{outdir}/tab_*.tex}} in your paper.")


if __name__ == "__main__":
    main()
