import os
import sys
import argparse


def run_ablation():
    """
    Orchestrator: runs MASUC once per (config × seed) to produce a full
    ablation study with multi-seed statistics.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int, default=3)
    parser.add_argument("--seeds",        type=int, nargs="+", default=[42, 123, 7],
                        help="Seeds for multi-run statistics (default: 42 123 7)")
    args = parser.parse_args()

    ds    = args.dataset
    fc    = args.forget_class
    seeds = args.seeds

    runs = [
        ("Full MASUC",                  ""),
        ("No Energy Alignment",         "--no_ea"),
        ("No Knowledge Distillation",   "--no_kd"),
        ("No Reciprocal Altruism",      "--no_ra"),
        ("No Erasure Loss",             "--no_erasure"),
    ]

    total = len(runs) * len(seeds)
    print(f"\n{'='*60}")
    print(f"  Ablation Study | Dataset: {ds.upper()} | Forget class: {fc}")
    print(f"  Seeds: {seeds}  |  Total runs: {total}")
    print(f"{'='*60}\n")

    for name, flag in runs:
        for seed in seeds:
            flag_str = f"{flag} " if flag else ""
            cmd = f"python -m scripts.run_masuc --dataset {ds} --forget_class {fc} {flag_str}--seed {seed}".strip()
            print(f"\n>>> [{name} | seed={seed}]\n>>> {cmd}\n")
            code = os.system(cmd)
            if code != 0:
                print(f"\n[!] Run '{name}' seed={seed} failed (exit code {code}). Aborting.")
                return

    print(f"\n{'='*60}")
    print(f"  Done! Run: python -m scripts.compare_ablation --dataset {ds} --seeds {' '.join(map(str, seeds))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_ablation()
