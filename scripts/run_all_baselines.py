import os
import argparse


def run_all_baselines():
    """
    Orchestrator: runs every baseline (FT, NegGrad+, Random Labels, Bad-Teaching,
    SCRUB, Retrain) across multiple seeds for direct multi-seed comparison
    against MASUC.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int, default=3)
    parser.add_argument("--seeds",        type=int, nargs="+", default=[42, 123, 7],
                        help="Seeds for multi-run statistics (default: 42 123 7)")
    parser.add_argument("--skip",         type=str, nargs="*", default=[],
                        help="Skip selected baselines, e.g. --skip retrain scrub")
    args = parser.parse_args()

    ds    = args.dataset
    fc    = args.forget_class
    seeds = args.seeds

    baselines = [
        ("Fine-Tuning",     "baseline_ft",            "ft"),
        ("NegGrad+",        "baseline_neggradplus",   "neggradplus"),
        ("Random Labels",   "baseline_random_labels", "random_labels"),
        ("Bad-Teaching",    "baseline_bad_teaching",  "bad_teaching"),
        ("SCRUB",           "baseline_scrub",         "scrub"),
        ("Retrain (Gold)",  "baseline_retrain",       "retrain"),
    ]

    baselines = [b for b in baselines if b[2] not in args.skip]

    total = len(baselines) * len(seeds)
    print(f"\n{'='*60}")
    print(f"  Baselines | Dataset: {ds.upper()} | Forget class: {fc}")
    print(f"  Seeds: {seeds}  |  Total runs: {total}")
    print(f"{'='*60}\n")

    for name, script, _key in baselines:
        for seed in seeds:
            cmd = f"python -m scripts.{script} --dataset {ds} --forget_class {fc} --seed {seed}"
            print(f"\n>>> [{name} | seed={seed}]\n>>> {cmd}\n")
            code = os.system(cmd)
            if code != 0:
                print(f"\n[!] Baseline '{name}' seed={seed} failed (exit code {code}). Aborting.")
                return

    print(f"\n{'='*60}")
    print(f"  Done! Run: python -m scripts.compare_methods --dataset {ds} --seeds {' '.join(map(str, seeds))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all_baselines()
