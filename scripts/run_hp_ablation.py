import os
import sys
import json
import argparse
import subprocess
import time

DEFAULTS = {
    "lambda_1":    1.0,
    "lambda_2":    0.1,
    "lambda_3":    0.5,
    "lr_student":  0.001,
    "epochs":      5,
    "temperature": 2.0,
}

SWEEPS = {
    "lambda_1":    [0.0, 0.25, 0.5, 1.0, 2.0, 5.0],
    "lambda_2":    [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
    "lambda_3":    [0.0, 0.1,  0.25, 0.5, 1.0, 2.0],
    "lr_student":  [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "epochs":      [2, 3, 5, 8, 10],
    "temperature": [1.0, 1.5, 2.0, 4.0, 8.0],
}


def build_run_id(ds: str, factor: str, value) -> str:
    """
    Build a unique run ID for a given factor/value combination.
    """
    val_str = str(value).replace(".", "p")
    return f"{ds}_hp_{factor}_{val_str}"


def run_single(ds: str, forget_class: int, factor: str, value, dry_run: bool = False) -> dict:
    """
    Launch one run_masuc subprocess with all hyperparams set to their defaults
    except for the swept factor.
    Returns the parsed curve JSON (last epoch metrics).
    """
    run_id = build_run_id(ds, factor, value)


    hp = dict(DEFAULTS)
    hp[factor] = value

    cmd = [
        sys.executable, "-m", "scripts.run_masuc",
        "--dataset",     ds,
        "--forget_class", str(forget_class),
        "--run_id",       run_id,
        "--lambda_1",     str(hp["lambda_1"]),
        "--lambda_2",     str(hp["lambda_2"]),
        "--lambda_3",     str(hp["lambda_3"]),
        "--lr_student",   str(hp["lr_student"]),
        "--epochs",       str(int(hp["epochs"])),
        "--temperature",  str(hp["temperature"]),
    ]

    print(f"\n{'─'*60}")
    print(f"  [{factor}={value}]  run_id={run_id}")
    print(f"  cmd: {' '.join(cmd[2:])}")
    print(f"{'─'*60}")

    if dry_run:
        return {"run_id": run_id, "factor": factor, "value": value, "dry_run": True}

    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.getcwd())
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"Run FAILED (returncode={result.returncode})")
        return {"run_id": run_id, "factor": factor, "value": value, "error": True, "elapsed": elapsed}


    curve_path = f"results/reports/{run_id}_curve.json"
    try:
        with open(curve_path) as f:
            curve = json.load(f)
        last_retain = curve["retain_acc"][-1]
        last_forget = curve["forget_acc"][-1]
        last_loss   = curve["train_loss"][-1]
    except Exception as e:
        print(f"    Could not read curve: {e}")
        last_retain = last_forget = last_loss = None

    return {
        "run_id":      run_id,
        "factor":      factor,
        "value":       value,
        "retain_acc":  last_retain,
        "forget_acc":  last_forget,
        "train_loss":  last_loss,
        "elapsed_s":   round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="MASUC Hyperparameter Ablation Study")
    parser.add_argument("--dataset",      type=str,  default="mnist",
                        choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int,  default=3)
    parser.add_argument("--factors",      type=str,  nargs="+", default=None,
                        help="Subset of factors to sweep (default: all). "
                             "Choices: lambda_1 lambda_2 lambda_3 lr_student epochs temperature")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    ds            = args.dataset
    forget_class  = args.forget_class
    factors       = args.factors if args.factors else list(SWEEPS.keys())

    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("results/plots",   exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HP Ablation | Dataset: {ds.upper()} | Forget class: {forget_class}")
    print(f"  Factors to sweep: {factors}")
    total_runs = sum(len(SWEEPS[f]) for f in factors)
    print(f"  Total runs: {total_runs}")
    print(f"{'='*60}\n")

    summary = []

    for factor in factors:
        print(f"\n{'#'*60}")
        print(f"  Sweeping: {factor}  (default={DEFAULTS[factor]})")
        print(f"  Values:   {SWEEPS[factor]}")
        print(f"{'#'*60}")

        for value in SWEEPS[factor]:
            result = run_single(ds, forget_class, factor, value, dry_run=args.dry_run)
            summary.append(result)
            if not args.dry_run:
                print(f"  → retain={result.get('retain_acc', 'N/A'):.4f}  "
                      f"forget={result.get('forget_acc', 'N/A'):.4f}  "
                      f"({result.get('elapsed_s', 0):.0f}s)")


    summary_path = f"results/reports/{ds}_hp_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print(f"   Run: python -m scripts.compare_hp_ablation --dataset {ds}")


if __name__ == "__main__":
    main()
