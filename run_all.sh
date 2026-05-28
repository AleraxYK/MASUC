#!/usr/bin/env bash
#
# Full MASUC reproduction pipeline.
# Designed for single-GPU CUDA (RTX 2080/3070 or T4/P100/V100/A100 on Kaggle/Colab).
#
# Auto-enables AMP (mixed precision) and the parallel MASUC variant when CUDA
# is detected.  Falls back gracefully on MPS/CPU.
#
# Stages (skippable via --skip):
#   pretrain    — student pretrain per dataset
#   teachers    — train K=5 teachers per dataset
#   splits      — forget/retain split per (dataset, forget_class)
#   ablation    — MASUC + 4 ablations × seeds
#   baselines   — FT, NegGrad+, RandLabels, Bad-Teaching, SCRUB, Retrain × seeds
#   hp          — HP OFAT sweep (CIFAR-10 only by default)
#   multiclass  — per-class robustness sweep (3 forget classes) — addresses C7
#   aggregate   — compare_*, eval_mia, eval_calibration, LaTeX tables
#
# Usage:
#   bash run_all.sh                                  # CIFAR-10, all stages, AMP+parallel auto
#   bash run_all.sh --datasets cifar10 mnist
#   bash run_all.sh --seeds 42 123 7
#   bash run_all.sh --skip pretrain teachers hp
#   bash run_all.sh --no-amp --no-parallel           # disable accelerations
#   bash run_all.sh --forget_class 3
#   CUDA_VISIBLE_DEVICES=0 bash run_all.sh

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DATASETS=("cifar10")
SEEDS=(42 123 7)
FORGET_CLASS=3
SKIP=()
USE_AMP=auto

# ─────────────────────────────────────────────────────────────────────────────
# Arg parse
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      shift
      DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done
      ;;
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
      ;;
    --skip)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do SKIP+=("$1"); shift; done
      ;;
    --forget_class) FORGET_CLASS="$2"; shift 2;;
    --no-amp)       USE_AMP=no;        shift;;
    -h|--help) sed -n '1,28p' "$0"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

skip_has() {
  local needle="$1"; shift
  for s in "${SKIP[@]:-}"; do [[ "$s" == "$needle" ]] && return 0; done
  return 1
}

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
sep() { echo "═════════════════════════════════════════════════════════════"; }

mkdir -p results/{checkpoints,reports,plots,splits,summaries}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: env detection
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Stage 0: env check"
python - <<'PY'
import torch
print(f"  torch={torch.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  CUDA={torch.version.cuda} | device={p.name} | VRAM={p.total_memory/1e9:.1f}GB")
elif torch.backends.mps.is_available():
    print("  MPS available (Apple Silicon)")
else:
    print("  CPU only — pipeline will be very slow")
PY

HAS_CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")

AMP_FLAG=""
MASUC_SCRIPT="scripts.run_masuc"
if [[ "$HAS_CUDA" == "1" ]]; then
  [[ "$USE_AMP" != "no" ]] && AMP_FLAG="--amp"
fi
echo "  → MASUC script: $MASUC_SCRIPT  |  AMP: ${AMP_FLAG:-disabled}"
echo "  → Datasets: ${DATASETS[*]}  |  Seeds: ${SEEDS[*]}  |  forget_class: $FORGET_CLASS"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: pretrain student (per dataset)
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "pretrain"; then
  for ds in "${DATASETS[@]}"; do
    sep
    CKPT="results/checkpoints/${ds}_model_before_best.pth"
    if [[ -f "$CKPT" ]]; then
      log "Stage 1 [$ds]: pretrained checkpoint exists → skip"
    else
      log "Stage 1 [$ds]: pretrain student"
      python -m scripts.train_model --dataset "$ds"
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: teacher society (per dataset)
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "teachers"; then
  for ds in "${DATASETS[@]}"; do
    sep
    T0="results/checkpoints/${ds}_teacher_0.pth"
    if [[ -f "$T0" ]]; then
      log "Stage 2 [$ds]: teachers already trained → skip"
    else
      log "Stage 2 [$ds]: train K=5 teachers"
      python -m scripts.train_agents --dataset "$ds"
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: forget/retain splits
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "splits"; then
  for ds in "${DATASETS[@]}"; do
    sep
    SP="results/splits/${ds}_split_forget_${FORGET_CLASS}.json"
    if [[ -f "$SP" ]]; then
      log "Stage 3 [$ds]: split exists → skip ($SP)"
    else
      log "Stage 3 [$ds]: generate forget/retain split (class $FORGET_CLASS)"
      python -m scripts.make_split --dataset "$ds" --forget_class "$FORGET_CLASS"
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: MASUC + 4 ablations × seeds (per dataset)
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "ablation"; then
  for ds in "${DATASETS[@]}"; do
    sep
    log "Stage 4 [$ds]: MASUC + 4 ablations × ${#SEEDS[@]} seeds  ($MASUC_SCRIPT $AMP_FLAG)"
    for seed in "${SEEDS[@]}"; do
      for flag in "" "--no_ea" "--no_kd" "--no_ra" "--no_erasure"; do
        echo ">>> MASUC ${flag:-full} | seed=$seed"
        python -m "$MASUC_SCRIPT" --dataset "$ds" --forget_class "$FORGET_CLASS" --seed "$seed" $AMP_FLAG $flag
      done
    done
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: all baselines × seeds (per dataset)
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "baselines"; then
  for ds in "${DATASETS[@]}"; do
    sep
    log "Stage 5 [$ds]: all baselines × ${#SEEDS[@]} seeds"
    for seed in "${SEEDS[@]}"; do
      python -m scripts.baseline_ft            --dataset "$ds" --forget_class "$FORGET_CLASS" --seed "$seed"
      python -m scripts.baseline_neggradplus   --dataset "$ds" --forget_class "$FORGET_CLASS" --seed "$seed"
      python -m scripts.baseline_random_labels --dataset "$ds" --forget_class "$FORGET_CLASS" --seed "$seed"
      python -m scripts.baseline_bad_teaching  --dataset "$ds" --forget_class "$FORGET_CLASS" --seed "$seed"
      python -m scripts.baseline_scrub         --dataset "$ds" --forget_class "$FORGET_CLASS" --seed "$seed"
      python -m scripts.baseline_retrain       --dataset "$ds" --forget_class "$FORGET_CLASS" --seed "$seed"
    done
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: HP OFAT sweep (CIFAR-10 only by default)
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "hp"; then
  if [[ " ${DATASETS[*]} " == *" cifar10 "* ]]; then
    sep
    log "Stage 6: HP ablation (CIFAR-10, 33 runs)"
    python -m scripts.run_hp_ablation --dataset cifar10 --forget_class "$FORGET_CLASS"
    python -m scripts.compare_hp_ablation --dataset cifar10
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6b: multi-class robustness (addresses reviewer C7)
#   Re-runs MASUC + baselines for 2 additional forget classes (5 and 7 on
#   CIFAR-10). Snapshots per-class results in results_classN/.
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "multiclass"; then
  for ds in "${DATASETS[@]}"; do
    if [[ "$ds" == "cifar10" ]]; then
      sep
      log "Stage 6b [$ds]: multi-class C7 sweep (classes 3 5 7 × ${#SEEDS[@]} seeds)"
      python -m scripts.run_multiclass --dataset "$ds" --classes 3 5 7 --seeds "${SEEDS[@]}"
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: aggregation + ECE + LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "aggregate"; then
  for ds in "${DATASETS[@]}"; do
    sep
    log "Stage 7 [$ds]: aggregate + calibration + LaTeX"
    python -m scripts.compare_ablation       --dataset "$ds" --seeds "${SEEDS[@]}"
    python -m scripts.compare_methods        --dataset "$ds" --seeds "${SEEDS[@]}"
    python -m scripts.eval_mia               --dataset "$ds" --forget_class "$FORGET_CLASS" --seeds "${SEEDS[@]}"
    python -m scripts.eval_calibration       --dataset "$ds" --forget_class "$FORGET_CLASS" --seeds "${SEEDS[@]}"
    python -m scripts.compute_total_cost     --dataset "$ds"
    python -m scripts.generate_paper_tables  --dataset "$ds"
  done
fi

sep
log "ALL DONE. Summaries in results/summaries/"
for ds in "${DATASETS[@]}"; do
  ls -la results/summaries/${ds}_*.md 2>/dev/null || true
done
