#!/usr/bin/env bash
#
# Full reproduction pipeline for MASUC.
# Designed for single-GPU Linux/CUDA box (e.g. RTX 3070).
#
# Stages (skippable via flags):
#   0. Env check
#   1. Pre-train student (per dataset)
#   2. Train teacher society (per dataset)
#   3. Generate forget/retain splits (per dataset, per forget class)
#   4. MASUC + ablations × seeds
#   5. Baselines × seeds
#   6. HP ablation (CIFAR-10 only, optional)
#   7. Aggregation: compare_ablation, compare_methods, eval_mia, compute_total_cost
#
# Usage:
#   bash run_all.sh                                  # default: all stages, CIFAR-10 only
#   bash run_all.sh --datasets cifar10 mnist         # multi-dataset
#   bash run_all.sh --seeds 42 123 7 2024            # custom seeds
#   bash run_all.sh --skip pretrain teachers hp      # skip stages
#   bash run_all.sh --forget_class 3                 # target class
#   CUDA_VISIBLE_DEVICES=0 bash run_all.sh           # pin GPU

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DATASETS=("cifar10")
SEEDS=(42 123 7)
FORGET_CLASS=3
SKIP=()

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
    --forget_class)
      shift; FORGET_CLASS="$1"; shift
      ;;
    -h|--help)
      sed -n '1,30p' "$0"; exit 0
      ;;
    *)
      echo "Unknown arg: $1"; exit 1
      ;;
  esac
done

skip_has() {
  local needle="$1"; shift
  for s in "${SKIP[@]:-}"; do [[ "$s" == "$needle" ]] && return 0; done
  return 1
}

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: env check
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 0: env check"
python - <<'PY'
import torch, sys
print(f"  torch={torch.__version__}")
if torch.cuda.is_available():
    print(f"  CUDA={torch.version.cuda} | device={torch.cuda.get_device_name(0)} | VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
elif torch.backends.mps.is_available():
    print("  MPS available (Apple Silicon)")
else:
    print("  CPU only — pipeline will be very slow")
PY

mkdir -p results/{checkpoints,reports,plots,splits,summaries}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: pre-train student
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "pretrain"; then
  for ds in "${DATASETS[@]}"; do
    CKPT="results/checkpoints/${ds}_model_before_best.pth"
    if [[ -f "$CKPT" ]]; then
      log "Stage 1 [$ds]: pretrained checkpoint exists → skip ($CKPT)"
    else
      log "Stage 1 [$ds]: pretrain student"
      python -m scripts.train_model --dataset "$ds"
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: teacher society
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "teachers"; then
  for ds in "${DATASETS[@]}"; do
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
# Stage 3: splits
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "splits"; then
  for ds in "${DATASETS[@]}"; do
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
# Stage 4: MASUC + ablations
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "ablation"; then
  for ds in "${DATASETS[@]}"; do
    log "Stage 4 [$ds]: MASUC + 4 ablations × ${#SEEDS[@]} seeds"
    python -m scripts.run_ablation --dataset "$ds" --forget_class "$FORGET_CLASS" --seeds "${SEEDS[@]}"
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: baselines
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "baselines"; then
  for ds in "${DATASETS[@]}"; do
    log "Stage 5 [$ds]: all baselines × ${#SEEDS[@]} seeds"
    python -m scripts.run_all_baselines --dataset "$ds" --forget_class "$FORGET_CLASS" --seeds "${SEEDS[@]}"
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: HP ablation (CIFAR-10 only by default)
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "hp"; then
  if [[ " ${DATASETS[*]} " == *" cifar10 "* ]]; then
    log "Stage 6: HP ablation (CIFAR-10, 33 runs)"
    python -m scripts.run_hp_ablation --dataset cifar10 --forget_class "$FORGET_CLASS"
    python -m scripts.compare_hp_ablation --dataset cifar10
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: aggregation
# ─────────────────────────────────────────────────────────────────────────────
if ! skip_has "aggregate"; then
  for ds in "${DATASETS[@]}"; do
    log "Stage 7 [$ds]: aggregate"
    python -m scripts.compare_ablation --dataset "$ds" --seeds "${SEEDS[@]}"
    python -m scripts.compare_methods  --dataset "$ds" --seeds "${SEEDS[@]}"
    python -m scripts.eval_mia         --dataset "$ds" --forget_class "$FORGET_CLASS" --seeds "${SEEDS[@]}"
    python -m scripts.compute_total_cost --dataset "$ds"
  done
fi

log "ALL DONE. Summaries in results/summaries/"
