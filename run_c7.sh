#!/usr/bin/env bash
#
# C7 fix: run MASUC + all baselines for forget classes 5 and 7 on CIFAR-10.
# Requires existing student + teacher checkpoints (stages 1-2 already done).
#
# Usage:
#   bash run_c7.sh
#   bash run_c7.sh --seeds 42 123 7
#   bash run_c7.sh --no-amp

set -euo pipefail

SEEDS=(42 123 7)
DS="cifar10"
CLASSES=(5 7)
USE_AMP=auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      shift; SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
      ;;
    --no-amp) USE_AMP=no; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
sep() { echo "═════════════════════════════════════════════════════════════"; }

HAS_CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
AMP_FLAG=""
[[ "$HAS_CUDA" == "1" && "$USE_AMP" != "no" ]] && AMP_FLAG="--amp"

sep
log "C7 multiclass | classes: ${CLASSES[*]} | seeds: ${SEEDS[*]} | AMP: ${AMP_FLAG:-disabled}"

# ─────────────────────────────────────────────────────────────────────────────
# Per forget class
# ─────────────────────────────────────────────────────────────────────────────
for fc in "${CLASSES[@]}"; do
  sep
  log "Forget class $fc"

  # Split
  SP="results/splits/${DS}_split_forget_${fc}.json"
  if [[ -f "$SP" ]]; then
    log "  Split exists → skip ($SP)"
  else
    log "  Generating split for class $fc"
    python -m scripts.make_split --dataset "$DS" --forget_class "$fc"
  fi

  # MASUC + baselines × seeds
  CLASS_DIR="results_class${fc}"
  mkdir -p "${CLASS_DIR}/reports" "${CLASS_DIR}/checkpoints"

  for seed in "${SEEDS[@]}"; do
    log "  seed=$seed"

    python -m scripts.run_masuc \
      --dataset "$DS" --forget_class "$fc" --seed "$seed" $AMP_FLAG

    python -m scripts.baseline_ft \
      --dataset "$DS" --forget_class "$fc" --seed "$seed"
    python -m scripts.baseline_neggradplus \
      --dataset "$DS" --forget_class "$fc" --seed "$seed"
    python -m scripts.baseline_random_labels \
      --dataset "$DS" --forget_class "$fc" --seed "$seed"
    python -m scripts.baseline_bad_teaching \
      --dataset "$DS" --forget_class "$fc" --seed "$seed"
    python -m scripts.baseline_scrub \
      --dataset "$DS" --forget_class "$fc" --seed "$seed"
    python -m scripts.baseline_retrain \
      --dataset "$DS" --forget_class "$fc" --seed "$seed"
  done

  # Snapshot results for this class
  log "  Snapshotting results → ${CLASS_DIR}/"
  for f in results/reports/${DS}_*_seed*_curve.json; do
    [[ -f "$f" ]] && mv "$f" "${CLASS_DIR}/reports/"
  done
  for f in results/checkpoints/${DS}_*_seed*.pth; do
    [[ -f "$f" ]] && mv "$f" "${CLASS_DIR}/checkpoints/"
  done

  log "  Class $fc done."
done

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate all 3 classes (3 already in results_class3/)
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Aggregating classes 3 5 7"
python -m scripts.run_multiclass \
  --dataset "$DS" --classes 3 5 7 --seeds "${SEEDS[@]}" --skip_run

sep
log "C7 DONE — results/summaries/${DS}_multiclass_stats.md"
cat "results/summaries/${DS}_multiclass_stats.md"
