#!/usr/bin/env bash
#
# Resume run_c7.sh from the interruption point.
#
# State at interruption (2026-06-05):
#   Class 5 → DONE (snapshot in results_class5/)
#   Class 7:
#     seed=42  → DONE (tutti i metodi)
#     seed=123 → DONE (tutti i metodi)
#     seed=7   → masuc + ft + neggradplus DONE; riprende da random_labels
#
# Usage: bash run_c7_resume.sh [--no-amp]

set -euo pipefail

SEEDS=(42 123 7)
DS="cifar10"
USE_AMP=auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-amp) USE_AMP=no; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
sep() { echo "═════════════════════════════════════════════════════════════"; }

HAS_CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
AMP_FLAG=""
[[ "$HAS_CUDA" == "1" && "$USE_AMP" != "no" ]] && AMP_FLAG="--amp"

snapshot_class() {
  local fc="$1"
  local CLASS_DIR="results_class${fc}"
  mkdir -p "${CLASS_DIR}/reports" "${CLASS_DIR}/checkpoints"
  log "  Snapshotting results → ${CLASS_DIR}/"
  for f in results/reports/${DS}_*_seed*_curve.json; do
    [[ -f "$f" ]] && mv "$f" "${CLASS_DIR}/reports/"
  done
  for f in results/checkpoints/${DS}_*_seed*.pth; do
    [[ -f "$f" ]] && mv "$f" "${CLASS_DIR}/checkpoints/"
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# CLASS 5 — DONE (snapshot in results_class5/)
# ─────────────────────────────────────────────────────────────────────────────
# sep
# log "CLASS 5 resume"
#
# log "  seed=123 — SCRUB (restarting after interrupt)"
# python -m scripts.baseline_scrub \
#   --dataset "$DS" --forget_class 5 --seed 123
#
# log "  seed=123 — Retrain"
# python -m scripts.baseline_retrain \
#   --dataset "$DS" --forget_class 5 --seed 123
#
# log "  seed=7 — all methods"
# python -m scripts.run_masuc \
#   --dataset "$DS" --forget_class 5 --seed 7 $AMP_FLAG
#
# python -m scripts.baseline_ft \
#   --dataset "$DS" --forget_class 5 --seed 7
# python -m scripts.baseline_neggradplus \
#   --dataset "$DS" --forget_class 5 --seed 7
# python -m scripts.baseline_random_labels \
#   --dataset "$DS" --forget_class 5 --seed 7
# python -m scripts.baseline_bad_teaching \
#   --dataset "$DS" --forget_class 5 --seed 7
# python -m scripts.baseline_scrub \
#   --dataset "$DS" --forget_class 5 --seed 7
# python -m scripts.baseline_retrain \
#   --dataset "$DS" --forget_class 5 --seed 7
#
# snapshot_class 5
# log "  Class 5 DONE."

# ─────────────────────────────────────────────────────────────────────────────
# CLASS 7 — all seeds from scratch
# ─────────────────────────────────────────────────────────────────────────────
sep
log "CLASS 7"

SP="results/splits/${DS}_split_forget_7.json"
if [[ -f "$SP" ]]; then
  log "  Split exists → skip ($SP)"
else
  log "  Generating split for class 7"
  python -m scripts.make_split --dataset "$DS" --forget_class 7
fi

# seed=42 — DONE
# log "  seed=42"
# python -m scripts.run_masuc \
#   --dataset "$DS" --forget_class 7 --seed 42 $AMP_FLAG      # DONE
# python -m scripts.baseline_ft \
#   --dataset "$DS" --forget_class 7 --seed 42                 # DONE
# python -m scripts.baseline_neggradplus \
#   --dataset "$DS" --forget_class 7 --seed 42                 # DONE
# python -m scripts.baseline_random_labels \
#   --dataset "$DS" --forget_class 7 --seed 42                 # DONE
# python -m scripts.baseline_bad_teaching \
#   --dataset "$DS" --forget_class 7 --seed 42                 # DONE
# python -m scripts.baseline_scrub \
#   --dataset "$DS" --forget_class 7 --seed 42                 # DONE
# python -m scripts.baseline_retrain \
#   --dataset "$DS" --forget_class 7 --seed 42                 # DONE

# seed=123 — DONE
# log "  seed=123"
# python -m scripts.run_masuc \
#   --dataset "$DS" --forget_class 7 --seed 123 $AMP_FLAG     # DONE
# python -m scripts.baseline_ft \
#   --dataset "$DS" --forget_class 7 --seed 123               # DONE
# python -m scripts.baseline_neggradplus \
#   --dataset "$DS" --forget_class 7 --seed 123               # DONE
# python -m scripts.baseline_random_labels \
#   --dataset "$DS" --forget_class 7 --seed 123               # DONE
# python -m scripts.baseline_bad_teaching \
#   --dataset "$DS" --forget_class 7 --seed 123               # DONE
# python -m scripts.baseline_scrub \
#   --dataset "$DS" --forget_class 7 --seed 123               # DONE
# python -m scripts.baseline_retrain \
#   --dataset "$DS" --forget_class 7 --seed 123               # DONE

# seed=7 — masuc + ft + neggradplus DONE; riprende da random_labels
log "  seed=7 — resume from random_labels"
# python -m scripts.run_masuc \
#   --dataset "$DS" --forget_class 7 --seed 7 $AMP_FLAG       # DONE
# python -m scripts.baseline_ft \
#   --dataset "$DS" --forget_class 7 --seed 7                  # DONE
# python -m scripts.baseline_neggradplus \
#   --dataset "$DS" --forget_class 7 --seed 7                  # DONE
python -m scripts.baseline_random_labels \
  --dataset "$DS" --forget_class 7 --seed 7
python -m scripts.baseline_bad_teaching \
  --dataset "$DS" --forget_class 7 --seed 7
python -m scripts.baseline_scrub \
  --dataset "$DS" --forget_class 7 --seed 7
python -m scripts.baseline_retrain \
  --dataset "$DS" --forget_class 7 --seed 7

snapshot_class 7
log "  Class 7 DONE."

# ─────────────────────────────────────────────────────────────────────────────
# Aggregation — classes 3, 5, 7
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Aggregating classes 3 5 7"
python -m scripts.run_multiclass \
  --dataset "$DS" --classes 3 5 7 --seeds "${SEEDS[@]}" --skip_run

sep
log "RESUME DONE — results/summaries/${DS}_multiclass_stats.md"
cat "results/summaries/${DS}_multiclass_stats.md"
