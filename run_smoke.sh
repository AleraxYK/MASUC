#!/usr/bin/env bash
#
# MASUC smoke-test pipeline: 1 epoch, 1 seed, end-to-end.
# Verifies every script runs without errors. NOT meaningful results.
#
# Auto-enables AMP (mixed precision) when CUDA is detected.
# Auto-uses the parallel MASUC variant when CUDA is detected (use --no-parallel to disable).
#
# Usage:
#   bash run_smoke.sh                            # CIFAR-10, class 3, seed 42
#   bash run_smoke.sh --dataset mnist
#   bash run_smoke.sh --forget_class 5
#   bash run_smoke.sh --epochs 2
#   bash run_smoke.sh --no-parallel              # force sequential MASUC even on CUDA
#   bash run_smoke.sh --no-amp                   # disable mixed precision
#   CUDA_VISIBLE_DEVICES=0 bash run_smoke.sh

set -euo pipefail

DS="cifar10"
FC=3
SEED=42
EP=1
USE_PARALLEL=auto
USE_AMP=auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)       DS="$2"; shift 2;;
    --forget_class)  FC="$2"; shift 2;;
    --seed)          SEED="$2"; shift 2;;
    --epochs)        EP="$2"; shift 2;;
    --no-parallel)   USE_PARALLEL=no;  shift;;
    --no-amp)        USE_AMP=no;       shift;;
    -h|--help) sed -n '1,21p' "$0"; exit 0;;
    *) echo "Unknown: $1"; exit 1;;
  esac
done

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
sep() { echo "─────────────────────────────────────────────────────"; }

mkdir -p results/{checkpoints,reports,plots,splits,summaries}

# ─────────────────────────────────────────────────────────────────────────────
# Detect CUDA → decide AMP + parallel variant
# ─────────────────────────────────────────────────────────────────────────────
log "Env check"
HAS_CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

AMP_FLAG=""
MASUC_SCRIPT="scripts.run_masuc"
if [[ "$HAS_CUDA" == "1" ]]; then
  [[ "$USE_AMP"      != "no" ]] && AMP_FLAG="--amp"
  [[ "$USE_PARALLEL" != "no" ]] && MASUC_SCRIPT="scripts.run_masuc_parallel"
fi
echo "  → MASUC script: $MASUC_SCRIPT  |  AMP: $AMP_FLAG"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: pretrain student (skip if checkpoint exists)
# ─────────────────────────────────────────────────────────────────────────────
sep
CKPT="results/checkpoints/${DS}_model_before_best.pth"
if [[ -f "$CKPT" ]]; then
  log "Stage 1 [$DS]: pretrained checkpoint exists → skip ($CKPT)"
else
  log "Stage 1 [$DS]: pretrain student ($EP epoch)"
  python -m scripts.train_model --dataset "$DS" --epochs "$EP"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: teachers (skip if first teacher exists)
# ─────────────────────────────────────────────────────────────────────────────
sep
T0="results/checkpoints/${DS}_teacher_0.pth"
if [[ -f "$T0" ]]; then
  log "Stage 2 [$DS]: teachers already trained → skip"
else
  log "Stage 2 [$DS]: teachers ×5 ($EP epoch each)"
  python -m scripts.train_agents --dataset "$DS" --epochs "$EP"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: split (skip if exists)
# ─────────────────────────────────────────────────────────────────────────────
sep
SP="results/splits/${DS}_split_forget_${FC}.json"
if [[ -f "$SP" ]]; then
  log "Stage 3 [$DS]: split exists → skip ($SP)"
else
  log "Stage 3 [$DS]: split forget_class=$FC"
  python -m scripts.make_split --dataset "$DS" --forget_class "$FC"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: MASUC + 4 ablations × 1 seed × 1 epoch
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Stage 4: MASUC + 4 ablations (seed=$SEED, $EP epoch, $MASUC_SCRIPT $AMP_FLAG)"
for flag in "" "--no_ea" "--no_kd" "--no_ra" "--no_erasure"; do
  echo ">>> MASUC ${flag:-full}"
  python -m "$MASUC_SCRIPT" --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP" $AMP_FLAG $flag
done

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: baselines × 1 seed × 1 epoch
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Stage 5: baselines (seed=$SEED, $EP epoch)"
python -m scripts.baseline_ft            --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_neggradplus   --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_random_labels --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_bad_teaching  --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_scrub         --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_retrain       --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: aggregation (CPU-only, parallel safe)
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Stage 6: aggregation (parallel, CPU)"
python -m scripts.compare_ablation   --dataset "$DS" --seeds "$SEED" &
python -m scripts.compare_methods    --dataset "$DS" --seeds "$SEED" --before_retain 0.8279 --before_forget 0.8200 &
python -m scripts.compute_total_cost --dataset "$DS" &
wait

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: MIA + Calibration eval (sequenziale, GPU)
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Stage 7: MIA + Calibration eval"
python -m scripts.eval_mia         --dataset "$DS" --forget_class "$FC" --seeds "$SEED"
python -m scripts.eval_calibration --dataset "$DS" --forget_class "$FC" --seeds "$SEED"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: LaTeX paper tables
# ─────────────────────────────────────────────────────────────────────────────
sep
log "Stage 8: generate LaTeX paper tables"
python -m scripts.generate_paper_tables --dataset "$DS"

sep
log "SMOKE TEST DONE — see results/summaries/ and paper_tables/"
ls -la results/summaries/${DS}_*.md 2>/dev/null || true
ls -la paper_tables/${DS}/*.tex 2>/dev/null || true
