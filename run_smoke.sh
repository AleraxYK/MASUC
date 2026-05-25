#!/usr/bin/env bash
#
# Smoke-test pipeline: 1 epoch each, 1 seed.
# Verifies every script runs end-to-end. NOT meaningful results.
#
# Designed for RTX 2080 / single-GPU CUDA. Parallelizes only what's safe on a
# single GPU (i.e. CPU-bound aggregation steps).
#
# Usage:
#   bash run_smoke.sh                            # CIFAR-10, class 3, seed 42
#   bash run_smoke.sh --dataset mnist            # other dataset
#   bash run_smoke.sh --forget_class 5
#   CUDA_VISIBLE_DEVICES=0 bash run_smoke.sh

set -euo pipefail

DS="cifar10"
FC=3
SEED=42
EP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)      DS="$2"; shift 2;;
    --forget_class) FC="$2"; shift 2;;
    --seed)         SEED="$2"; shift 2;;
    --epochs)       EP="$2"; shift 2;;
    -h|--help) sed -n '1,15p' "$0"; exit 0;;
    *) echo "Unknown: $1"; exit 1;;
  esac
done

log() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }

mkdir -p results/{checkpoints,reports,plots,splits,summaries}

# ─────────────────────────────────────────────────────────────────────────────
# Env
# ─────────────────────────────────────────────────────────────────────────────
log "Env check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: pretrain student (1 epoch)
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 1: pretrain student ($EP epoch)"
python -m scripts.train_model --dataset "$DS" --epochs "$EP"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: teachers (1 epoch each)
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 2: teachers ×5 ($EP epoch each)"
python -m scripts.train_agents --dataset "$DS" --epochs "$EP"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: split
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 3: split forget_class=$FC"
python -m scripts.make_split --dataset "$DS" --forget_class "$FC"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: MASUC + 4 ablations × 1 seed × 1 epoch
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 4: MASUC + 4 ablations (1 seed × $EP epoch)"
for flag in "" "--no_ea" "--no_kd" "--no_ra" "--no_erasure"; do
  echo ">>> MASUC ${flag:-full}"
  python -m scripts.run_masuc --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP" $flag
done

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: baselines × 1 seed × 1 epoch
#   NOTE: sequenziale — su single GPU lanciarli in parallelo causa contention
#   (OOM o slowdown). Restano sequenziali ma rapidi a 1 epoca.
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 5: baselines (1 seed × $EP epoch)"
python -m scripts.baseline_ft            --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_neggradplus   --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_random_labels --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_bad_teaching  --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_scrub         --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"
python -m scripts.baseline_retrain       --dataset "$DS" --forget_class "$FC" --seed "$SEED" --epochs "$EP"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: aggregation — CPU-only, parallelizzabili in background
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 6: aggregation (parallel, CPU)"
python -m scripts.compare_ablation   --dataset "$DS" --seeds "$SEED" &
python -m scripts.compare_methods    --dataset "$DS" --seeds "$SEED" --before_retain 0.8279 --before_forget 0.8200 &
python -m scripts.compute_total_cost --dataset "$DS" &
wait

# eval_mia DEVE girare in foreground (usa GPU per forward pass su tutti i ckpt)
log "Stage 7: MIA eval (sequenziale, GPU)"
python -m scripts.eval_mia --dataset "$DS" --forget_class "$FC" --seeds "$SEED"

log "SMOKE TEST DONE — see results/summaries/"
ls -la results/summaries/${DS}_*.md 2>/dev/null || true
