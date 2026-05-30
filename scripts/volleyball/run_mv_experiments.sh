#!/usr/bin/env bash
# Sequentially: train mv_unet on aria01 -> train mv_unet on aria04 ->
# autoregressive benchmark on both. Uses GPUs 0,1,4,5 (set inside train script).

set -euo pipefail
cd /local/home/zinyou/projects/Difix3D
export PATH=/local/home/zinyou/projects/Difix3D/venv/bin:$PATH

LOG_DIR=logs/volleyball_mv_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR"

ARIA01_OUT=outputs/difix/train_volleyball_aria01_mv
ARIA04_OUT=outputs/difix/train_volleyball_aria04_mv
METRICS_BASE=data/volleyball_static/volleyball_mv_benchmark_metrics

# --- 1) Train aria01 (mv_unet) ---
echo "================ [1/4] Train aria01 (mv_unet) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
SCENE=aria01 \
OUTPUT_DIR="$ARIA01_OUT" \
DATASET="data/volleyball_static/ego_aria01_finetune_mv.json" \
RUN_NAME="train_aria01_mv" \
bash scripts/volleyball/train_mv.sh 2>&1 | tee "$LOG_DIR/01_aria01_train.log"

# --- 2) Train aria04 (mv_unet) ---
echo "================ [2/4] Train aria04 (mv_unet) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
SCENE=aria04 \
OUTPUT_DIR="$ARIA04_OUT" \
DATASET="data/volleyball_static/ego_aria04_finetune_mv.json" \
RUN_NAME="train_aria04_mv" \
bash scripts/volleyball/train_mv.sh 2>&1 | tee "$LOG_DIR/02_aria04_train.log"

# Pick the final checkpoint for each
ARIA01_CKPT=$(ls -1v "$ARIA01_OUT"/checkpoints/model_*.pkl 2>/dev/null | tail -1 || true)
ARIA04_CKPT=$(ls -1v "$ARIA04_OUT"/checkpoints/model_*.pkl 2>/dev/null | tail -1 || true)
echo "aria01 ckpt: $ARIA01_CKPT" | tee -a "$LOG_DIR/00_summary.log"
echo "aria04 ckpt: $ARIA04_CKPT" | tee -a "$LOG_DIR/00_summary.log"

if [[ -z "$ARIA01_CKPT" || -z "$ARIA04_CKPT" ]]; then
  echo "Missing checkpoint(s); aborting benchmark." | tee -a "$LOG_DIR/00_summary.log"
  exit 1
fi

# --- 3) Autoregressive benchmark aria01 ---
echo "================ [3/4] AR benchmark aria01 ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
CUDA_VISIBLE_DEVICES=0 python scripts/volleyball/benchmark_mv_autoregressive.py \
  --data_root data_volleyball \
  --scene ego_aria01 \
  --split_json data/volleyball_static/ego_aria01_finetune_mv.json \
  --split test \
  --finetuned_checkpoint "$ARIA01_CKPT" \
  --output_subdir finetuned_aria01_mv_ar \
  --comparison_video_name comparison_aria01_mv_ar.mp4 \
  --metrics_path "${METRICS_BASE}_aria01.json" \
  --overwrite 2>&1 | tee "$LOG_DIR/03_aria01_ar_benchmark.log"

# --- 4) Autoregressive benchmark aria04 ---
echo "================ [4/4] AR benchmark aria04 ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
CUDA_VISIBLE_DEVICES=0 python scripts/volleyball/benchmark_mv_autoregressive.py \
  --data_root data_volleyball \
  --scene ego_aria04 \
  --split_json data/volleyball_static/ego_aria04_finetune_mv.json \
  --split test \
  --finetuned_checkpoint "$ARIA04_CKPT" \
  --output_subdir finetuned_aria04_mv_ar \
  --comparison_video_name comparison_aria04_mv_ar.mp4 \
  --metrics_path "${METRICS_BASE}_aria04.json" \
  --overwrite 2>&1 | tee "$LOG_DIR/04_aria04_ar_benchmark.log"

echo "================ DONE ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
ls -la "${METRICS_BASE}_aria01.json" "${METRICS_BASE}_aria04.json" 2>&1 | tee -a "$LOG_DIR/00_summary.log"
