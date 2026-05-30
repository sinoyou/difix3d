#!/usr/bin/env bash
# Sequentially train Difix with the temporal warping loss on aria01 and aria04,
# then benchmark both checkpoints with the standard (non-mv) benchmark script.

set -euo pipefail
cd /local/home/zinyou/projects/Difix3D
export PATH=/local/home/zinyou/projects/Difix3D/venv/bin:$PATH

LOG_DIR=logs/volleyball_temporal_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR"

ARIA01_OUT=outputs/difix/train_volleyball_aria01_temporal
ARIA04_OUT=outputs/difix/train_volleyball_aria04_temporal
METRICS_BASE=data/volleyball_static/volleyball_temporal_benchmark_metrics

# --- 1) Train aria01 ---
echo "================ [1/4] Train aria01 (temporal) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
SCENE=aria01 \
OUTPUT_DIR="$ARIA01_OUT" \
DATASET="data/volleyball_static/ego_aria01_finetune.json" \
RUN_NAME="train_aria01_temporal" \
bash scripts/volleyball/train_temporal.sh 2>&1 | tee "$LOG_DIR/01_aria01_train.log"

# --- 2) Train aria04 ---
echo "================ [2/4] Train aria04 (temporal) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
SCENE=aria04 \
OUTPUT_DIR="$ARIA04_OUT" \
DATASET="data/volleyball_static/ego_aria04_finetune.json" \
RUN_NAME="train_aria04_temporal" \
bash scripts/volleyball/train_temporal.sh 2>&1 | tee "$LOG_DIR/02_aria04_train.log"

ARIA01_CKPT=$(ls -1v "$ARIA01_OUT"/checkpoints/model_*.pkl 2>/dev/null | tail -1 || true)
ARIA04_CKPT=$(ls -1v "$ARIA04_OUT"/checkpoints/model_*.pkl 2>/dev/null | tail -1 || true)
echo "aria01 ckpt: $ARIA01_CKPT" | tee -a "$LOG_DIR/00_summary.log"
echo "aria04 ckpt: $ARIA04_CKPT" | tee -a "$LOG_DIR/00_summary.log"

if [[ -z "$ARIA01_CKPT" || -z "$ARIA04_CKPT" ]]; then
  echo "Missing checkpoint(s); aborting benchmark." | tee -a "$LOG_DIR/00_summary.log"
  exit 1
fi

# --- 3) Benchmark aria01 ---
echo "================ [3/4] Benchmark aria01 (temporal) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
CUDA_VISIBLE_DEVICES=0 python scripts/common/benchmark.py \
  --data_root data_volleyball \
  --scenes ego_aria01 \
  --finetuned_checkpoint "$ARIA01_CKPT" \
  --pretrained_output_subdir pretrained_aria01 \
  --finetuned_output_subdir finetuned_aria01_temporal \
  --comparison_video_name comparison_aria01_temporal.mp4 \
  --metrics_path "${METRICS_BASE}_aria01.json" \
  --split_json data/volleyball_static/ego_aria01_finetune.json \
  --split test 2>&1 | tee "$LOG_DIR/03_aria01_benchmark.log"

# --- 4) Benchmark aria04 ---
echo "================ [4/4] Benchmark aria04 (temporal) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
CUDA_VISIBLE_DEVICES=0 python scripts/common/benchmark.py \
  --data_root data_volleyball \
  --scenes ego_aria04 \
  --finetuned_checkpoint "$ARIA04_CKPT" \
  --pretrained_output_subdir pretrained_aria04 \
  --finetuned_output_subdir finetuned_aria04_temporal \
  --comparison_video_name comparison_aria04_temporal.mp4 \
  --metrics_path "${METRICS_BASE}_aria04.json" \
  --split_json data/volleyball_static/ego_aria04_finetune.json \
  --split test 2>&1 | tee "$LOG_DIR/04_aria04_benchmark.log"

echo "================ DONE ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
ls -la "${METRICS_BASE}_aria01.json" "${METRICS_BASE}_aria04.json" 2>&1 | tee -a "$LOG_DIR/00_summary.log"
