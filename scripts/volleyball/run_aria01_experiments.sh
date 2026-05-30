#!/usr/bin/env bash
# Sequentially: full-FT train -> LoRA train -> benchmark both on ego_aria01.
# Runs on GPUs 0,1,4,5 (set inside each train script). Streams to per-stage logs.

set -euo pipefail
cd /local/home/zinyou/projects/Difix3D

LOG_DIR=logs/aria01_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR"

FULLFT_OUT=outputs/difix/train_volleyball_aria01
LORA_OUT=outputs/difix/train_volleyball_aria01_lora
METRICS_OUT=data/tagging_zjumocap/volleyball_aria01_benchmark_metrics.json

echo "================ [1/3] Full-FT train ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
bash scripts/volleyball/train.sh 2>&1 | tee "$LOG_DIR/01_fullft_train.log"

echo "================ [2/3] LoRA train ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
bash scripts/volleyball/train_lora.sh 2>&1 | tee "$LOG_DIR/02_lora_train.log"

# Pick the final checkpoint from each run
FULLFT_CKPT=$(ls -1v "$FULLFT_OUT"/checkpoints/model_*.pkl 2>/dev/null | tail -1)
LORA_CKPT=$(ls -1v "$LORA_OUT"/checkpoints/model_*.pkl 2>/dev/null | tail -1)
echo "Full-FT ckpt: $FULLFT_CKPT" | tee -a "$LOG_DIR/00_summary.log"
echo "LoRA ckpt:    $LORA_CKPT" | tee -a "$LOG_DIR/00_summary.log"

if [[ -z "$FULLFT_CKPT" || -z "$LORA_CKPT" ]]; then
  echo "Missing checkpoint(s); aborting benchmark." | tee -a "$LOG_DIR/00_summary.log"
  exit 1
fi

echo "================ [3/3] Benchmark (full-FT) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
CUDA_VISIBLE_DEVICES=0 python scripts/common/benchmark.py \
  --data_root data_volleyball \
  --scenes ego_aria01 \
  --finetuned_checkpoint "$FULLFT_CKPT" \
  --pretrained_output_subdir pretrained_aria01 \
  --finetuned_output_subdir finetuned_aria01_fullft \
  --comparison_video_name comparison_aria01_fullft.mp4 \
  --metrics_path "${METRICS_OUT%.json}_fullft.json" \
  --split_json data/volleyball_static/ego_aria01_finetune.json \
  --split test 2>&1 | tee "$LOG_DIR/03_benchmark_fullft.log"

echo "================ [3/3] Benchmark (LoRA) ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
CUDA_VISIBLE_DEVICES=0 python scripts/common/benchmark.py \
  --data_root data_volleyball \
  --scenes ego_aria01 \
  --finetuned_checkpoint "$LORA_CKPT" \
  --pretrained_output_subdir pretrained_aria01 \
  --finetuned_output_subdir finetuned_aria01_lora \
  --comparison_video_name comparison_aria01_lora.mp4 \
  --metrics_path "${METRICS_OUT%.json}_lora.json" \
  --split_json data/volleyball_static/ego_aria01_finetune.json \
  --split test 2>&1 | tee "$LOG_DIR/04_benchmark_lora.log"

echo "================ DONE ================" | tee -a "$LOG_DIR/00_summary.log"
date | tee -a "$LOG_DIR/00_summary.log"
echo "Metrics:" | tee -a "$LOG_DIR/00_summary.log"
ls -la "${METRICS_OUT%.json}_fullft.json" "${METRICS_OUT%.json}_lora.json" 2>&1 | tee -a "$LOG_DIR/00_summary.log"
