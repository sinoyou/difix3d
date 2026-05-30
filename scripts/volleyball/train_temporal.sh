#!/usr/bin/env bash
# Training Difix3D with a temporal warping loss on volleyball data.
# Each training step samples a clip of CLIP_LENGTH consecutive frames; the
# warping loss compares cross-warped predictions over forward-backward
# consistent regions of GT optical flow (torchvision RAFT).

set -euo pipefail

SCENE="${SCENE:-aria01}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/difix/train_volleyball_${SCENE}_temporal}"
DATASET="${DATASET:-data/volleyball_static/ego_${SCENE}_finetune.json}"
RUN_NAME="${RUN_NAME:-train_${SCENE}_temporal}"
CLIP_LENGTH="${CLIP_LENGTH:-3}"
LAMBDA_TEMPORAL="${LAMBDA_TEMPORAL:-1.0}"
RAFT_VARIANT="${RAFT_VARIANT:-large}"

export NUM_NODES=1
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5}"

accelerate launch src/train_difix_temporal.py \
  --output_dir="${OUTPUT_DIR}" \
  --dataset_path="${DATASET}" \
  --pretrained_model_name_or_path nvidia/difix \
  --clip_length "${CLIP_LENGTH}" \
  --lambda_temporal "${LAMBDA_TEMPORAL}" \
  --raft_variant "${RAFT_VARIANT}" \
  --learning_rate 1e-5 \
  --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 500 \
  --checkpointing_steps 2000 --eval_freq 200 --viz_freq 20 \
  --train_batch_size 1 --dataloader_num_workers 4 \
  --resolution 512 \
  --num_training_epochs 20 \
  --gradient_checkpointing \
  --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "${RUN_NAME}" --timestep 199
