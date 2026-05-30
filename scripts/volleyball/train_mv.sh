#!/usr/bin/env bash
# Training Difix3D with the mv_unet (multi-view) backbone on volleyball data.
# Reference image is the previous frame's GT; for the first frame, the last
# frame of the sequence is used as the reference (see scripts/volleyball/add_ref_views.py).

set -euo pipefail

SCENE="${SCENE:-aria01}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/difix/train_volleyball_${SCENE}_mv}"
DATASET="${DATASET:-data/volleyball_static/ego_${SCENE}_finetune_mv.json}"
RUN_NAME="${RUN_NAME:-train_${SCENE}_mv}"

export NUM_NODES=1
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5}"

accelerate launch src/train_difix.py \
  --output_dir="${OUTPUT_DIR}" \
  --dataset_path="${DATASET}" \
  --pretrained_model_name_or_path nvidia/difix \
  --mv_unet \
  --learning_rate 1e-5 \
  --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 500 \
  --checkpointing_steps 2000 --eval_freq 200 --viz_freq 20 \
  --train_batch_size 1 --dataloader_num_workers 4 \
  --resolution 512 \
  --num_training_epochs 20 \
  --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "${RUN_NAME}" --timestep 199
