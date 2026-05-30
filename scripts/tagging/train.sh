export NUM_NODES=1
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch src/train_difix.py \
  --output_dir=outputs/difix/train_tagging \
  --dataset_path="data/004_tagging/aria_tagging_finetune.json" \
  --pretrained_model_name_or_path nvidia/difix \
  --learning_rate 1e-5 \
  --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 500 \
  --checkpointing_steps 2000 --eval_freq 200 --viz_freq 20 \
  --train_batch_size 1 --dataloader_num_workers 4 \
  --resolution 512 \
  --num_training_epochs 20 \
  --report_to "wandb" --tracker_project_name "difix" --tracker_run_name "train_tagging" --timestep 199
