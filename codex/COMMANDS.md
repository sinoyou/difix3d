# Useful Commands

## Check Current State

```bash
git status --short
find outputs/difix/train_dynamic3dgs/checkpoints -maxdepth 1 -type f | sort
find data/ego_aria01/dynamic3dgs data/ego_aria02/dynamic3dgs -maxdepth 1 -type f -name '*.png' | wc -l
```

## Syntax Check

```bash
python -m py_compile scripts/benchmark_egohuman_difix.py
```

If imports matter, use:

```bash
venv/bin/python -m py_compile scripts/benchmark_egohuman_difix.py
```

## Dynamic3DGS Benchmark Command

Run this from repo root:

```bash
venv/bin/python scripts/benchmark_egohuman_difix.py \
  --scenes ego_aria01 ego_aria02 \
  --input_subdir dynamic3dgs \
  --pretrained_output_subdir pretrained_dynamic3dgs \
  --finetuned_output_subdir finetuned_dynamic3dgs \
  --comparison_video_name comparison_clean_dynamic3dgs_pretrained_dynamic3dgs_finetuned_dynamic3dgs.mp4 \
  --metrics_path data/egohuman_benchamark_matrics_dynamic3dgs.json \
  --finetuned_checkpoint outputs/difix/train_dynamic3dgs/checkpoints/model_3000.pkl \
  --image_size 512 \
  --fps 15 \
  --overwrite
```

## Metrics Only For Dynamic3DGS

After outputs exist, recompute masked metrics without inference/video generation:

```bash
venv/bin/python scripts/benchmark_egohuman_difix.py \
  --scenes ego_aria01 ego_aria02 \
  --input_subdir dynamic3dgs \
  --pretrained_output_subdir pretrained_dynamic3dgs \
  --finetuned_output_subdir finetuned_dynamic3dgs \
  --metrics_path data/egohuman_benchamark_matrics_dynamic3dgs.json \
  --eval_only
```

## Verify Dynamic3DGS Outputs

```bash
find data/ego_aria01/pretrained_dynamic3dgs -maxdepth 1 -type f -name '*.png' | wc -l
find data/ego_aria02/pretrained_dynamic3dgs -maxdepth 1 -type f -name '*.png' | wc -l
find data/ego_aria01/finetuned_dynamic3dgs -maxdepth 1 -type f -name '*.png' | wc -l
find data/ego_aria02/finetuned_dynamic3dgs -maxdepth 1 -type f -name '*.png' | wc -l
python -m json.tool data/egohuman_benchamark_matrics_dynamic3dgs.json >/tmp/dynamic3dgs_metrics.valid.json
```

## Video FPS Verification

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=avg_frame_rate,nb_read_frames,duration \
  -of default=nokey=0:noprint_wrappers=1 \
  data/ego_aria01/comparison_clean_dynamic3dgs_pretrained_dynamic3dgs_finetuned_dynamic3dgs.mp4
```

