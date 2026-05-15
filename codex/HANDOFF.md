# Difix3D Handoff Context

## Current Goal

Continue from the dynamic3dgs benchmark work. The user asked to load the new trained checkpoint under `outputs/difix/train_dynamic3dgs`, run the benchmark using `dynamic3dgs` inputs, and save outputs with dynamic3dgs-specific names.

No benchmark process is currently running.

## Repository

Working directory:

```bash
/local/home/zinyou/projects/Difix3D
```

Sandboxed commands have often failed with `bwrap: loopback: Failed RTM_NEWADDR`; use escalated shell commands when needed.

## Important Existing Changes

Several files are modified/untracked and should be preserved. Do not reset or revert.

Modified tracked files:

- `src/model.py`
- `src/train_difix.py`
- `src/inference_difix.py`
- `src/dataset.py`

Important untracked artifacts:

- `scripts/prepare_egohuman_finetune.py`
- `scripts/prepare_egohuman_masks.py`
- `scripts/benchmark_egohuman_difix.py`
- `data/egohuman_finetune.json`
- `data/egohuman_dynamic3dgs.json`
- `data/ego_aria01/...`
- `data/ego_aria02/...`
- `data/ego_aria03/...`
- `data/ego_aria04/...`
- `outputs/difix/train_dynamic3dgs/...`
- `dynamic3dgs_ego.mp4`

## Completed Work Summary

### HF pretrained loading

`src/model.py`, `src/train_difix.py`, and `src/inference_difix.py` were updated so training/inference can initialize from a Hugging Face `DifixPipeline` repo such as `nvidia/difix`. The custom VAE skip connections and `vae_skip` LoRA adapter metadata are preserved/recreated.

### Finetune dataset

`scripts/prepare_egohuman_finetune.py` extracts `clean.mp4` and `corrupted.mp4` from `finetuning_data/ego_aria0x/` into:

- `data/ego_aria0x/clean`
- `data/ego_aria0x/corrupted`

It writes `data/egohuman_finetune.json`.

### Masks

`scripts/prepare_egohuman_masks.py` generates invalid-region masks in:

- `data/ego_aria0x/mask`

The mask convention is `1 = invalid/distractor`. Masks were eroded with `--erode_kernel 3`.

`src/dataset.py` now loads optional JSON `mask` paths and returns `invalid_pixel_mask`. `src/train_difix.py` masks invalid supervision regions before L2, LPIPS, and Gram losses.

### Benchmark

`scripts/benchmark_egohuman_difix.py` benchmarks corrupted/pretrained/finetuned outputs, creates comparison videos, and computes PSNR/SSIM/LPIPS(VGG). It supports:

- `--eval_only`
- `--metrics_only`
- masked metric computation by blacking invalid pixels in both prediction and clean before standard metrics
- `--fps` default `15`

The current script was further parameterized for dynamic3dgs:

- `--scenes`
- `--input_subdir`
- `--pretrained_output_subdir`
- `--finetuned_output_subdir`
- `--comparison_video_name`

This script has not yet been fully run for the new dynamic3dgs checkpoint in the current turn.

### dynamic3dgs frames

`dynamic3dgs_ego.mp4` is `1024x512`, `121` frames. It was split into two 512x512 streams:

- `data/ego_aria01/dynamic3dgs/000001.png` through `000121.png`
- `data/ego_aria02/dynamic3dgs/000001.png` through `000121.png`

### dynamic3dgs JSON

`data/egohuman_dynamic3dgs.json` was created:

- `train`: `aria01_000001` through `aria01_000121`
- `test`: `aria02_000001` through `aria02_000121`
- `image` points to `data/ego_aria0x/dynamic3dgs/000NNN.png`
- `target_image`, `prompt`, and `mask` are preserved from `data/egohuman_finetune.json`

## Current Pending Task

Run the dynamic3dgs benchmark with:

- input folder: `dynamic3dgs`
- pretrained output folder: `pretrained_dynamic3dgs`
- finetuned output folder: `finetuned_dynamic3dgs`
- comparison video name: `comparison_clean_dynamic3dgs_pretrained_dynamic3dgs_finetuned_dynamic3dgs.mp4`
- metrics path: `data/egohuman_benchamark_matrics_dynamic3dgs.json`
- checkpoint: latest in `outputs/difix/train_dynamic3dgs/checkpoints`, currently `model_3000.pkl`
- likely scenes: `ego_aria01 ego_aria02`, because dynamic3dgs frames exist only for those two

Use the project venv for full runs:

```bash
venv/bin/python ...
```

The base `python` environment previously lacked `diffusers`, while `venv/bin/python -c "import diffusers, torchmetrics"` passed.

