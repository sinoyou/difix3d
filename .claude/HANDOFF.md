# Difix3D Handoff Context

Last updated: 2026-05-30. Active task: finetuning Difix3D on the
`egohuman_processed/004_tagging_extraction` tagging dataset.

## Repository

```bash
/local/home/zinyou/projects/Difix3D
```

The project venv is conda-style (no `bin/activate`) at
`/local/home/zinyou/projects/Difix3D/venv`. Activate by prepending PATH:

```bash
PATH=/local/home/zinyou/projects/Difix3D/venv/bin:$PATH
```

The base `python` lacks `lpips`/`diffusers`; always use the venv for runs.

Sandboxed commands have historically failed with
`bwrap: loopback: Failed RTM_NEWADDR`; use escalated shell when needed.

## Current Goal

Finetune `nvidia/difix` using `egohuman_processed/004_tagging_extraction`:

- `image` (corrupted)   = `egohuman_processed/004_tagging_extraction/rgb_blend/{scene}_{idx}.png` (512×512)
- `target_image` (GT)   = `egohuman_processed/004_tagging_extraction/rgb/{scene}_{idx}.png`       (1408×1408)
- `mask` (invalid)      = `egohuman_processed/004_tagging_extraction/mask/{scene}_{idx}/_invalid.png` (1408×1408)

Mask convention is `1 = invalid`; the existing `src/dataset.py:PairedDataset`
already loads JSON `mask` as `invalid_pixel_mask`, and `src/train_difix.py`
does `valid_mask = 1.0 - invalid_mask` and applies it to L2/LPIPS/Gram — so
no dataset code changes were required for this task.

Train/test split is global random, 90/10, seed 42 (336 train / 37 test out of
373 frames across `aria01..aria04`). The builder also supports
`--test_scenes aria04` for a scene-based holdout.

## Active Training

Started via tmux session `difix_tagging`:

```bash
tmux attach -t difix_tagging
```

- Launcher: `scripts/tagging/train.sh`
- GPUs: 0,1,2,3 (`NUM_GPUS=4`, accelerate default DDP)
- Output dir: `outputs/difix/train_tagging/`
- Log: `outputs/difix/train_tagging.log`
- Checkpoints every 2000 steps, eval every 200, viz every 20, total 10000 steps (20 epochs)
- wandb project `difix`, run name `train_tagging` (user `sinoyou`)

Before re-launching from a fresh shell, prepend the venv's PATH (see Repository).

## New / Changed Files This Turn

Untracked:

- `scripts/tagging/build_dataset.py` — generates the train/test JSON. Args:
  `--src_root` (default `egohuman_processed/004_tagging_extraction`), `--out`,
  `--train_frac` (0.9), `--seed` (42), `--prompt` (`remove degradation`),
  `--test_scenes` (e.g. `aria04` to hold out a whole scene; disables random split).
- `data/004_tagging/aria_tagging_finetune.json` — generated JSON.
- `scripts/tagging/train.sh` — launcher (CUDA_VISIBLE_DEVICES=0,1,2,3, lr 1e-5,
  lambda_lpips/l2/gram=1.0, gram warmup 500, resolution 512, 20 epochs, timestep 199).

No tracked source files were modified for this task.

## Resolution Handling (FYI)

`PairedDataset` resizes input/target/mask independently to `(--resolution,
--resolution)`, bilinear for RGB and **nearest** for the mask. The tagging
data is all square so independent resize keeps everything pixel-aligned even
though the source resolutions differ (1408 GT / 512 blend / 1408 mask). If
non-square or differently-cropped data is added later, this breaks and needs
either a `--height/--width` split or anchor-image-based resize logic.

## Carry-Over Context (from prior sessions, still valid)

- HF pretrained loading: `src/model.py`, `src/train_difix.py`,
  `src/inference_difix.py` already support initializing from a HF
  `DifixPipeline` repo such as `nvidia/difix` (custom VAE skip connections
  and `vae_skip` LoRA adapter metadata preserved).
- Mask-aware training (`invalid_pixel_mask` excluded from L2/LPIPS/Gram) was
  added in a previous turn and is what this task reuses.
- The repo now has both `data/volleyball_static/` and `data/004_tagging/` JSON
  configurations; `scripts/volleyball/train*.sh` are the reference
  launchers, and `scripts/tagging/train.sh` follows the same pattern.
- A previous dynamic3dgs benchmarking task exists in `outputs/difix/train_dynamic3dgs`
  and `scripts/common/benchmark.py`; not active in this session but
  files are present.

## Open Tasks

See `.claude/OPEN_TASKS.md`.

## Useful Commands

See `.claude/COMMANDS.md`.
