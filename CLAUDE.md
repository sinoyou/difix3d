# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session bootstrap

This project keeps its own cross-session handoff in [.claude/](.claude/) —
read all of these at the start of every session:

- [.claude/HANDOFF.md](.claude/HANDOFF.md) — current goal, active runs, new files
- [.claude/COMMANDS.md](.claude/COMMANDS.md) — copy-pasteable commands for the active workstream
- [.claude/OPEN_TASKS.md](.claude/OPEN_TASKS.md) — checklist of next steps
- [.claude/memory/](.claude/memory/) — mirror of auto-memory (see below)
- [.claude/notes/](.claude/notes/) — milestone summaries (see below)

These reflect *this user's* workstream (which has diverged significantly from
upstream README usage); they take precedence over README when the two disagree.

## Memory and notes — in-repo mirror

The auto-memory at `${HOME}/.claude/projects/<slug>/memory/` is fragile:
if `~/.claude` is renamed, moved, or wiped, every cross-session memory
for this project is lost. To survive that, this repo mirrors auto-memory
inside the project itself.

**[.claude/memory/](.claude/memory/) — mirror of Claude's auto-memory.**
Every time Claude writes, updates, or deletes a file under
`${HOME}/.claude/projects/-local-home-zinyou-projects-Difix3D/memory/`,
it MUST make the identical change under `.claude/memory/` in the same
turn. The two trees stay byte-identical (same `MEMORY.md` index, same
per-memory files, same frontmatter). At session start, if the two
disagree, treat the in-repo mirror as the source of truth and rewrite
the auto-memory side to match — the home dir may have been reset.

**[.claude/notes/](.claude/notes/) — milestone summaries.** A note is a
narrative record of a significant transition (a finetune that finally
converged, an approach abandoned, a benchmark result worth remembering).
*Notes are user-triggered, not Claude-triggered.* Create a new note only
when the user explicitly asks ("save a note about this") or confirms a
suggestion. If a note for the current task already exists, update it
in-place without re-asking. See [.claude/notes/README.md](.claude/notes/README.md)
for the naming convention and the memory-vs-note distinction.

## Environment

Use the project venv `venv/` — it is a **conda-style env, not a stdlib venv**
(no `bin/activate`). Always activate by prepending PATH or calling binaries
directly:

```bash
PATH=/local/home/zinyou/projects/Difix3D/venv/bin:$PATH bash <launcher>.sh
# or
venv/bin/python <script>.py
venv/bin/accelerate launch ...
```

The system `base` conda env (`~/miniconda3`) is missing `lpips`/`diffusers`
and has a CUDA mismatch — launchers will fail under it.

## Common commands

Training launchers live under `scripts/<experiment>/train*.sh` and are tuned
for the user's local 4×GPU box. All paths inside the launchers are relative
to the repo root, so always invoke them from the repo root. They do **not**
activate the venv themselves; the caller must.

```bash
# Tagging finetune (current active workstream)
bash scripts/tagging/train.sh

# Volleyball variants
bash scripts/volleyball/train.sh             # single-view
bash scripts/volleyball/train_lora.sh        # UNet LoRA
bash scripts/volleyball/train_mv.sh          # multi-view UNet
bash scripts/volleyball/train_temporal.sh    # temporal warp loss

# Inference
bash scripts/common/inference_example.sh
```

Long jobs are launched under tmux (session name conventionally matches the
output dir, e.g. `difix_tagging`). See `.claude/COMMANDS.md` for the exact
tmux/PATH wrapper used.

Dataset JSONs are produced by scripts under `scripts/`, e.g.:

```bash
venv/bin/python scripts/tagging/build_dataset.py [--seed 42] [--test_scenes aria04]
venv/bin/python scripts/egohuman/prepare_finetune.py ...
venv/bin/python scripts/egohuman/prepare_masks.py ...
```

Benchmarks (PSNR/SSIM/LPIPS, optionally masked) live in
`scripts/common/benchmark.py` and `scripts/volleyball/benchmark_*.py`.

There is no test suite. Quick sanity checks:

```bash
venv/bin/python -m py_compile src/train_difix.py src/dataset.py
venv/bin/python -c "import lpips, torch, accelerate, diffusers; print(torch.cuda.device_count())"
```

## Architecture

### Model wrapper (`src/model.py`)

`Difix` wraps a SD-turbo-style pipeline customized for single-step
artifact removal. Three non-obvious pieces:

1. **VAE skip connections** (`add_vae_skip_connections`,
   `my_vae_encoder_fwd`, `my_vae_decoder_fwd`). The encoder caches per-level
   feature maps; the decoder consumes them through skip-conv adapters. This
   is what makes the VAE able to reconstruct fine detail from a single
   diffusion step — losing the skips silently degrades quality.
2. **VAE LoRA adapter `vae_skip`** (`vae_lora_target_modules`,
   `vae_skip_metadata`, `lora_rank_vae`). The skip-conv adapters are LoRA
   modules; their metadata is persisted in checkpoints and restored on load.
3. **Initialization sources** — `Difix.__init__` accepts a HF
   `DifixPipeline` repo (`pretrained_pipeline_name_or_path`, e.g.
   `nvidia/difix`) *or* a local `model_*.pkl`
   (`pretrained_path`/`pretrained_name`). Training defaults to the HF
   path; the volleyball/tagging recipes finetune from `nvidia/difix`.

`--mv_unet` swaps the UNet for the multi-view variant in `src/mv_unet.py`,
which adds cross-view attention so a clip of frames is denoised jointly.
`--use_unet_lora` keeps the UNet frozen and trains only a LoRA adapter.

### Pipeline (`src/pipeline_difix.py`)

`DifixPipeline` is a `diffusers`-style pipeline used both for training-time
sampling and for `pipe.from_pretrained("nvidia/difix", trust_remote_code=True)`
inference. The reference-image variant (`nvidia/difix_ref`) adds an optional
`ref_image` arg that concatenates a reference view as additional context —
the same convention is reflected in `PairedDataset` (`ref_image` field).

### Datasets (`src/dataset.py`) — invalid-mask convention

**Critical convention**: the JSON `"mask"` field is loaded as
`invalid_pixel_mask`. Pixels > 0.5 are **excluded** from supervision. Loss
code does `valid_mask = 1.0 - invalid_mask` and multiplies it into L2,
LPIPS, and (post-warmup) Gram. White in `_invalid.png` = "do not
supervise". Flipping the polarity silently trains on the wrong region.

Two dataset classes share this convention:

- `PairedDataset` — one (input, target, optional ref, optional mask) per
  item; resizes each image independently to `(--resolution, --resolution)`
  (bilinear for RGB, **nearest** for mask). Source resolutions can differ
  per stream as long as aspect ratios match and FOV is identical
  (e.g. tagging data has 1408² GT / 512² blend / 1408² mask).
- `ConsecutiveClipDataset` — yields `clip_length` consecutive frames per
  scene as `(V, C, H, W)`; used by the temporal/multi-view trainers.
  Scenes are inferred from the JSON key suffix `..._NNNNNN`.

Standard dataset JSON shape (all variants are subsets of this):

```json
{
  "train": {
    "<scene>_<idx>": {
      "image":        "<corrupted/input>.png",
      "target_image": "<clean/GT>.png",
      "mask":         "<invalid_mask>.png",  // optional, "1 = invalid"
      "ref_image":    "<reference>.png",     // optional, ref pipeline only
      "prompt":       "remove degradation"
    }
  },
  "test": { ... }
}
```

### Training entry points

- `src/train_difix.py` — image trainer used by `scripts/*/train*.sh`
  (excluding `_temporal`). Loss = `lambda_l2 * masked_MSE + lambda_lpips *
  masked_LPIPS + lambda_gram * Gram` (Gram added after
  `--gram_loss_warmup_steps`). The Gram loss (`src/loss.py`) is a
  VGG-feature style loss applied to the predicted/target images after
  masking.
- `src/train_difix_temporal.py` — clip trainer used by
  `scripts/volleyball/train_temporal.sh`. Adds an optical-flow-based
  temporal-warp loss (`src/temporal_loss.py`, uses RAFT) on top of the
  per-frame losses, with occlusion masking.

Both honor `--checkpointing_steps`, `--eval_freq`, `--viz_freq` and write
to `outputs/difix/<run>/` with wandb logging.

### Inference (`src/inference_difix.py`)

Loads a Difix3D `model_*.pkl` (or HF repo), runs the pipeline at the
chosen `--timestep` (default 199 — single-step), and writes images.
`scripts/common/benchmark.py` and `scripts/volleyball/benchmark_*.py` are wrappers that run inference over a dataset
JSON, generate comparison videos, and compute masked metrics.

### Data layout

- `data/volleyball_static/` — egocentric volleyball clips (`ego_aria01..04`)
  with corrupted/clean/mask subdirs and per-config JSONs.
- `data/004_tagging/` — tagging dataset built from
  `egohuman_processed/004_tagging_extraction/{rgb,rgb_blend,mask}` by
  `scripts/tagging/build_dataset.py`. Mask = each frame's `_invalid.png`.
- `data/tagging_zjumocap/` — egohuman/zjumocap finetune + dynamic3dgs assets from prior workstreams (ego_aria01..04, egohuman_*.json, dynamic3dgs_ego.mp4 + frames).
- `outputs/difix/<run>/` — training output dirs, including
  `checkpoints/model_*.pkl` (custom serialization via `save_ckpt` /
  `load_ckpt_from_state_dict` in `src/model.py`).

### Downstream 3D pipelines (`examples/`)

The "Difix3D+" reconstruction pipelines (nerfstudio and gsplat
integrations under `examples/`) call into the trained model for
post-render artifact removal. These are upstream/canonical paths from the
paper and are largely independent of the local-finetune workstream.

## Conventions worth honoring

- New dataset JSONs reuse `PairedDataset` / `ConsecutiveClipDataset` —
  don't subclass unless data shape genuinely differs.
- Mask polarity is **always** "1 = invalid" across all JSONs in this
  repo. New mask producers must follow this.
- When adding a new finetune recipe, follow the `scripts/*/train*.sh`
  pattern (top-of-file `CUDA_VISIBLE_DEVICES`, `accelerate launch`, no
  venv activation) and add a matching JSON-builder under `scripts/`.
- Resolution is currently single-int square (`--resolution N`). Mixing
  square and non-square data needs a `--height/--width` split or anchor
  resize — currently absent.
