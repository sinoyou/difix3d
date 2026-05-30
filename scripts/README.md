# scripts/ layout

Subfolders group scripts by **experiment family**, with `common/` for
cross-cutting utilities. All paths inside the scripts are relative to the
repo root, so always invoke from `/local/home/zinyou/projects/Difix3D`.

| Folder           | Contents                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `common/`        | `benchmark.py` (standard PSNR/SSIM/LPIPS w/ masked metrics, used by every experiment), `make_comparison_video.py`, `inference_example.sh` |
| `tagging/`       | `build_dataset.py`, `infer_animate.py`, `train.sh` — egohuman_processed/004 tagging-extraction finetune (writes JSON to `data/004_tagging/`) |
| `volleyball/`    | `train.sh`, `train_lora.sh`, `train_mv.sh`, `train_temporal.sh` plus `add_ref_views.py`, `benchmark_mv_autoregressive.py`, and three `run_*_experiments.sh` orchestrators |
| `egohuman/`      | `prepare_finetune.py`, `prepare_masks.py`, `train.sh` — egohuman finetune                  |
| `dynamic3dgs/`   | `train.sh`, `train_only12.sh` — dynamic3dgs subset finetunes (reuse `common/benchmark.py`) |

## Conventions

- **Training launchers** (`scripts/<exp>/train*.sh`) wrap `accelerate launch
  src/train_difix.py` (or `src/train_difix_temporal.py`) with experiment-
  specific dataset paths, output dirs, and lambda weights. They set
  `CUDA_VISIBLE_DEVICES` at the top and do **not** activate the venv —
  caller must prepend `venv/bin/` to `PATH` (see CLAUDE.md → Environment).
- **Orchestrators** (`scripts/volleyball/run_*_experiments.sh`) `cd` to the
  repo root, then chain `bash scripts/<exp>/train*.sh` and
  `python scripts/.../benchmark*.py` invocations, teeing each stage to a
  timestamped log under `logs/`.
- **Data-prep scripts** are Python (`build_dataset.py`, `prepare_*.py`,
  `add_ref_views.py`); always run via `venv/bin/python`.
- **Benchmark scripts** live in `common/` if they are dataset-agnostic and
  in the experiment folder if they encode experiment-specific assumptions
  (e.g. `volleyball/benchmark_mv_autoregressive.py` feeds back the previous
  prediction as the next reference — only meaningful for the mv pipeline).

## Adding a new experiment

1. Create `scripts/<new_experiment>/`.
2. Add a JSON-builder (`build_dataset.py` or `prepare_*.py`) that emits a
   train/test JSON in the standard shape (see CLAUDE.md → Datasets).
3. Add `train.sh` cloned from the closest existing recipe — adjust
   `--output_dir`, `--dataset_path`, `--tracker_run_name`, and lambdas.
4. If a multi-stage workflow is needed, add a `run_*.sh` orchestrator
   that `cd`s to the repo root and chains training + `common/benchmark.py`.
