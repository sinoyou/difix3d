# Open Tasks

Active focus: tagging finetune (`outputs/difix/train_tagging`).

1. Monitor the running `difix_tagging` tmux session until the first
   eval (step 200) lands in `outputs/difix/train_tagging.log` and confirm
   the masked L2/LPIPS losses look reasonable (i.e. `_invalid.png` regions
   are actually being excluded — visualizations are dumped every 20 steps).

2. After step 2000 a checkpoint should appear in
   `outputs/difix/train_tagging/checkpoints/`. Confirm it exists and is
   loadable via the same HF/pretrained code path used in
   `src/inference_difix.py`.

3. Decide on the held-out evaluation strategy: the current JSON is a
   *random* 90/10 split, so train and test share scenes. If a stricter
   generalization metric is wanted, regenerate with
   `--test_scenes aria04` (scene-level holdout) and re-launch.

4. Optional: add a benchmark script analogous to
   `scripts/common/benchmark.py` but pointed at the tagging
   `test` split, so PSNR/SSIM/LPIPS can be computed with the invalid
   mask applied. Reuse the masked-metric helper that already blacks out
   invalid pixels in both prediction and GT.

5. Optional: extend `PairedDataset` to accept a `(--height, --width)`
   pair (or compute target size from the GT image's aspect ratio) before
   any non-square dataset is added — current resize-to-square logic
   silently breaks alignment when sources have different aspect ratios.

## Previous Open Tasks (carried over, not active)

- Dynamic3DGS benchmark items from prior session — see git history of
  this file for the dynamic3dgs checklist. Re-prioritize only if the
  user revisits that workstream.
