# Open Tasks

1. Run the dynamic3dgs benchmark command in `codex/COMMANDS.md`.

2. Confirm outputs exist:

   - `data/ego_aria01/pretrained_dynamic3dgs`
   - `data/ego_aria02/pretrained_dynamic3dgs`
   - `data/ego_aria01/finetuned_dynamic3dgs`
   - `data/ego_aria02/finetuned_dynamic3dgs`

3. Confirm comparison videos exist:

   - `data/ego_aria01/comparison_clean_dynamic3dgs_pretrained_dynamic3dgs_finetuned_dynamic3dgs.mp4`
   - `data/ego_aria02/comparison_clean_dynamic3dgs_pretrained_dynamic3dgs_finetuned_dynamic3dgs.mp4`

4. Confirm metrics JSON exists and parses:

   - `data/egohuman_benchamark_matrics_dynamic3dgs.json`

5. If benchmark fails due to CUDA/model cache, inspect whether `nvidia/difix` is cached or rerun with network permission if needed.

