# Useful Commands

All commands assume CWD = `/local/home/zinyou/projects/Difix3D`. Always run
via the project venv (`PATH=/local/home/zinyou/projects/Difix3D/venv/bin:$PATH`
or `venv/bin/python ...`); base `python` lacks `lpips`/`diffusers`.

## Inspect Current Tagging Training

```bash
tmux ls | grep difix_tagging
tmux attach -t difix_tagging                  # detach: Ctrl-b d
tail -f outputs/difix/train_tagging.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls outputs/difix/train_tagging/checkpoints 2>/dev/null
```

Stop the run:

```bash
tmux kill-session -t difix_tagging
```

## Regenerate the Tagging Dataset JSON

Global 90/10 random split (default, seed 42):

```bash
venv/bin/python scripts/tagging/build_dataset.py
```

Hold out a whole scene as the test split:

```bash
venv/bin/python scripts/tagging/build_dataset.py --test_scenes aria04
```

Custom seed / output:

```bash
venv/bin/python scripts/tagging/build_dataset.py --seed 7 --out data/004_tagging/aria_tagging_seed7.json
```

## (Re)launch Tagging Training

```bash
tmux new-session -d -s difix_tagging -c "$PWD" \
  "PATH=/local/home/zinyou/projects/Difix3D/venv/bin:\$PATH \
   bash scripts/tagging/train.sh 2>&1 | tee outputs/difix/train_tagging.log; exec bash"
```

GPUs used (set inside the script): `CUDA_VISIBLE_DEVICES=0,1,2,3`.

## Sanity-Check a JSON Entry

```bash
venv/bin/python - <<'PY'
import json
d = json.load(open('data/004_tagging/aria_tagging_finetune.json'))
print('train', len(d['train']), 'test', len(d['test']))
k = next(iter(d['train'])); print(k, d['train'][k])
PY
```

## Quick Compile / Import Checks

```bash
venv/bin/python -m py_compile src/train_difix.py src/dataset.py scripts/tagging/build_dataset.py
venv/bin/python -c "import lpips, torch, accelerate, diffusers; print(torch.cuda.device_count())"
```

## Git State

```bash
git status --short
git log --oneline -10
```

## Previous-Task Commands (still available, not active)

Dynamic3DGS benchmark — see `scripts/common/benchmark.py` and the
pre-2026-05-30 revision of this file in git history.
