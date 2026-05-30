---
name: project-venv
description: "Difix3D project venv path, type, and how to activate it (base python is missing required deps)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9840db9a-7944-4e31-8888-4963a4b4f638
---

The Difix3D project venv lives at `/local/home/zinyou/projects/Difix3D/venv`.
It is a **conda-style env**, NOT a stdlib venv — there is no
`bin/activate` script. Trying `source venv/bin/activate` fails with
"No such file or directory".

To use it in a non-interactive shell (tmux launchers, scripts), prepend
its `bin/` to `PATH`:

```bash
PATH=/local/home/zinyou/projects/Difix3D/venv/bin:$PATH bash <script>.sh
```

Or call binaries directly: `venv/bin/python ...`, `venv/bin/accelerate ...`.

**Why:** the system `base` conda env (`/home/zinyou/miniconda3`) is missing
`lpips` and ships a torch built for a newer CUDA than the host driver
supports. Launching `accelerate launch src/train_difix.py` under `base`
fails immediately with `ModuleNotFoundError: No module named 'lpips'`.
The project venv has `torch 2.11.0+cu128`, `accelerate`, `lpips`,
`diffusers`, etc., and CUDA works (8 GPUs visible).

**How to apply:** every time you launch training/inference for Difix3D —
especially when spawning a tmux session from a fresh shell — make sure
the venv's `bin/` is on PATH before running. Reference launchers like
`quick_local_train_tagging.sh` and `quick_local_train_volleyball*.sh` do
NOT activate the venv themselves; the caller must.
