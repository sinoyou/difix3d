---
name: mask-convention
description: "In Difix3D JSON datasets, the `mask` field is the *invalid* mask; loss code derives valid_mask = 1 - invalid_mask"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9840db9a-7944-4e31-8888-4963a4b4f638
---

Across all Difix3D dataset JSONs (`data_volleyball/*.json`,
`data_tagging/aria_tagging_finetune.json`, `data/egohuman_*.json`):

```
"mask": "<path>"   →  loaded as `invalid_pixel_mask`
                     pixels > 0.5 are EXCLUDED from supervision
```

`src/dataset.py:PairedDataset` thresholds the mask at 0.5 with NEAREST
resize. `src/train_difix.py` computes `valid_mask = 1.0 - invalid_mask`
and multiplies it into the L2 loss, the LPIPS input, and (after warmup)
the Gram loss.

**Why:** this is the existing project convention — `1 = invalid` — and
it matches how `_invalid.png` files are authored (white = exclude).
If you ever flip the polarity (e.g. "valid" mask) you must invert
before writing to the JSON, otherwise the model is supervised only on
the regions you meant to exclude.

**How to apply:** when designing a new dataset for Difix3D, reuse
`PairedDataset` and write `mask` as the invalid-pixel mask. No new
Dataset subclass is needed unless aspect-ratio or multi-view structure
changes. When debugging "loss looks weird", double-check that the
mask polarity matches this convention.
