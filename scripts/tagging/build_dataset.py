"""Build a Difix3D train/test JSON for egohuman_processed/004_tagging_extraction.

Convention (matches src/dataset.py:PairedDataset and the masked losses in
src/train_difix.py): the value at the `mask` key is treated as the
*invalid* pixel mask, i.e. pixels > 0.5 are excluded from L2 / LPIPS / Gram.
We point `mask` at each frame's `_invalid.png`.

  image        := rgb_blend/{scene}_{idx}.png   (corrupted input)
  target_image := rgb/{scene}_{idx}.png         (clean GT)
  mask         := mask/{scene}_{idx}/_invalid.png

The 90/10 split is taken globally over all frames with a fixed seed.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path


FRAME_RE = re.compile(r"^(?P<scene>[a-zA-Z0-9]+)_(?P<idx>\d+)\.png$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src_root",
        default="egohuman_processed/004_tagging_extraction",
        help="Root containing rgb/, rgb_blend/, mask/ subdirectories.",
    )
    p.add_argument(
        "--out",
        default="data/004_tagging/aria_tagging_finetune.json",
        help="Output JSON path (relative to repo root).",
    )
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="remove degradation")
    p.add_argument(
        "--test_scenes",
        default=None,
        help=(
            "Comma-separated scene prefixes used as the test split (e.g. 'aria04'). "
            "When set, the random fractional split is disabled and all other scenes go to train."
        ),
    )
    args = p.parse_args()
    test_scenes = (
        {s.strip() for s in args.test_scenes.split(",") if s.strip()}
        if args.test_scenes
        else None
    )

    src_root = Path(args.src_root)
    rgb_dir = src_root / "rgb"
    blend_dir = src_root / "rgb_blend"
    mask_dir = src_root / "mask"

    if not rgb_dir.is_dir():
        raise SystemExit(f"missing {rgb_dir}")

    entries = []  # (key, dict)
    skipped = []
    for fname in sorted(os.listdir(rgb_dir)):
        m = FRAME_RE.match(fname)
        if not m:
            continue
        scene, idx = m.group("scene"), m.group("idx")
        key = f"{scene}_{idx}"
        rgb_path = rgb_dir / fname
        blend_path = blend_dir / fname
        invalid_path = mask_dir / key / "_invalid.png"
        if not blend_path.is_file() or not invalid_path.is_file():
            skipped.append((key, blend_path.is_file(), invalid_path.is_file()))
            continue
        entries.append(
            (
                key,
                {
                    "image": str(blend_path),
                    "target_image": str(rgb_path),
                    "prompt": args.prompt,
                    "mask": str(invalid_path),
                },
            )
        )

    if not entries:
        raise SystemExit("no frames found")

    if test_scenes is not None:
        # Scene-based split: every frame whose scene prefix is in test_scenes is
        # held out; everything else is training.
        train_entries = [(k, v) for (k, v) in entries if k.rsplit("_", 1)[0] not in test_scenes]
        test_entries = [(k, v) for (k, v) in entries if k.rsplit("_", 1)[0] in test_scenes]
        seen = {k.rsplit("_", 1)[0] for k, _ in entries}
        missing = test_scenes - seen
        if missing:
            raise SystemExit(f"--test_scenes references scenes not present in data: {missing}")
        if not test_entries:
            raise SystemExit(f"--test_scenes {test_scenes} matched zero frames")
    else:
        rng = random.Random(args.seed)
        shuffled = entries[:]
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * args.train_frac))
        train_entries = shuffled[:n_train]
        test_entries = shuffled[n_train:]

    # Sort each split by key for stable JSON ordering.
    train_entries.sort(key=lambda kv: kv[0])
    test_entries.sort(key=lambda kv: kv[0])

    out = {
        "train": {k: v for k, v in train_entries},
        "test": {k: v for k, v in test_entries},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    per_scene_train = {}
    per_scene_test = {}
    for k, _ in train_entries:
        per_scene_train[k.rsplit("_", 1)[0]] = per_scene_train.get(k.rsplit("_", 1)[0], 0) + 1
    for k, _ in test_entries:
        per_scene_test[k.rsplit("_", 1)[0]] = per_scene_test.get(k.rsplit("_", 1)[0], 0) + 1

    print(f"wrote {out_path}")
    print(f"  total={len(entries)}  train={len(train_entries)}  test={len(test_entries)}  seed={args.seed}")
    print(f"  per-scene train: {per_scene_train}")
    print(f"  per-scene test:  {per_scene_test}")
    if skipped:
        print(f"  skipped {len(skipped)} frames (missing blend or invalid mask); first few: {skipped[:5]}")


if __name__ == "__main__":
    main()
