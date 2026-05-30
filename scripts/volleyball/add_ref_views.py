"""Add reference views to volleyball finetune JSONs for mv_unet training.

For each frame, the reference image is the GT (clean) image of the previous frame.
For the very first frame in each scene, the reference is the last frame's GT,
so every entry has a valid reference.
"""
import argparse
import json
import os
from pathlib import Path


def add_refs(json_path: Path, out_path: Path):
    with open(json_path) as f:
        data = json.load(f)

    # Collect all known frame keys per scene across splits to find the global "last frame"
    # per scene prefix (e.g. "aria01"). We sort by the trailing frame index.
    all_keys_by_prefix = {}
    for split, items in data.items():
        for key in items.keys():
            prefix = key.rsplit("_", 1)[0]
            all_keys_by_prefix.setdefault(prefix, []).append(key)
    for prefix in all_keys_by_prefix:
        all_keys_by_prefix[prefix].sort(key=lambda k: int(k.rsplit("_", 1)[1]))

    # Also build a flat lookup: key -> clean (target) image path
    target_lookup = {}
    for split, items in data.items():
        for key, entry in items.items():
            target_lookup[key] = entry["target_image"]

    for split, items in data.items():
        for key, entry in items.items():
            prefix = key.rsplit("_", 1)[0]
            idx = int(key.rsplit("_", 1)[1])
            sorted_keys = all_keys_by_prefix[prefix]
            # Find position of current key
            pos = sorted_keys.index(key)
            if pos == 0:
                ref_key = sorted_keys[-1]  # wrap around: use last frame
            else:
                ref_key = sorted_keys[pos - 1]
            entry["ref_image"] = target_lookup[ref_key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--suffix", default="_mv", help="Suffix added before .json")
    args = parser.parse_args()

    for json_path in args.inputs:
        out_path = json_path.with_name(json_path.stem + args.suffix + json_path.suffix)
        add_refs(json_path, out_path)


if __name__ == "__main__":
    main()
