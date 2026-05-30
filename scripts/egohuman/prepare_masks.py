import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_SOURCE_ROOT = Path("/home/zinyou/local/projects/LHM/data/004_tagging_gt")
DATA_ROOT = Path("data")
EGO_VIEW_PREFIX = "ego_aria"


def read_mask(path):
    return np.array(Image.open(path).convert("L")) > 127


def resize_mask(mask, size):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return image.resize(size, Image.NEAREST)


def erode_mask(mask_image, kernel_size):
    if kernel_size <= 1:
        return mask_image
    if kernel_size % 2 == 0:
        raise ValueError(f"Erosion kernel size must be odd, got {kernel_size}")
    return mask_image.filter(ImageFilter.MinFilter(kernel_size))


def source_frame_name(frame_stem):
    return f"{int(frame_stem):05d}"


def make_wearer_mask(source_root, view, frame_stem):
    source_frame = source_frame_name(frame_stem)
    timestep_dir = source_root / source_frame

    person_path = timestep_dir / "person" / f"{view}_{source_frame}.jpg"
    if not person_path.is_file():
        raise FileNotFoundError(f"Missing all-person mask: {person_path}")

    person_mask = read_mask(person_path)
    other_people = np.zeros_like(person_mask, dtype=bool)

    identity_dir = timestep_dir / "identity_mask_new_refine" / view
    if not identity_dir.is_dir():
        raise FileNotFoundError(f"Missing identity mask directory: {identity_dir}")

    identity_paths = sorted(identity_dir.glob("*.png"))
    if not identity_paths:
        raise FileNotFoundError(f"No identity masks found in: {identity_dir}")

    for identity_path in identity_paths:
        other_people |= read_mask(identity_path)

    return person_mask & ~other_people


def update_entry(entry, source_root, overwrite=False, erode_kernel=3):
    image_path = Path(entry["image"])
    target_path = Path(entry["target_image"])
    if len(image_path.parts) < 3 or image_path.parts[0] != DATA_ROOT.name:
        raise ValueError(f"Expected image path under data/<view>/: {image_path}")

    view = image_path.parts[1]
    if not view.startswith(EGO_VIEW_PREFIX):
        raise ValueError(f"Expected ego_aria view in image path: {image_path}")

    frame_stem = image_path.stem

    mask_path = DATA_ROOT / view / "mask" / f"{frame_stem}.png"
    if overwrite or not mask_path.is_file():
        wearer_mask = make_wearer_mask(source_root, view, frame_stem)
        target_size = Image.open(target_path).size
        mask_image = resize_mask(wearer_mask, target_size)
        mask_image = erode_mask(mask_image, erode_kernel)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_image.save(mask_path)

    entry["mask"] = str(mask_path)


def prepare_masks(source_root, json_path, overwrite=False, erode_kernel=3):
    with json_path.open("r") as f:
        dataset = json.load(f)

    for split in ("train", "test"):
        if split not in dataset:
            raise KeyError(f"Missing split in {json_path}: {split}")
        for entry in dataset[split].values():
            update_entry(entry, source_root=source_root, overwrite=overwrite, erode_kernel=erode_kernel)

    with json_path.open("w") as f:
        json.dump(dataset, f, indent=2)
        f.write("\n")

    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--json_path", type=Path, default=Path("data/tagging_zjumocap/egohuman_finetune.json"))
    parser.add_argument("--erode_kernel", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = prepare_masks(
        args.source_root,
        args.json_path,
        overwrite=args.overwrite,
        erode_kernel=args.erode_kernel,
    )
    print(f"Updated {args.json_path}")
    print(f"Train masks: {len(dataset['train'])}")
    print(f"Test masks: {len(dataset['test'])}")


if __name__ == "__main__":
    main()
