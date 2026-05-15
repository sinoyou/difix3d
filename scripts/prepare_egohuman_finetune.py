import argparse
import json
import subprocess
from pathlib import Path


SCENES = ("ego_aria01", "ego_aria02", "ego_aria03", "ego_aria04")
TRAIN_SCENES = {"ego_aria01", "ego_aria02", "ego_aria03"}
FRAME_PATTERN = "[0-9][0-9][0-9][0-9][0-9][0-9].png"
PROMPT = "remove degradation"


def extract_video(video_path, output_dir, overwrite=False):
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob(FRAME_PATTERN))
    if existing and not overwrite:
        return existing

    if overwrite:
        for path in output_dir.glob("*.png"):
            path.unlink()

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            str(output_dir / "%06d.png"),
        ],
        check=True,
    )

    frames = sorted(output_dir.glob(FRAME_PATTERN))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


def build_split_entries(scene, corrupted_frames, clean_frames):
    if len(corrupted_frames) != len(clean_frames):
        raise RuntimeError(
            f"Frame count mismatch for {scene}: "
            f"{len(corrupted_frames)} corrupted vs {len(clean_frames)} clean"
        )

    entries = {}
    aria_name = scene.replace("ego_", "")
    for corrupted_frame, clean_frame in zip(corrupted_frames, clean_frames):
        if corrupted_frame.name != clean_frame.name:
            raise RuntimeError(
                f"Frame name mismatch for {scene}: "
                f"{corrupted_frame.name} vs {clean_frame.name}"
            )
        data_id = f"{aria_name}_{corrupted_frame.stem}"
        entries[data_id] = {
            "image": str(corrupted_frame),
            "target_image": str(clean_frame),
            "prompt": PROMPT,
        }
    return entries


def prepare_dataset(source_root, output_root, json_path, overwrite=False):
    dataset = {"train": {}, "test": {}}

    for scene in SCENES:
        scene_dir = source_root / scene
        corrupted_video = scene_dir / "corrupted.mp4"
        clean_video = scene_dir / "clean.mp4"

        if not corrupted_video.is_file():
            raise FileNotFoundError(f"Missing corrupted video: {corrupted_video}")
        if not clean_video.is_file():
            raise FileNotFoundError(f"Missing clean video: {clean_video}")

        scene_output = output_root / scene
        corrupted_frames = extract_video(corrupted_video, scene_output / "corrupted", overwrite=overwrite)
        clean_frames = extract_video(clean_video, scene_output / "clean", overwrite=overwrite)

        split = "train" if scene in TRAIN_SCENES else "test"
        dataset[split].update(build_split_entries(scene, corrupted_frames, clean_frames))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(dataset, f, indent=2)
        f.write("\n")

    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, default=Path("finetuning_data"))
    parser.add_argument("--output_root", type=Path, default=Path("data"))
    parser.add_argument("--json_path", type=Path, default=Path("data/egohuman_finetune.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        json_path=args.json_path,
        overwrite=args.overwrite,
    )
    print(f"Wrote {args.json_path}")
    print(f"Train pairs: {len(dataset['train'])}")
    print(f"Test pairs: {len(dataset['test'])}")


if __name__ == "__main__":
    main()
