"""Run the fine-tuned Difix model on every sequence under tagging_animate/.

Layout of the input root (3 subdirs × 4 scenes):
    <root>/{original,stable,stable_view}/{aria01,aria02,aria03,aria04}/*.png

For each sequence we write:
    <root>/<subdir>/<scene>_fix/<frame>.png   # restored frames
    <root>/<subdir>/<scene>_fix.mp4           # 30 fps mp4

The model is loaded once and reused across all sequences. Single GPU.
"""

import argparse
import os
import sys
from glob import glob

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
from model import Difix  # noqa: E402


SUBDIRS = ("original", "stable", "stable_view")
SCENES = ("aria01", "aria02", "aria03", "aria04")


def list_sequence_frames(seq_dir):
    return sorted(glob(os.path.join(seq_dir, "*.png")))


def run_one_sequence(model, seq_dir, out_img_dir, out_mp4, prompt, height, width, fps):
    frames = list_sequence_frames(seq_dir)
    if not frames:
        print(f"  [skip] no PNGs in {seq_dir}")
        return
    os.makedirs(out_img_dir, exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=fps)
    try:
        for path in tqdm(frames, desc=os.path.basename(seq_dir), leave=False):
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                out = model.sample(img, height=height, width=width, prompt=prompt)
            out_path = os.path.join(out_img_dir, os.path.basename(path))
            out.save(out_path)
            writer.append_data(np.array(out))
    finally:
        writer.close()
    print(f"  -> {len(frames)} frames | {out_img_dir} | {out_mp4}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="./egohuman_processed/004_tagging_animate",
        help="Root containing original/, stable/, stable_view/ subdirs.",
    )
    p.add_argument(
        "--ckpt",
        default="outputs/difix/train_tagging/checkpoints/model_final_1420.pkl",
        help="Fine-tuned checkpoint (.pkl) from train_difix.py.",
    )
    p.add_argument("--pretrained_name", default="nvidia/difix")
    p.add_argument("--prompt", default="remove degradation")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--timestep", type=int, default=199)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--subdirs",
        default=",".join(SUBDIRS),
        help="Comma-separated subdirs under root to process.",
    )
    p.add_argument(
        "--scenes",
        default=",".join(SCENES),
        help="Comma-separated scene names to process within each subdir.",
    )
    p.add_argument("--suffix", default="_fix")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    print(f"Loading model: pretrained={args.pretrained_name}  ckpt={args.ckpt}")
    model = Difix(
        pretrained_name=args.pretrained_name,
        pretrained_path=args.ckpt,
        timestep=args.timestep,
        mv_unet=False,
    )
    model.set_eval()

    plan = []
    for sub in subdirs:
        for scene in scenes:
            seq_dir = os.path.join(args.root, sub, scene)
            if not os.path.isdir(seq_dir):
                print(f"  [skip] missing {seq_dir}")
                continue
            out_img_dir = os.path.join(args.root, sub, f"{scene}{args.suffix}")
            out_mp4 = os.path.join(args.root, sub, f"{scene}{args.suffix}.mp4")
            if not args.overwrite and os.path.isfile(out_mp4) and os.path.isdir(out_img_dir):
                done = len(glob(os.path.join(out_img_dir, "*.png")))
                expected = len(list_sequence_frames(seq_dir))
                if done == expected and expected > 0:
                    print(f"  [skip done] {out_mp4} ({done} frames)")
                    continue
            plan.append((seq_dir, out_img_dir, out_mp4))

    print(f"Will process {len(plan)} sequences.")
    for i, (seq_dir, out_img_dir, out_mp4) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {seq_dir}")
        run_one_sequence(
            model, seq_dir, out_img_dir, out_mp4,
            prompt=args.prompt, height=args.height, width=args.width, fps=args.fps,
        )

    print("Done.")


if __name__ == "__main__":
    main()
