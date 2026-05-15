#!/usr/bin/env python
"""Horizontally concat 4 image sequences per aria_id with bottom labels and emit a 15fps video."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ARIA_IDS = ["aria01", "aria02", "aria03", "aria04"]
FPS = 15
LABEL_H = 48
LABELS = ["Ground Truth", "Baseline", "Ours (LHM)", "Ours (Difix3D)"]


def folders_for(aria_id):
    return [
        ROOT / "data" / f"ego_{aria_id}" / "clean",
        ROOT / "data" / f"ego_{aria_id}" / "corrupted",
        ROOT / "finetuning_data" / f"ego_{aria_id}" / "corrupted_lhm_ego",
        ROOT / "data" / f"ego_{aria_id}" / "finetuned",
    ]


def load_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = load_font(24)


def compose_frame(paths):
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = min(im.height for im in imgs)
    imgs = [im if im.height == h else im.resize((int(im.width * h / im.height), h)) for im in imgs]
    total_w = sum(im.width for im in imgs)
    canvas = Image.new("RGB", (total_w, h + LABEL_H), (0, 0, 0))
    x = 0
    draw = ImageDraw.Draw(canvas)
    for im, label in zip(imgs, LABELS):
        canvas.paste(im, (x, 0))
        bbox = draw.textbbox((0, 0), label, font=FONT)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x + (im.width - tw) // 2
        ty = h + (LABEL_H - th) // 2 - bbox[1]
        draw.text((tx, ty), label, fill=(255, 255, 255), font=FONT)
        x += im.width
    return canvas


def process(aria_id, out_path):
    folders = folders_for(aria_id)
    for f in folders:
        if not f.is_dir():
            print(f"missing: {f}", file=sys.stderr)
            return
    lists = [sorted(p for p in f.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) for f in folders]
    n = min(len(l) for l in lists)
    if n == 0:
        print(f"no images for {aria_id}", file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(n):
            frame = compose_frame([lists[k][i] for k in range(4)])
            frame.save(td / f"{i:06d}.png")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(td / "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    print(f"wrote {out_path}")


def main():
    out_dir = ROOT / "outputs" / "comparison_4way"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] or ARIA_IDS
    for aria_id in targets:
        process(aria_id, out_dir / f"comparison_{aria_id}.mp4")


if __name__ == "__main__":
    main()
