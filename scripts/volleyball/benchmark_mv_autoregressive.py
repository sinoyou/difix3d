"""Autoregressive benchmark for the mv_unet (multi-view) Difix3D model.

For each test frame, the reference image fed to the model is the *previously
predicted* enhanced frame (i.e. the model's own output from the previous step),
which encourages temporal consistency. For the very first test frame, the
reference is taken from the JSON split (typically the last training frame's
clean image, as set by scripts/volleyball/add_ref_views.py).

PSNR / SSIM / LPIPS are computed against the ground-truth clean frames, masked
by the invalid pixel mask.
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_resized(path, size):
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def pil_to_tensor(image):
    arr = np.asarray(image).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.cuda()


def mask_to_tensor(path, size):
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = (np.asarray(m).astype(np.float32) > 127.0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).cuda()


def run_autoregressive(model, entries, scene_dir, output_subdir, image_size, prompt, overwrite, root):
    """entries: ordered list of (key, entry_dict). Returns list of output paths in order."""
    out_dir = scene_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    prev_pred_pil = None  # most recent enhanced output

    for key, entry in tqdm(entries, desc=f"Autoregressive {out_dir.name}"):
        in_path = (root / entry["image"]).resolve()
        out_path = out_dir / Path(entry["image"]).name
        output_paths.append(out_path)

        # Reference: previous predicted frame (if any), else fall back to ref_image in JSON.
        if prev_pred_pil is not None:
            ref_pil = prev_pred_pil
        else:
            ref_path = entry.get("ref_image")
            if ref_path is None:
                raise ValueError(f"No ref_image and no previous prediction available for {key}")
            ref_pil = Image.open((root / ref_path).resolve()).convert("RGB")

        if out_path.is_file() and not overwrite:
            # Reuse the cached prediction so future steps still chain from disk.
            prev_pred_pil = Image.open(out_path).convert("RGB")
            continue

        image = Image.open(in_path).convert("RGB")
        # Match training resize.
        output = model.sample(
            image,
            height=image_size,
            width=image_size,
            ref_image=ref_pil,
            prompt=prompt,
        )
        output = output.resize((image_size, image_size), Image.LANCZOS)
        output.save(out_path)
        prev_pred_pil = output

    return output_paths


def evaluate(pred_paths, clean_paths, mask_paths, image_size, lpips_metric):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lpips_values = []
    for p, c, m in tqdm(list(zip(pred_paths, clean_paths, mask_paths)), desc="Metrics"):
        pred = pil_to_tensor(load_resized(p, image_size))
        clean = pil_to_tensor(load_resized(c, image_size))
        valid = 1.0 - mask_to_tensor(m, image_size)
        pred = pred * valid
        clean = clean * valid
        psnr_metric.update(pred, clean)
        ssim_metric.update(pred, clean)
        with torch.no_grad():
            lpips_values.append(lpips_metric(pred, clean).item())
    return {
        "psnr": psnr_metric.compute().item(),
        "ssim": ssim_metric.compute().item(),
        "lpips_vgg": float(np.mean(lpips_values)),
    }


def write_video(out_path, frame_lists, image_size, fps):
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for tup in tqdm(list(zip(*frame_lists)), desc=f"Video {out_path.name}"):
            frames = [np.asarray(load_resized(p, image_size)) for p in tup]
            writer.append_data(np.concatenate(frames, axis=1))
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data_volleyball"))
    parser.add_argument("--scene", required=True, help='e.g. "ego_aria01"')
    parser.add_argument("--split_json", type=Path, required=True,
                        help="JSON with train/test splits including ref_image (e.g. *_finetune_mv.json)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--finetuned_checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", default="nvidia/difix")
    parser.add_argument("--input_subdir", default="corrupted")
    parser.add_argument("--output_subdir", required=True,
                        help="Output folder under <data_root>/<scene>/<output_subdir>")
    parser.add_argument("--comparison_video_name", default="comparison_mv_autoregressive.mp4")
    parser.add_argument("--metrics_path", type=Path, required=True)
    parser.add_argument("--prompt", default="remove degradation")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    scene_dir = (args.data_root / args.scene).resolve() if args.data_root.is_absolute() \
        else (repo_root / args.data_root / args.scene)

    with open(args.split_json) as f:
        data = json.load(f)
    if args.split not in data:
        raise ValueError(f"Split {args.split!r} not in {args.split_json}")
    # Sort keys by trailing frame index to enforce temporal order.
    entries = sorted(data[args.split].items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1]))

    # Resolve clean/mask paths from entries
    clean_paths = [(repo_root / e["target_image"]).resolve() for _, e in entries]
    mask_paths = [(repo_root / e["mask"]).resolve() for _, e in entries]
    input_paths = [(repo_root / e["image"]).resolve() for _, e in entries]

    from model import Difix

    print(f"Loading finetuned mv_unet model from {args.finetuned_checkpoint}")
    model = Difix(
        pretrained_name=args.pretrained_model_name_or_path,
        pretrained_path=str(args.finetuned_checkpoint),
        timestep=args.timestep,
        mv_unet=True,
    )
    model.set_eval()

    output_paths = run_autoregressive(
        model, entries, scene_dir, args.output_subdir,
        args.image_size, args.prompt, args.overwrite, repo_root,
    )

    del model
    torch.cuda.empty_cache()

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).cuda().eval()
    metrics = {
        args.input_subdir: evaluate(input_paths, clean_paths, mask_paths, args.image_size, lpips_metric),
        args.output_subdir: evaluate(output_paths, clean_paths, mask_paths, args.image_size, lpips_metric),
    }

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w") as f:
        json.dump({args.scene: metrics}, f, indent=2)
        f.write("\n")
    print(f"Wrote metrics to {args.metrics_path}")

    video_path = scene_dir / args.comparison_video_name
    write_video(video_path, [clean_paths, input_paths, output_paths], args.image_size, args.fps)
    print(f"Wrote comparison video to {video_path}")


if __name__ == "__main__":
    main()
