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

DEFAULT_SCENES = ("ego_aria01", "ego_aria02", "ego_aria03", "ego_aria04")
PROMPT = "remove degradation"


def image_paths(folder, names=None):
    paths = sorted(Path(folder).glob("*.png"))
    if names is None:
        return paths
    name_set = set(names)
    return [p for p in paths if p.name in name_set]


def load_split_names(split_json, scene, split):
    if split_json is None:
        return None
    with open(split_json) as f:
        data = json.load(f)
    if split not in data:
        raise ValueError(f"split '{split}' not found in {split_json}")
    prefix = scene.replace("ego_", "") + "_"
    names = []
    for key in data[split].keys():
        if key.startswith(prefix):
            names.append(key[len(prefix):] + ".png")
        else:
            names.append(key + ".png")
    return names


def load_resized_image(path, image_size):
    return Image.open(path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)


def pil_to_metric_tensor(image):
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.cuda()


def mask_to_tensor(mask_path, image_size):
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    array = (np.asarray(mask).astype(np.float32) > 127.0).astype(np.float32)
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor.cuda()


def run_model(model, input_paths, output_dir, image_size, prompt, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in tqdm(input_paths, desc=f"Inference {output_dir}"):
        output_path = output_dir / input_path.name
        if output_path.is_file() and not overwrite:
            continue

        image = load_resized_image(input_path, image_size)
        output = model.sample(
            image,
            height=image_size,
            width=image_size,
            prompt=prompt,
        )
        output.resize((image_size, image_size), Image.LANCZOS).save(output_path)


def evaluate_folder(pred_paths, clean_paths, mask_paths, image_size, lpips_metric):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lpips_values = []
    for pred_path, clean_path, mask_path in tqdm(
        list(zip(pred_paths, clean_paths, mask_paths)),
        desc=f"Metrics {pred_paths[0].parent}",
    ):
        pred = pil_to_metric_tensor(load_resized_image(pred_path, image_size))
        clean = pil_to_metric_tensor(load_resized_image(clean_path, image_size))
        invalid_mask = mask_to_tensor(mask_path, image_size)
        valid_mask = 1.0 - invalid_mask
        pred = pred * valid_mask
        clean = clean * valid_mask

        psnr_metric.update(pred, clean)
        ssim_metric.update(pred, clean)
        with torch.no_grad():
            lpips_values.append(lpips_metric(pred, clean).item())

    return {
        "psnr": psnr_metric.compute().item(),
        "ssim": ssim_metric.compute().item(),
        "lpips_vgg": float(np.mean(lpips_values)),
    }


def write_comparison_video(scene_dir, clean_paths, input_paths, pretrained_paths, finetuned_paths, image_size, fps, video_name):
    video_path = scene_dir / video_name
    writer = imageio.get_writer(video_path, fps=fps)
    try:
        for clean_path, input_path, pretrained_path, finetuned_path in tqdm(
            list(zip(clean_paths, input_paths, pretrained_paths, finetuned_paths)),
            desc=f"Video {scene_dir.name}",
        ):
            frames = [
                np.asarray(load_resized_image(clean_path, image_size)),
                np.asarray(load_resized_image(input_path, image_size)),
                np.asarray(load_resized_image(pretrained_path, image_size)),
                np.asarray(load_resized_image(finetuned_path, image_size)),
            ]
            writer.append_data(np.concatenate(frames, axis=1))
    finally:
        writer.close()
    return video_path


def assert_aligned(clean_paths, corrupted_paths):
    clean_names = [p.name for p in clean_paths]
    corrupted_names = [p.name for p in corrupted_paths]
    if clean_names != corrupted_names:
        raise ValueError("Clean and corrupted image names are not aligned.")


def compute_metrics(args, write_videos=True):
    metrics = {}
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).cuda().eval()

    with torch.no_grad():
        for scene in args.scenes:
            scene_dir = args.data_root / scene
            names = load_split_names(args.split_json, scene, args.split)
            clean_paths = image_paths(scene_dir / "clean", names)
            input_paths = image_paths(scene_dir / args.input_subdir, names)
            pretrained_paths = image_paths(scene_dir / args.pretrained_output_subdir, names)
            finetuned_paths = image_paths(scene_dir / args.finetuned_output_subdir, names)
            mask_paths = image_paths(scene_dir / "mask", names)

            assert_aligned(clean_paths, input_paths)
            assert_aligned(clean_paths, pretrained_paths)
            assert_aligned(clean_paths, finetuned_paths)
            assert_aligned(clean_paths, mask_paths)

            metrics[scene] = {
                args.input_subdir: evaluate_folder(input_paths, clean_paths, mask_paths, args.image_size, lpips_metric),
                args.pretrained_output_subdir: evaluate_folder(pretrained_paths, clean_paths, mask_paths, args.image_size, lpips_metric),
                args.finetuned_output_subdir: evaluate_folder(finetuned_paths, clean_paths, mask_paths, args.image_size, lpips_metric),
            }
            if write_videos:
                video_path = write_comparison_video(
                    scene_dir,
                    clean_paths,
                    input_paths,
                    pretrained_paths,
                    finetuned_paths,
                    args.image_size,
                    args.fps,
                    args.comparison_video_name,
                )
                metrics[scene]["video"] = str(video_path)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(f"Wrote metrics to {args.metrics_path}")


def benchmark(args):
    from model import Difix

    pretrained_model = Difix(
        pretrained_name=args.pretrained_model_name_or_path,
        timestep=args.timestep,
    )
    pretrained_model.set_eval()
    with torch.no_grad():
        for scene in args.scenes:
            scene_dir = args.data_root / scene
            names = load_split_names(args.split_json, scene, args.split)
            input_paths = image_paths(scene_dir / args.input_subdir, names)
            run_model(
                pretrained_model,
                input_paths,
                scene_dir / args.pretrained_output_subdir,
                args.image_size,
                args.prompt,
                args.overwrite,
            )
    del pretrained_model
    torch.cuda.empty_cache()

    finetuned_model = Difix(
        pretrained_name=args.pretrained_model_name_or_path,
        pretrained_path=args.finetuned_checkpoint,
        timestep=args.timestep,
    )
    finetuned_model.set_eval()
    with torch.no_grad():
        for scene in args.scenes:
            scene_dir = args.data_root / scene
            names = load_split_names(args.split_json, scene, args.split)
            input_paths = image_paths(scene_dir / args.input_subdir, names)
            run_model(
                finetuned_model,
                input_paths,
                scene_dir / args.finetuned_output_subdir,
                args.image_size,
                args.prompt,
                args.overwrite,
            )
    del finetuned_model
    torch.cuda.empty_cache()

    compute_metrics(args, write_videos=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--input_subdir", default="corrupted")
    parser.add_argument("--pretrained_output_subdir", default="pretrained")
    parser.add_argument("--finetuned_output_subdir", default="finetuned")
    parser.add_argument("--comparison_video_name", default="comparison_clean_corrupted_pretrained_finetuned.mp4")
    parser.add_argument("--pretrained_model_name_or_path", default="nvidia/difix")
    parser.add_argument("--finetuned_checkpoint", type=Path, default=Path("outputs/difix/train/checkpoints/model_4400.pkl"))
    parser.add_argument("--metrics_path", type=Path, default=Path("data/tagging_zjumocap/egohuman_benchmark_metrics.json"))
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split_json", type=Path, default=None, help="Optional JSON file with train/test splits (keys map to image filenames).")
    parser.add_argument("--split", default="test", help="Which split key to evaluate when --split_json is provided.")
    parser.add_argument("--eval_only", action="store_true", help="Only recompute masked metrics from existing outputs.")
    parser.add_argument("--metrics_only", action="store_true", help="Alias for --eval_only.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only or args.metrics_only:
        compute_metrics(args, write_videos=False)
    else:
        benchmark(args)
