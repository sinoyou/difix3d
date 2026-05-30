import json
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        mask_img = self.data[img_id]["mask"] if "mask" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"]
        
        try:
            input_img = Image.open(input_img).convert("RGB")
            output_img = Image.open(output_img).convert("RGB")
            mask_img = Image.open(mask_img).convert("L") if mask_img is not None else None
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)

        img_t = F.to_tensor(input_img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if mask_img is not None:
            mask_t = F.to_tensor(mask_img)
            mask_t = F.resize(mask_t, self.image_size, interpolation=InterpolationMode.NEAREST)
            mask_t = (mask_t > 0.5).float()
        else:
            mask_t = torch.zeros((1, *self.image_size), dtype=output_t.dtype)

        if ref_img is not None:
            ref_img = Image.open(ref_img).convert("RGB")
            ref_t = F.to_tensor(ref_img)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)
            mask_t = torch.stack([mask_t, torch.zeros_like(mask_t)], dim=0)
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)
            mask_t = mask_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "invalid_pixel_mask": mask_t,
            "caption": caption,
        }
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out


class ConsecutiveClipDataset(torch.utils.data.Dataset):
    """Yields a clip of `clip_length` consecutive frames per item.

    Each entry's `image` / `target_image` / `mask` are stacked along a leading V
    dimension so the existing (B, V, C, H, W) training pipeline works unchanged.
    Items are indexed by clip-start positions within each scene (a scene is
    defined by the JSON key prefix before the trailing `_NNNNNN` index).
    """

    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None, clip_length=4):
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        # Group keys by scene prefix, sort by trailing frame index.
        scenes = {}
        for key, entry in self.data.items():
            prefix, idx = key.rsplit("_", 1)
            scenes.setdefault(prefix, []).append((int(idx), key, entry))
        for prefix in scenes:
            scenes[prefix].sort()
        self.scenes = scenes
        self.clip_length = clip_length
        # Valid (scene, start) positions where a full clip fits.
        self.starts = []
        for prefix, frames in scenes.items():
            for s in range(0, len(frames) - clip_length + 1):
                self.starts.append((prefix, s))
        if not self.starts:
            raise ValueError(
                f"No clips of length {clip_length} can be formed from {dataset_path}:{split}"
            )
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        prefix, start = self.starts[idx]
        clip = self.scenes[prefix][start:start + self.clip_length]

        imgs, tgts, masks = [], [], []
        caption = None
        for _, _key, entry in clip:
            input_path = entry["image"]
            target_path = entry["target_image"]
            mask_path = entry.get("mask")
            caption = entry["prompt"]

            try:
                input_img = Image.open(input_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
                mask_img = Image.open(mask_path).convert("L") if mask_path else None
            except Exception:
                print("Error loading clip frame:", input_path, target_path)
                return self.__getitem__((idx + 1) % len(self))

            img_t = F.to_tensor(input_img)
            img_t = F.resize(img_t, self.image_size)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

            tgt_t = F.to_tensor(target_img)
            tgt_t = F.resize(tgt_t, self.image_size)
            tgt_t = F.normalize(tgt_t, mean=[0.5], std=[0.5])

            if mask_img is not None:
                mask_t = F.to_tensor(mask_img)
                mask_t = F.resize(mask_t, self.image_size, interpolation=InterpolationMode.NEAREST)
                mask_t = (mask_t > 0.5).float()
            else:
                mask_t = torch.zeros((1, *self.image_size), dtype=tgt_t.dtype)

            imgs.append(img_t)
            tgts.append(tgt_t)
            masks.append(mask_t)

        img_t = torch.stack(imgs, dim=0)   # (V, C, H, W)
        tgt_t = torch.stack(tgts, dim=0)
        mask_t = torch.stack(masks, dim=0)

        out = {
            "output_pixel_values": tgt_t,
            "conditioning_pixel_values": img_t,
            "invalid_pixel_mask": mask_t,
            "caption": caption,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
