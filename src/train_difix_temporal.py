"""Training script for Difix with a temporal warping loss.

Differences vs. ``train_difix.py``:
  * Uses ``ConsecutiveClipDataset`` so every batch item is a clip of ``V``
    consecutive frames (sampled within a scene).
  * Adds a temporal warping loss computed with torchvision RAFT on the GT
    frames. For every ordered pair (i, j) within a clip we warp pred_j into
    frame i's coordinate system using flow_ij and compute L1 against pred_i
    over forward-backward consistent regions.
"""
import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model import Difix, load_ckpt_from_state_dict, save_ckpt
from dataset import ConsecutiveClipDataset
from loss import gram_loss
from temporal_loss import load_raft, temporal_warp_loss_center


def apply_valid_mask(x, valid_mask):
    return x * valid_mask


def masked_mse_loss(pred, target, valid_mask):
    loss = (pred.float() - target.float()).pow(2) * valid_mask.float()
    denom = valid_mask.float().sum() * pred.shape[1]
    return loss.sum() / denom.clamp_min(1.0)


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_difix = Difix(
        pretrained_name=args.pretrained_model_name_or_path,
        pretrained_pipeline_name_or_path=args.pretrained_pipeline_name_or_path,
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=False,  # temporal training uses the standard UNet
        lora_rank_unet=args.lora_rank_unet if args.use_unet_lora else 0,
        lora_alpha_unet=args.lora_alpha_unet,
        lora_dropout_unet=args.lora_dropout_unet,
        target_modules_unet=args.lora_target_modules_unet,
        freeze_vae=args.freeze_vae,
    )
    net_difix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    raft_model = load_raft(args.raft_variant, device="cuda")

    layers_to_opt = []
    if args.use_unet_lora:
        layers_to_opt += [p for n, p in net_difix.unet.named_parameters()
                          if "lora_" in n and p.requires_grad]
    else:
        layers_to_opt += list(net_difix.unet.parameters())

    if not args.freeze_vae:
        for n, _p in net_difix.vae.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert _p.requires_grad
                layers_to_opt.append(_p)
        layers_to_opt = layers_to_opt + list(net_difix.vae.decoder.skip_conv_1.parameters()) + \
            list(net_difix.vae.decoder.skip_conv_2.parameters()) + \
            list(net_difix.vae.decoder.skip_conv_3.parameters()) + \
            list(net_difix.vae.decoder.skip_conv_4.parameters())

    assert len(layers_to_opt) > 0, "No trainable params collected"
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in layers_to_opt)
        print(f"Trainable params (optimizer): {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    dataset_train = ConsecutiveClipDataset(
        dataset_path=args.dataset_path,
        split="train",
        height=args.resolution,
        width=args.resolution,
        tokenizer=net_difix.tokenizer,
        clip_length=args.clip_length,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    dataset_val = ConsecutiveClipDataset(
        dataset_path=args.dataset_path,
        split="test",
        height=args.resolution,
        width=args.resolution,
        tokenizer=net_difix.tokenizer,
        clip_length=args.clip_length,
    )
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Resume
    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = sorted(
                glob(os.path.join(args.resume, "*.pkl")),
                key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")),
            )
            assert ckpt_files, f"No checkpoint files in {args.resume}"
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(net_difix, optimizer, ckpt_files[-1])
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(net_difix, optimizer, args.resume)
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print("Training from scratch"); print("="*50)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_difix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    raft_model.to(accelerator.device)  # RAFT stays in fp32

    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if accelerator.is_main_process:
        init_kwargs = {"wandb": {"name": args.tracker_run_name, "dir": args.output_dir}}
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                invalid_mask = batch["invalid_pixel_mask"]
                B, V, C, H, W = x_src.shape

                # Forward pass on the whole clip — UNet treats each view as a
                # separate sample but the temporal loss bridges them below.
                x_tgt_pred = net_difix(x_src, prompt_tokens=batch["input_ids"])

                x_tgt_flat = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred_flat = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                invalid_flat = rearrange(invalid_mask, 'b v c h w -> (b v) c h w')
                valid_flat = 1.0 - invalid_flat
                x_tgt_loss = apply_valid_mask(x_tgt_flat, valid_flat)
                x_tgt_pred_loss = apply_valid_mask(x_tgt_pred_flat, valid_flat)

                # Reconstruction losses
                loss_l2 = masked_mse_loss(x_tgt_pred_flat, x_tgt_flat, valid_flat) * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred_loss.float(), x_tgt_loss.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                # Gram loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred_loss * 0.5 + 0.5)
                        crop_h, crop_w = 400, 400
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        x_tgt_renorm = t_vgg_renorm(x_tgt_loss * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        loss_gram = gram_loss(
                            x_tgt_pred_renorm.to(weight_dtype),
                            x_tgt_renorm.to(weight_dtype),
                            net_vgg,
                        ) * args.lambda_gram
                        loss = loss + loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)
                else:
                    loss_gram = torch.tensor(0.0).to(weight_dtype)

                # Center-only temporal warping loss (per clip, averaged across the batch)
                temp_viz = None
                if args.lambda_temporal > 0 and V > 1:
                    loss_temp_sum = x_tgt_pred.new_zeros(())
                    valid_frac_sum = x_tgt_pred.new_zeros(())
                    for b in range(B):
                        gt_clip = x_tgt[b].to(torch.float32)              # (V, C, H, W)
                        extra_valid = (1.0 - invalid_mask[b]).to(x_tgt_pred.dtype)  # (V, 1, H, W)
                        l_temp, vf, viz = temporal_warp_loss_center(
                            x_tgt_pred[b], gt_clip, raft_model,
                            alpha=args.temporal_alpha, beta=args.temporal_beta,
                            extra_valid=extra_valid,
                        )
                        loss_temp_sum = loss_temp_sum + l_temp
                        valid_frac_sum = valid_frac_sum + vf
                        if b == 0:
                            temp_viz = viz
                    loss_temp = loss_temp_sum / max(B, 1) * args.lambda_temporal
                    valid_frac = valid_frac_sum / max(B, 1)
                    loss = loss + loss_temp
                else:
                    loss_temp = torch.tensor(0.0).to(weight_dtype)
                    valid_frac = torch.tensor(0.0).to(weight_dtype)

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_temporal": loss_temp.detach().item(),
                        "valid_frac": valid_frac.detach().item(),
                    }
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        if temp_viz is not None:
                            # Horizontal concat: pred_center | warped_pred_j * valid_j (black = masked).
                            # Convert from [-1, 1] to [0, 1] so masked regions render as actual black.
                            center01 = (temp_viz["pred_center"] * 0.5 + 0.5).clamp(0, 1)
                            panels = [center01]
                            for warped_j, mask_j in zip(temp_viz["warped_preds"], temp_viz["valid_masks"]):
                                w01 = (warped_j * 0.5 + 0.5).clamp(0, 1)
                                panels.append(w01 * mask_j)
                            warp_viz = torch.cat(panels, dim=-1)  # (C, H, k*W)
                            log_dict["train/temporal_warp"] = wandb.Image(
                                warp_viz.float().detach().cpu(),
                                caption=f"center | warped non-center frames (masked) — step {global_step}",
                            )
                        logs.update(log_dict)

                    if global_step > 0 and global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)

                    if args.eval_freq > 0 and global_step > 0 and global_step % args.eval_freq == 0:
                        l_l2, l_lpips, l_temp, l_vf = [], [], [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for vstep, batch_val in enumerate(dl_val):
                            if vstep >= args.num_samples_eval:
                                break
                            xs = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            xt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            im = batch_val["invalid_pixel_mask"].to(accelerator.device, dtype=weight_dtype)
                            Bv, Vv, Cv, Hv, Wv = xs.shape
                            assert Bv == 1, "Use batch size 1 for eval."
                            c_val = Vv // 2
                            with torch.no_grad():
                                xt_pred = accelerator.unwrap_model(net_difix)(xs, prompt_tokens=batch_val["input_ids"].cuda())
                                if vstep % 10 == 0:
                                    log_dict["sample/source"].append(wandb.Image(rearrange(xs, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))
                                    log_dict["sample/target"].append(wandb.Image(rearrange(xt, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))
                                    log_dict["sample/model_output"].append(wandb.Image(rearrange(xt_pred, "b v c h w -> b c (v h) w")[0].float().detach().cpu()))
                                # Score the central frame of each clip — the frame the temporal loss optimizes.
                                xt_c = xt[:, c_val]; xtp_c = xt_pred[:, c_val]
                                valid_c = 1.0 - im[:, c_val]
                                l_l2.append(masked_mse_loss(xtp_c, xt_c, valid_c).item())
                                l_lpips.append(
                                    net_lpips(
                                        apply_valid_mask(xtp_c, valid_c).float(),
                                        apply_valid_mask(xt_c, valid_c).float(),
                                    ).mean().item()
                                )
                                # Temporal warping loss on validation clips (same formulation as training).
                                if Vv > 1:
                                    extra_valid_val = (1.0 - im[0]).to(xt_pred.dtype)
                                    lt, vf, _ = temporal_warp_loss_center(
                                        xt_pred[0], xt[0].to(torch.float32), raft_model,
                                        alpha=args.temporal_alpha, beta=args.temporal_beta,
                                        extra_valid=extra_valid_val,
                                    )
                                    l_temp.append(lt.item())
                                    l_vf.append(vf.item())
                        logs["val/l2"] = float(np.mean(l_l2))
                        logs["val/lpips"] = float(np.mean(l_lpips))
                        if l_temp:
                            logs["val/temporal"] = float(np.mean(l_temp))
                            logs["val/valid_frac"] = float(np.mean(l_vf))
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect(); torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Loss weights
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)
    parser.add_argument("--lambda_temporal", default=1.0, type=float,
        help="Weight for temporal warping L1 loss. Defaults to match lambda_l2.")
    parser.add_argument("--temporal_alpha", default=0.01, type=float,
        help="Forward-backward consistency relative threshold.")
    parser.add_argument("--temporal_beta", default=0.5, type=float,
        help="Forward-backward consistency absolute threshold.")
    parser.add_argument("--raft_variant", default="large", choices=["large", "small"])

    # Dataset / clip
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--clip_length", default=3, type=int,
        help="Number of consecutive frames per training clip (>= 2). The "
             "temporal loss is computed at the central frame (index k//2).")

    # Validation eval
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100)

    parser.add_argument("--viz_freq", type=int, default=100)
    parser.add_argument("--tracker_project_name", type=str, default="difix")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    # Model
    parser.add_argument("--pretrained_model_name_or_path", default="nvidia/difix")
    parser.add_argument("--pretrained_pipeline_name_or_path", default=None, type=str)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--use_unet_lora", action="store_true")
    parser.add_argument("--lora_rank_unet", type=int, default=8)
    parser.add_argument("--lora_alpha_unet", type=int, default=None)
    parser.add_argument("--lora_dropout_unet", type=float, default=0.0)
    parser.add_argument("--lora_target_modules_unet", nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"])
    parser.add_argument("--freeze_vae", action="store_true")

    # Training
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()
    if not args.pretrained_model_name_or_path:
        args.pretrained_model_name_or_path = None
    if args.clip_length < 2:
        raise ValueError("--clip_length must be >= 2 for temporal training")

    main(args)
