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
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model import Difix, load_ckpt_from_state_dict, save_ckpt
from dataset import PairedDataset
from loss import gram_loss


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
        mv_unet=args.mv_unet,
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

    # make the optimizer
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

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    dataset_train = PairedDataset(
        dataset_path=args.dataset_path,
        split="train",
        height=args.resolution,
        width=args.resolution,
        tokenizer=net_difix.tokenizer,
    )
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(
        dataset_path=args.dataset_path,
        split="test",
        height=args.resolution,
        width=args.resolution,
        tokenizer=net_difix.tokenizer,
    )
    # random.Random(42).shuffle(dataset_val.img_img_idsnames)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Resume from checkpoint
    global_step = 0    
    if args.resume is not None:
        if os.path.isdir(args.resume):
            # Resume from last ckpt
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, args.resume
            )    
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_difix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare everything with our `accelerator`.
    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    # renorm with image net statistics
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }        
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                invalid_mask = batch["invalid_pixel_mask"]
                B, V, C, H, W = x_src.shape

                # forward pass
                x_tgt_pred = net_difix(x_src, prompt_tokens=batch["input_ids"])       
                
                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                invalid_mask = rearrange(invalid_mask, 'b v c h w -> (b v) c h w')
                valid_mask = 1.0 - invalid_mask
                x_tgt_loss = apply_valid_mask(x_tgt, valid_mask)
                x_tgt_pred_loss = apply_valid_mask(x_tgt_pred, valid_mask)
                         
                # Reconstruction loss
                loss_l2 = masked_mse_loss(x_tgt_pred, x_tgt, valid_mask) * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred_loss.float(), x_tgt_loss.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                
                # Gram matrix loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred_loss * 0.5 + 0.5)
                        crop_h, crop_w = 400, 400
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        
                        x_tgt_renorm = t_vgg_renorm(x_tgt_loss * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        
                        loss_gram = gram_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)                    

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        # Dataset normalizes to [-1, 1]; denormalize back to [0, 1] for wandb.
                        src_viz = (rearrange(x_src, "b v c h w -> b c (v h) w") * 0.5 + 0.5).clamp(0, 1).float().detach().cpu()
                        tgt_viz = (rearrange(x_tgt, "b v c h w -> b c (v h) w") * 0.5 + 0.5).clamp(0, 1).float().detach().cpu()
                        pred_viz = (rearrange(x_tgt_pred, "b v c h w -> b c (v h) w") * 0.5 + 0.5).clamp(0, 1).float().detach().cpu()
                        log_dict = {
                            "train/source": [wandb.Image(src_viz[idx], caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(tgt_viz[idx], caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(pred_viz[idx], caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step > 0 and global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # accelerator.unwrap_model(net_difix).save_model(outf)
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)

                    # compute validation set L2, LPIPS
                    if args.eval_freq > 0 and global_step > 0 and global_step % args.eval_freq == 0:
                        l_l2, l_lpips = [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            invalid_mask = batch_val["invalid_pixel_mask"].to(accelerator.device, dtype=weight_dtype)
                            B, V, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_difix)(x_src, prompt_tokens=batch_val["input_ids"].cuda())
                                
                                if step % 10 == 0:
                                    # Dataset normalizes to [-1, 1]; denormalize back to [0, 1] for wandb.
                                    src_viz = (rearrange(x_src, "b v c h w -> b c (v h) w")[0] * 0.5 + 0.5).clamp(0, 1).float().detach().cpu()
                                    tgt_viz = (rearrange(x_tgt, "b v c h w -> b c (v h) w")[0] * 0.5 + 0.5).clamp(0, 1).float().detach().cpu()
                                    pred_viz = (rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[0] * 0.5 + 0.5).clamp(0, 1).float().detach().cpu()
                                    log_dict["sample/source"].append(wandb.Image(src_viz, caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/target"].append(wandb.Image(tgt_viz, caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/model_output"].append(wandb.Image(pred_viz, caption=f"idx={len(log_dict['sample/source'])}"))
                                
                                x_tgt = x_tgt[:, 0] # take the input view
                                x_tgt_pred = x_tgt_pred[:, 0] # take the input view
                                valid_mask = 1.0 - invalid_mask[:, 0]
                                x_tgt_loss = apply_valid_mask(x_tgt, valid_mask)
                                x_tgt_pred_loss = apply_valid_mask(x_tgt_pred, valid_mask)
                                # compute the reconstruction losses
                                loss_l2 = masked_mse_loss(x_tgt_pred, x_tgt, valid_mask)
                                loss_lpips = net_lpips(x_tgt_pred_loss.float(), x_tgt_loss.float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)

    # Always save a final checkpoint when the epoch loop finishes — guards
    # against the case where total steps < checkpointing_steps.
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "checkpoints", f"model_final_{global_step}.pkl")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, final_path)
        print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset options
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="difix", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path", default="nvidia/difix", help="Hugging Face DifixPipeline repo to initialize from")
    parser.add_argument("--pretrained_pipeline_name_or_path", default=None, type=str,
        help="Optional Hugging Face DifixPipeline repo/path to load tokenizer, text encoder, scheduler, UNet, and VAE from.")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")

    # UNet LoRA / VAE freezing
    parser.add_argument("--use_unet_lora", action="store_true",
        help="Train a LoRA adapter on the UNet instead of full fine-tuning.")
    parser.add_argument("--lora_rank_unet", type=int, default=8)
    parser.add_argument("--lora_alpha_unet", type=int, default=None,
        help="LoRA alpha for UNet. Defaults to 2 * lora_rank_unet.")
    parser.add_argument("--lora_dropout_unet", type=float, default=0.0)
    parser.add_argument("--lora_target_modules_unet", nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
        help="UNet module name suffixes to apply LoRA to.")
    parser.add_argument("--freeze_vae", action="store_true",
        help="Freeze VAE entirely (no decoder LoRA updates, no skip-conv updates).")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resize train/test images to resolution×resolution (square).",
    )
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()
    if not args.pretrained_model_name_or_path:
        args.pretrained_model_name_or_path = None

    main(args)
