"""Temporal warping loss using torchvision RAFT.

Workflow per training step:
  1) Compute bidirectional optical flow between every ordered GT frame pair
     (V*(V-1) flows) with torchvision RAFT (no_grad).
  2) Forward-backward consistency check yields a per-pair validity mask in the
     source frame's coordinate system.
  3) Warp the model's predicted frame_j into frame_i's coordinates with
     flow_ij and take L1 against pred_i over the valid mask.
The loss is averaged over all valid pairs.
"""
from typing import Tuple

import torch
import torch.nn.functional as F

from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)


def load_raft(variant: str = "large", device: str = "cuda"):
    if variant == "large":
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    elif variant == "small":
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
    else:
        raise ValueError(f"Unknown RAFT variant: {variant}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_grid(B: int, H: int, W: int, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    return grid  # (B, 2, H, W) in pixel coords


def warp_with_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``img`` by ``flow``.

    Convention (matches torchvision RAFT): ``flow[b, :, y, x] = (dx, dy)`` is
    the displacement from pixel ``(x, y)`` in the source frame to its
    correspondence in the target frame, so ``warped[y, x] = img[y+dy, x+dx]``
    via bilinear sampling.

    Args:
        img: (B, C, H, W) tensor to be sampled.
        flow: (B, 2, H, W) flow defined at each pixel of the *source* frame.
    Returns:
        (B, C, H, W) warped image in the source-frame coordinate system.
    """
    B, _, H, W = flow.shape
    grid = _build_grid(B, H, W, flow.device, flow.dtype)
    sample = grid + flow
    sample_x = 2.0 * sample[:, 0] / max(W - 1, 1) - 1.0
    sample_y = 2.0 * sample[:, 1] / max(H - 1, 1) - 1.0
    sample = torch.stack([sample_x, sample_y], dim=-1)  # (B, H, W, 2)
    return F.grid_sample(
        img, sample, mode="bilinear", padding_mode="zeros", align_corners=True
    )


@torch.no_grad()
def compute_pairwise_flows(raft, gt_frames: torch.Tensor) -> torch.Tensor:
    """Compute flow for every ordered i != j pair.

    Args:
        gt_frames: (V, C, H, W) in [-1, 1].
    Returns:
        flows: (V, V, 2, H, W) where ``flows[i, j]`` is the flow from frame i
        to frame j. Diagonal entries are zero.
    """
    V, C, H, W = gt_frames.shape
    pairs_i, pairs_j = [], []
    for i in range(V):
        for j in range(V):
            if i != j:
                pairs_i.append(i)
                pairs_j.append(j)

    img1 = gt_frames[pairs_i].contiguous()
    img2 = gt_frames[pairs_j].contiguous()
    # torchvision RAFT expects images already in [-1, 1] floats.
    flow_list = raft(img1.float(), img2.float())
    flow = flow_list[-1]  # (P, 2, H, W)

    flows = torch.zeros(
        (V, V, 2, H, W), device=gt_frames.device, dtype=flow.dtype
    )
    for p, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        flows[i, j] = flow[p]
    return flows


def occlusion_mask(
    flow_ij: torch.Tensor, flow_ji: torch.Tensor, alpha: float = 0.01, beta: float = 0.5
) -> torch.Tensor:
    """Forward-backward consistency mask in frame i's coordinate system.

    Following Brox/Sundaram: a pixel is "valid" (non-occluded) if
    ``|flow_ij + warp(flow_ji, flow_ij)|^2 < alpha * (|flow_ij|^2 + |warp(flow_ji, flow_ij)|^2) + beta``.
    """
    f_ji_warped = warp_with_flow(flow_ji, flow_ij)
    diff = flow_ij + f_ji_warped
    mag_sq = (diff ** 2).sum(dim=1, keepdim=True)
    ref_sq = (flow_ij ** 2).sum(dim=1, keepdim=True) + (f_ji_warped ** 2).sum(dim=1, keepdim=True)
    valid = mag_sq < alpha * ref_sq + beta
    return valid


def temporal_warp_loss_center(
    preds: torch.Tensor,
    gt_frames: torch.Tensor,
    raft,
    alpha: float = 0.01,
    beta: float = 0.5,
    extra_valid: torch.Tensor = None,
):
    """Center-only temporal warping loss.

    For each non-center frame ``j`` in a clip of ``V`` frames (center index
    ``c = V // 2``), compute bidirectional GT optical flow between frames
    ``c`` and ``j``, warp ``pred_j`` into ``c``'s coordinate system using
    ``flow_{c->j}``, then take L1 against ``pred_c`` over a validity mask
    formed from the forward-backward consistency check AND ``extra_valid[c]``
    (a per-frame global mask, e.g. the dataset's non-corrupted region in
    frame ``c``'s coords).

    Args:
        preds: (V, C, H, W) predicted frames in [-1, 1].
        gt_frames: (V, C, H, W) GT frames in [-1, 1] (used for flow only).
        raft: torchvision RAFT model in eval mode.
        alpha, beta: forward-backward consistency thresholds.
        extra_valid: optional (V, 1, H, W) per-frame validity mask. Only the
            central slice ``extra_valid[c]`` is used.

    Returns:
        loss: scalar warp L1 loss (mean over the ``V - 1`` non-center frames).
        valid_fraction: mean fraction of valid pixels across the pairs.
        viz: dict with ``pred_center`` (C, H, W) and lists ``warped_preds``,
            ``valid_masks`` of length ``V - 1`` for logging.
    """
    V, C, H, W = preds.shape
    if V < 2:
        return preds.new_zeros(()), preds.new_zeros(()), None
    c = V // 2
    other = [j for j in range(V) if j != c]

    # Build flow pairs: for each non-center j we need flow(c->j) and flow(j->c).
    pair_src, pair_dst = [], []
    for j in other:
        pair_src.append(c); pair_dst.append(j)  # flow(c -> j)
        pair_src.append(j); pair_dst.append(c)  # flow(j -> c)

    with torch.no_grad():
        img1 = gt_frames[pair_src].contiguous().float()
        img2 = gt_frames[pair_dst].contiguous().float()
        flow_list = raft(img1, img2)
        flows_pair = flow_list[-1]  # (2 * len(other), 2, H, W)

    pred_c = preds[c]
    extra_valid_c = extra_valid[c] if extra_valid is not None else None

    loss_total = preds.new_zeros(())
    valid_frac_total = preds.new_zeros(())
    warped_list, mask_list = [], []

    for k, j in enumerate(other):
        f_cj = flows_pair[2 * k].to(preds.dtype).unsqueeze(0)        # in c's coords
        f_jc = flows_pair[2 * k + 1].to(preds.dtype).unsqueeze(0)    # in j's coords
        pred_j_b = preds[j].unsqueeze(0)

        warped_pred_j = warp_with_flow(pred_j_b, f_cj)               # (1, C, H, W) in c
        valid = occlusion_mask(f_cj, f_jc, alpha=alpha, beta=beta).to(preds.dtype)
        if extra_valid_c is not None:
            valid = valid * extra_valid_c.unsqueeze(0).to(preds.dtype)

        l1 = (warped_pred_j - pred_c.unsqueeze(0)).abs()
        denom = valid.sum() * C
        loss_j = (l1 * valid).sum() / denom.clamp_min(1.0)

        loss_total = loss_total + loss_j
        valid_frac_total = valid_frac_total + valid.mean()
        warped_list.append(warped_pred_j.squeeze(0))
        mask_list.append(valid.squeeze(0))

    n = max(len(other), 1)
    viz = {
        "pred_center": pred_c.detach(),
        "warped_preds": [w.detach() for w in warped_list],
        "valid_masks": [m.detach() for m in mask_list],
    }
    return loss_total / n, valid_frac_total / n, viz
