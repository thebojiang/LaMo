#!/usr/bin/env python3

# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified interpretability for lamo.

For each sampled OpenVid caption we:

    1. Generate ONE video with the lamo pipeline (LoRA on; motion guidance
       optional via --guidance_lambda, OFF by default).
    2. On the resulting latent, run BOTH analyses at the same latent frame
       index ``t*`` (chosen as ``argmax_t ||BMV(t)||`` -- the pair with the
       most motion, where any "captures motion" claim is most testable):

         a) BMV-projection map (validates the BMV-relL2 loss):
              R_bmv(h, w) = | Delta(t*)[:, h, w] . BMV(t*) | / ||BMV(t*)||
            where Delta(t*) = z[:, t*+tau] - z[:, t*],  BMV = mean_{H,W}(Delta).

         b) Motion predictor spatial response (validates motion guidance):

              z_static    = mean_{t} latents[:, t]                 [1,16,H,W]
              delta_t     = predictor(z[:, t*],  prompt_pooled)
              delta_base  = predictor(z_static,  prompt_pooled)
              R_pred(h,w) = || (delta_t - delta_base)[:, h, w] ||_2

            BMV-relL2 only supervises the spatial *mean* of the
            predictor's 16-D output; per-pixel sparsity is unconstrained
            and a CNN forward produces non-zero "default response" at
            every pixel, so plain ``||delta_pred||`` (and even projecting
            onto ``mean(delta_pred)``) looks dense everywhere. Subtracting
            the predictor's response on a temporally-averaged baseline
            ``z_static`` -- where motion has been averaged out -- cancels
            this CNN noise floor and isolates the motion-specific
            response: static regions of ``z[t]`` equal those of
            ``z_static`` and (by CNN locality) produce near-identical
            predictor outputs there, so the residual is ~0 in the
            background. This mirrors the BMV trick: both rely on a
            differencing operation to make the underlying signal
            naturally sparse before the heatmap collapse.

            Pass ``--predictor_metric proj`` or ``norm`` to compare with
            the dense alternatives.

       Both heatmaps share the same RGB frame so the user can eyeball
       whether each highlights the same moving regions in the RGB.

    3. Save THREE images per sample into a sub-folder named after the
       prompt:
              ``{idx:03d}_{slug(caption)}/``
                  rgb.png                -- the RGB frame at t*
                  bmv_heatmap.png        -- pure BMV-projection heatmap
                                            (no RGB underneath; upsampled to
                                            RGB resolution)
                  predictor_heatmap.png  -- pure R_pred heatmap
                  info.txt               -- caption, t*, ||BMV||, Pearson(R_bmv, R_pred)

    4. Aggregate a combined N x 3 grid (RGB | pure BMV heatmap | pure
       predictor heatmap) at ``combined_grid.png`` and dump every raw
       array to ``arrays.npz`` for offline analysis.

Why both heatmaps at the SAME ``t*``:
    The two methods analyse different aspects of the same latent. Putting
    them on the same RGB frame makes "do they both pick out the moving
    subject?" a direct visual question.

Why motion guidance OFF by default during generation:
    With guidance on, the predictor's gradients have already shaped the
    latent toward what the predictor likes -- evaluating the predictor on
    that latent is partly tautological. Default off so both heatmaps probe
    the lamo-trained transformer + LoRA on its own. Pass
    ``--guidance_lambda 25`` to re-enable for the production setting.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python tests/interpret_lamo.py \\
        --model_path     .../CogVideoX-5b-Diffusers \\
        --lora_path      .../pytorch_lora_weights.safetensors \\
        --predictor_ckpt .../predictor.safetensors \\
        --num_samples 6 \\
        --work_dir outputs/interpret_lamo
"""

import argparse
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from _interpret_common import (
    add_common_args,
    build_lamo_pipeline_and_predictor,
    encode_prompt_full_and_pooled,
    load_caption_csv,
    parse_dtype,
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def slugify(text: str, max_len: int = 60) -> str:
    """Turn a caption into a filesystem-safe folder name."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return s[:max_len] if s else "untitled"


def _safe_normalize_map(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalise a 2D map to [0, 1] (constant maps -> zeros)."""
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - a_min) / (a_max - a_min)).astype(np.float32)


def _upsample_to(map_lat: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear upsample a (H_lat, W_lat) map to (target_h, target_w)."""
    t = torch.from_numpy(map_lat.astype(np.float32))[None, None, ...]
    up = F.interpolate(
        t, size=(target_h, target_w), mode="bilinear", align_corners=False,
    )
    return up.squeeze().numpy()


def _latent_t_to_rgb_index(
    latent_t: int, num_latent: int, num_rgb: int,
) -> int:
    """Map latent frame index ``t`` to a representative RGB frame index.

    CogVideoX VAE temporally compresses 4:1 with a leading singleton
    (49 RGB <-> 13 latent). Aim near the centre of latent ``t``'s chunk.
    """
    if num_latent <= 1:
        return 0
    rgb_per_latent = max(1, (num_rgb - 1) // (num_latent - 1))
    rgb_idx = int(round(latent_t * rgb_per_latent))
    return max(0, min(rgb_idx, num_rgb - 1))


# ---------------------------------------------------------------------------
# Per-sample numerics
# ---------------------------------------------------------------------------


def compute_bmv_norms_per_pair(latents: torch.Tensor, tau: int) -> np.ndarray:
    """Return ||BMV(t)|| for every t in [0, T-tau)."""
    if latents.shape[1] <= tau:
        raise ValueError(
            f"Need T > tau; got T={latents.shape[1]}, tau={tau}"
        )
    delta = (latents[:, tau:] - latents[:, :-tau]).float()      # [1, T-tau, C, H, W]
    bmv = delta.mean(dim=(-2, -1))                              # [1, T-tau, C]
    return bmv.norm(dim=-1)[0].cpu().numpy()


def compute_bmv_response_at(
    latents: torch.Tensor, t_idx: int, tau: int,
):
    """BMV-projection heatmap at fixed t.

    Returns:
        response_map  -- [H, W] float32, ``|Delta . BMV| / ||BMV||``, abs.
        bmv           -- [C]  float32, the 16-D BMV vector
        delta_norm    -- [H, W] float32, ``||Delta[:, h, w]||`` (raw motion magnitude)
        bmv_norm      -- python float, ``||BMV||``
    """
    delta = (latents[:, t_idx + tau] - latents[:, t_idx]).float()  # [1, C, H, W]
    d = delta[0]                                                   # [C, H, W]
    bmv = d.mean(dim=(-2, -1))                                     # [C]
    bmv_n = bmv.norm().clamp(min=1e-8)
    proj = (d * bmv.view(-1, 1, 1)).sum(dim=0) / bmv_n             # [H, W] signed
    response = proj.abs().cpu().numpy().astype(np.float32)
    delta_norm = d.norm(dim=0).cpu().numpy().astype(np.float32)
    return response, bmv.cpu().numpy().astype(np.float32), delta_norm, float(bmv_n.item())


@torch.no_grad()
def compute_predictor_response_at(
    predictor,
    latents_full: torch.Tensor,
    t_idx: int,
    prompt_pooled: torch.Tensor,
    metric: str = "resid",
) -> np.ndarray:
    """Predictor response heatmap at frame ``t_idx`` -> (H, W).

    Unlike the BMV heatmap, the predictor's 16-D output is dense (CNN
    forwards always produce non-zero values at every spatial position).
    BMV-relL2 only supervises the *mean*, so per-pixel sparsity is
    unconstrained. Different metrics try to recover a clean motion-
    localised heatmap from this dense output:

      - ``"resid"`` (default, **recommended**): residual against a
        no-motion baseline.

            z_static    = mean_{t} latents_full[:, t]            [1,16,H,W]
            delta_t     = predictor(z[:, t_idx])                 [1,16,H,W]
            delta_base  = predictor(z_static)                    [1,16,H,W]
            R_pred(h,w) = || (delta_t - delta_base)[:, h, w] ||_2

        Why it cleans up: ``z_static`` averages motion out across time,
        so the predictor sees an in-distribution but motionless input.
        Whatever CNN-architectural "default response" the predictor
        outputs (the noise floor) appears identically in both forward
        passes and cancels in the difference. The residual is the
        predictor's response that exists ONLY because of the motion
        content present at frame ``t_idx``. Static regions of ``z[t]``
        equal those of ``z_static`` and -- thanks to CNN locality --
        produce near-identical predictor outputs there, so the residual
        is near zero in static background.

        Mirrors the BMV heatmap's logic: both rely on the input signal
        being naturally sparse via differencing (``z[t+tau] - z[t]`` for
        BMV; ``predictor(z[t]) - predictor(z_static)`` here).
        Falsifiability is preserved: if the predictor's output is
        invariant to motion content (degenerate / prompt-only), the
        residual is zero everywhere -> a fully dark heatmap that does
        not align with motion in the RGB.

      - ``"proj"``: project predictor's own output onto its spatial
        mean direction (``bmv_pred``). Looks dense in practice because
        ``mean_{H,W}(delta_pred . bmv_pred) = ||bmv_pred||^2`` by
        definition -- the average projection is large, so half the
        spatial cells are near or above this floor.

      - ``"norm"`` (previous): R(h,w) = ``||delta_pred(:, h, w)||_2``.
        Plain channel-wise L2 norm. CNN noise floor everywhere.
    """
    z_t = latents_full[:, t_idx]                          # [1, 16, H, W]

    if metric == "resid":
        z_static = latents_full.mean(dim=1)               # [1, 16, H, W]
        delta_t = predictor(z_t, prompt_pooled=prompt_pooled, prev_delta=None)
        delta_base = predictor(z_static, prompt_pooled=prompt_pooled, prev_delta=None)
        residual = (delta_t - delta_base).float()         # [1, 16, H, W]
        return residual.norm(dim=1)[0].cpu().numpy().astype(np.float32)

    if metric == "proj":
        delta_pred = predictor(z_t, prompt_pooled=prompt_pooled, prev_delta=None)
        d0 = delta_pred[0].float()                        # [16, H, W]
        bmv_pred = d0.mean(dim=(-2, -1))                  # [16]
        bmv_pred_n = bmv_pred.norm().clamp(min=1e-8)
        proj = (d0 * bmv_pred.view(-1, 1, 1)).sum(dim=0) / bmv_pred_n  # [H, W]
        return proj.abs().cpu().numpy().astype(np.float32)

    if metric == "norm":
        delta_pred = predictor(z_t, prompt_pooled=prompt_pooled, prev_delta=None)
        return delta_pred.float().norm(dim=1)[0].cpu().numpy().astype(np.float32)

    raise ValueError(
        f"Unknown predictor metric: {metric!r}; expected 'resid', 'proj', or 'norm'."
    )


# ---------------------------------------------------------------------------
# Per-sample image saving
# ---------------------------------------------------------------------------


def _save_rgb_png(path: str, rgb: np.ndarray):
    """Save an HxWx3 uint8 array as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=160)
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_full_video_mp4(
    path: str, rgb_video: torch.Tensor, fps: int = 8,
):
    """Save the decoded video tensor as MP4.

    rgb_video: [1, 3, T, H, W] float in [-1, 1]  (output of pipe.decode_latents).

    NOTE: ``diffusers.utils.export_to_video`` internally does
    ``(frame * 255).astype(np.uint8)`` whenever the input is a ``np.ndarray``
    (it expects FLOAT [0, 1] frames -- see export_utils.py).  If we hand it
    uint8 [0, 255] frames, that ``* 255`` overflows uint8 and the colours
    look "inverted" (e.g. 200 -> 200*255=51000 -> 51000 % 256 = 8).  So we
    must pass float32 [0, 1] and let ``export_to_video`` do the conversion.
    """
    from diffusers.utils import export_to_video

    frames = rgb_video[0]                                 # [3, T, H, W]
    # [-1, 1] -> [0, 1] float32 (do NOT pre-multiply by 255)
    frames = ((frames.float().clamp(-1, 1) + 1.0) * 0.5)
    frames_np = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)  # [T, H, W, 3]
    frames_list = [frames_np[t] for t in range(frames_np.shape[0])]
    export_to_video(frames_list, path, fps=fps)


def _save_heatmap_png(
    path: str, heat_lat: np.ndarray, target_h: int, target_w: int,
    cmap_name: str = "inferno",
):
    """Save a pure heatmap PNG (no RGB underneath).

    The map is bilinear-upsampled to ``(target_h, target_w)`` so the file
    has the same dimensions as ``rgb.png`` and can be opened side-by-side.
    Per-image min-max normalisation -> values map to [0, 1] before colormap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heat_up = _upsample_to(heat_lat, target_h, target_w)
    heat_norm = _safe_normalize_map(heat_up)

    fig, ax = plt.subplots(figsize=(target_w / 100, target_h / 100), dpi=160)
    ax.imshow(heat_norm, cmap=cmap_name, vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_sample_outputs(
    sample_dir: str,
    rgb: np.ndarray,
    bmv_map: np.ndarray,
    pred_map: np.ndarray,
    info: dict,
    cmap_name: str = "inferno",
):
    """Write {rgb, bmv_heatmap, predictor_heatmap}.png + info.txt.

    Heatmaps are pure (no RGB underneath) but rendered at the same
    (H_rgb, W_rgb) as ``rgb.png`` so users can flip between them or
    overlay externally if they want.
    """
    os.makedirs(sample_dir, exist_ok=True)
    H_rgb, W_rgb = rgb.shape[:2]

    _save_rgb_png(os.path.join(sample_dir, "rgb.png"), rgb)
    _save_heatmap_png(
        os.path.join(sample_dir, "bmv_heatmap.png"),
        bmv_map, H_rgb, W_rgb, cmap_name=cmap_name,
    )
    _save_heatmap_png(
        os.path.join(sample_dir, "predictor_heatmap.png"),
        pred_map, H_rgb, W_rgb, cmap_name=cmap_name,
    )

    with open(os.path.join(sample_dir, "info.txt"), "w") as f:
        for k in [
            "caption", "latent_t", "rgb_t", "bmv_norm",
            "pearson_bmv_pred", "pearson_bmv_motion",
        ]:
            if k in info:
                v = info[k]
                if isinstance(v, float):
                    f.write(f"{k}: {v:+.4f}\n")
                else:
                    f.write(f"{k}: {v}\n")


# ---------------------------------------------------------------------------
# Combined grid figure
# ---------------------------------------------------------------------------


def plot_combined_grid(
    rows_data,
    output_path: str,
    cmap_name: str = "inferno",
    title_suffix: str = "",
):
    """N x 3 grid: RGB | pure BMV heatmap | pure predictor heatmap.

    No overlays -- each heatmap is rendered independently with the
    sequential colormap and per-image min-max normalisation, upsampled
    bilinearly to the RGB resolution so all three columns share the same
    aspect ratio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows_data)
    if n == 0:
        raise ValueError("No rows to plot")

    fig_h = 2.6 * n + 0.4
    fig, axes = plt.subplots(
        n, 3, figsize=(12, fig_h), dpi=140, squeeze=False,
    )
    col_titles = [
        "lamo-generated RGB frame",
        "BMV-projection heatmap   (|Delta . BMV| / ||BMV||)",
        "Predictor heatmap        (|| predictor(z, pooled) ||)",
    ]

    for r_idx, row in enumerate(rows_data):
        rgb = row["rgb"]
        H_rgb, W_rgb = rgb.shape[:2]

        ax0 = axes[r_idx][0]
        ax0.imshow(rgb)
        ax0.set_xticks([])
        ax0.set_yticks([])
        cap = row.get("caption", "")
        if len(cap) > 50:
            cap = cap[:50] + "..."
        ax0.set_ylabel(
            f"#{r_idx + 1} (t={row['latent_t']})\n{cap}",
            rotation=0, ha="right", va="center",
            fontsize=8, labelpad=18,
        )
        if r_idx == 0:
            ax0.set_title(col_titles[0], fontsize=10)

        for col_idx, key in enumerate(["bmv_map", "pred_map"], start=1):
            heat = row[key]
            heat_up = _upsample_to(heat, H_rgb, W_rgb)
            heat_norm = _safe_normalize_map(heat_up)
            ax = axes[r_idx][col_idx]
            ax.imshow(heat_norm, cmap=cmap_name, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10)

    suptitle = (
        "lamo interpretability -- one generation per row, "
        "two heatmaps share the same latent t* (most-motion pair)"
    )
    if title_suffix:
        suptitle = suptitle + "\n" + title_suffix
    fig.suptitle(suptitle, fontsize=11, y=0.998)
    fig.tight_layout(rect=(0.07, 0.0, 1.0, 0.985))
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined grid -> {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Pearson helper
# ---------------------------------------------------------------------------


def _spatial_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 2D maps, treating each spatial cell as a sample."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-12
    return float((a * b).sum() / denom)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified lamo interpretability: BMV + motion-guidance "
                    "heatmaps on the same lamo-generated video.",
    )
    add_common_args(parser)

    parser.add_argument("--bmv_tau", type=int, default=2,
                        help="Temporal lag tau for BMV (training default 2).")

    parser.add_argument("--cfg_scale", type=float, default=6.0,
                        help="Classifier-free guidance scale (prompt CFG).")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--negative_prompt", type=str,
                        default="low quality, worst quality, blurry")

    parser.add_argument(
        "--guidance_lambda", type=float, default=0.0,
        help="Inference-time motion guidance strength. 0 (default) keeps "
             "the BMV / predictor analyses non-tautological -- both heatmaps "
             "probe the lamo transformer + LoRA on its own. Set > 0 to also "
             "include guidance (production setting).",
    )
    parser.add_argument("--guidance_step_ratio", type=float, default=0.85)

    parser.add_argument("--cmap", type=str, default="inferno",
                        help="Sequential colormap for both heatmaps.")

    parser.add_argument(
        "--predictor_metric", type=str, default="resid",
        choices=["resid", "proj", "norm"],
        help="How to collapse predictor's 16-D output to a scalar per "
             "spatial position. 'resid' (default) subtracts predictor's "
             "response on a temporally-averaged baseline latent (which "
             "has motion averaged out), giving ||predictor(z[t]) - "
             "predictor(z_static)||; this cancels the CNN noise floor "
             "and isolates motion-specific response. 'proj' projects "
             "onto the predictor's own BMV direction (still dense). "
             "'norm' is the plain L2 norm (noisiest).",
    )

    # ---- single-sample / video options -------------------------------------
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt string. When set, --prompt_csv / --num_samples "
             "/ --shuffle are ignored and only this one prompt is run. "
             "Generation seed is then exactly --seed (no per-sample offset), "
             "so passing the same seed reproduces a previous run.",
    )
    parser.add_argument(
        "--save_full_video", action="store_true",
        help="Also save the full generated video as 'generated.mp4' inside "
             "each per-sample folder (8 fps, decoded with the lamo pipe's "
             "VAE). Auto-enabled when --prompt is used.",
    )
    parser.add_argument(
        "--video_fps", type=int, default=8,
        help="FPS for saved generated.mp4 (default 8, matches CogVideoX).",
    )

    args = parser.parse_args()

    if args.lora_path is None:
        raise SystemExit("--lora_path is required (we need lamo weights).")
    if args.predictor_ckpt is None:
        raise SystemExit("--predictor_ckpt is required (predictor heatmap).")

    dtype = parse_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)

    print("=" * 72, flush=True)
    print(f"  Unified lamo interpretability  --  {args.num_samples} videos",
          flush=True)
    print(f"  guidance_lambda = {args.guidance_lambda} "
          f"({'OFF' if args.guidance_lambda <= 0 else 'ON'})  |  "
          f"bmv_tau = {args.bmv_tau}", flush=True)
    print("=" * 72, flush=True)

    # --- Pipeline + LoRA + predictor ---
    pipe, predictor, lora_scale, lora_name = build_lamo_pipeline_and_predictor(
        model_path=args.model_path,
        lora_path=args.lora_path,
        predictor_ckpt=args.predictor_ckpt,
        lora_weight_name=args.lora_weight_name,
        lora_name=args.lora_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_scale=args.lora_scale,
        predictor_hidden_channels=args.predictor_hidden_channels,
        predictor_num_res_blocks=args.predictor_num_res_blocks,
        predictor_use_se=bool(args.predictor_use_se),
        predictor_use_prev_delta=bool(args.predictor_use_prev_delta),
        predictor_use_prompt_cond=bool(args.predictor_use_prompt_cond),
        prompt_text_dim=args.prompt_text_dim,
        prompt_cond_dim=args.prompt_cond_dim,
        device=device, dtype=dtype,
    )
    if predictor is None:
        raise SystemExit("Predictor failed to load.")

    pipe.enable_lora()
    pipe.set_adapters([lora_name], [lora_scale])

    if args.guidance_lambda > 0:
        pipe.set_temporal_predictor(
            predictor,
            guidance_lambda=args.guidance_lambda,
            guidance_step_ratio=args.guidance_step_ratio,
        )

    # --- Captions ---
    if args.prompt is not None:
        # Single-sample mode: skip CSV entirely, the per-sample seed offset
        # is dropped so torch_seed == args.seed (exact reproducibility of a
        # previously-saved sample by passing the same seed).
        captions = [args.prompt]
        per_sample_seed_offset = False
        save_video_default = True
        print(f"  Single-prompt mode (seed={args.seed}): "
              f"{args.prompt[:90]}", flush=True)
    else:
        if not args.prompt_csv:
            raise ValueError("Either --prompt or --prompt_csv is required.")
        rows = load_caption_csv(
            args.prompt_csv, args.num_samples, args.seed,
            shuffle=args.shuffle,
        )
        captions = [str(r["caption"]) for _, r in rows.iterrows()]
        per_sample_seed_offset = True
        save_video_default = False
        mode = "shuffled" if args.shuffle else "in-order"
        print(f"  Loaded {len(captions)} captions ({mode}, seed={args.seed}) "
              f"from {args.prompt_csv}", flush=True)

    save_video = args.save_full_video or save_video_default

    rows_data = []

    for idx, caption in enumerate(captions):
        slug = slugify(caption)
        sample_dir = os.path.join(
            args.work_dir, f"{idx + 1:03d}_{slug}",
        )
        print(f"\n  [{idx + 1}/{len(captions)}] {slug}", flush=True)
        print(f"      caption: {caption[:90]}", flush=True)

        # --- 1. Generate ---
        torch_seed = args.seed + idx if per_sample_seed_offset else args.seed
        gen = torch.Generator("cpu").manual_seed(torch_seed)
        with torch.no_grad():
            out = pipe(
                prompt=caption,
                negative_prompt=args.negative_prompt,
                height=args.height, width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                use_dynamic_cfg=True,
                generator=gen,
                output_type="latent",
            )
            latents = out.frames                              # [1, T, 16, H_lat, W_lat]
            rgb_video = pipe.decode_latents(latents)          # [1, 3, T_rgb, H, W]

        # --- 2. Pick best t* by BMV norm ---
        bmv_norms = compute_bmv_norms_per_pair(latents.float(), args.bmv_tau)
        best_t = int(np.argmax(bmv_norms))

        # --- 3. BMV map at best_t ---
        bmv_map, bmv_vec, delta_norm_map, bmv_norm = compute_bmv_response_at(
            latents.float(), best_t, args.bmv_tau,
        )

        # --- 4. Predictor map at the same best_t ---
        _, prompt_pooled = encode_prompt_full_and_pooled(pipe, caption)
        pred_param = next(predictor.parameters())
        latents_for_pred = latents.to(
            device=pred_param.device, dtype=pred_param.dtype,
        )
        prompt_pooled = prompt_pooled.to(
            device=pred_param.device, dtype=pred_param.dtype,
        )
        pred_map = compute_predictor_response_at(
            predictor, latents_for_pred, best_t, prompt_pooled,
            metric=args.predictor_metric,
        )

        # --- 5. RGB frame at best_t ---
        T_lat = latents.shape[1]
        T_rgb = rgb_video.shape[2]
        rgb_idx = _latent_t_to_rgb_index(best_t, T_lat, T_rgb)
        rgb_frame = rgb_video[0, :, rgb_idx, :, :]
        rgb_np = (
            ((rgb_frame.float().clamp(-1, 1) + 1.0) * 127.5)
            .byte().permute(1, 2, 0).cpu().numpy()
        )

        # --- 6. Diagnostics: spatial Pearson(BMV, predictor) ---
        pearson_bmv_pred = _spatial_pearson(bmv_map, pred_map)
        pearson_bmv_motion = _spatial_pearson(bmv_map, delta_norm_map)
        print(
            f"      latent_t={best_t} (rgb_t={rgb_idx})  "
            f"||BMV||={bmv_norm:.4f}  "
            f"r(BMV, predictor)={pearson_bmv_pred:+.3f}  "
            f"r(BMV, ||Delta||)={pearson_bmv_motion:+.3f}",
            flush=True,
        )

        # --- 7. Save per-sample images ---
        info = dict(
            caption=caption,
            latent_t=best_t,
            rgb_t=rgb_idx,
            bmv_norm=bmv_norm,
            pearson_bmv_pred=pearson_bmv_pred,
            pearson_bmv_motion=pearson_bmv_motion,
        )
        save_sample_outputs(
            sample_dir, rgb_np, bmv_map, pred_map, info,
            cmap_name=args.cmap,
        )
        if save_video:
            video_path = os.path.join(sample_dir, "generated.mp4")
            _save_full_video_mp4(video_path, rgb_video, fps=args.video_fps)
            print(f"      saved -> {sample_dir}  (+generated.mp4)",
                  flush=True)
        else:
            print(f"      saved -> {sample_dir}", flush=True)

        rows_data.append({
            "rgb": rgb_np,
            "bmv_map": bmv_map,
            "pred_map": pred_map,
            "delta_norm_map": delta_norm_map,
            "bmv_vec": bmv_vec,
            "bmv_norm": bmv_norm,
            "caption": caption,
            "latent_t": best_t,
            "rgb_t": rgb_idx,
            "pearson_bmv_pred": pearson_bmv_pred,
            "pearson_bmv_motion": pearson_bmv_motion,
        })

        del out, latents, rgb_video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows_data:
        raise SystemExit("No videos generated -- nothing to plot.")

    # --- Combined grid ---
    pearsons_bp = np.array([r["pearson_bmv_pred"]   for r in rows_data])
    pearsons_bm = np.array([r["pearson_bmv_motion"] for r in rows_data])
    title_suffix = (
        f"n={len(rows_data)}, tau={args.bmv_tau}, "
        f"guidance={'on (lambda=%g)' % args.guidance_lambda if args.guidance_lambda > 0 else 'off'}\n"
        f"mean Pearson(R_bmv, R_predictor) = {pearsons_bp.mean():+.3f}  |  "
        f"mean Pearson(R_bmv, ||Delta||)    = {pearsons_bm.mean():+.3f}"
    )
    plot_combined_grid(
        rows_data, os.path.join(args.work_dir, "combined_grid.png"),
        cmap_name=args.cmap,
        title_suffix=title_suffix,
    )

    # --- NPZ ---
    np.savez(
        os.path.join(args.work_dir, "arrays.npz"),
        rgb=np.stack([r["rgb"] for r in rows_data], axis=0),
        bmv_map=np.stack([r["bmv_map"] for r in rows_data], axis=0),
        pred_map=np.stack([r["pred_map"] for r in rows_data], axis=0),
        delta_norm_map=np.stack([r["delta_norm_map"] for r in rows_data], axis=0),
        bmv_vec=np.stack([r["bmv_vec"] for r in rows_data], axis=0),
        bmv_norm=np.array([r["bmv_norm"] for r in rows_data], dtype=np.float32),
        latent_t=np.array([r["latent_t"] for r in rows_data], dtype=np.int64),
        rgb_t=np.array([r["rgb_t"] for r in rows_data], dtype=np.int64),
        pearson_bmv_pred=pearsons_bp.astype(np.float32),
        pearson_bmv_motion=pearsons_bm.astype(np.float32),
        captions=np.array([r["caption"] for r in rows_data], dtype=object),
    )
    print(f"\n  Done. Per-sample folders + grid + arrays under {args.work_dir}",
          flush=True)


if __name__ == "__main__":
    main()
