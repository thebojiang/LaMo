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
Shared utilities for lamo interpretability scripts.

Two scripts in this directory consume these helpers:

    interpret_bmv_distribution.py     - one figure: 2D BMV scatter (gt / baseline / lamo)
    interpret_motion_guidance.py      - one figure: spatial heatmap grid

Common responsibilities:
    * Sample (caption, video) pairs from a local OpenVid CSV.
    * Load each local video into memory.
    * Resize to the lamo training bucket, encode through the VAE, and apply
      the same VAE scaling factor used during training (so latents live in
      the same space the transformer / BMV loss / predictor were trained on).
    * Build a CogVideoXPipelineLaMo, attach LoRA (kept disabled by
      default), and return the lamo prompt-conditioned predictor separately.
    * Encode prompts to (encoder_hidden_states, mean-pooled vector).

Nothing in this module produces a figure or a metric on its own.
"""

import os
import sys
from typing import Optional, Tuple

import torch
import pandas as pd

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)


# ---------------------------------------------------------------------------
# OpenVid CSV / video loading
# ---------------------------------------------------------------------------


def download_openvid_csv(local_path: str, work_dir: str) -> str:
    """Return a local CSV path."""
    if not os.path.isabs(local_path):
        local_path = os.path.join(_WORKSPACE, local_path)
    return local_path


def sample_openvid_rows(
    csv_path: str, num_samples: int, seed: int,
) -> pd.DataFrame:
    """Sample ``num_samples`` rows from the OpenVid CSV (columns: video, caption)."""
    df = pd.read_csv(csv_path).dropna(subset=["caption", "video"])
    n = min(num_samples, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def load_caption_csv(
    csv_path: str, num_samples: int, seed: int, shuffle: bool = True,
) -> pd.DataFrame:
    """Read a local CSV that has at least a ``caption`` column.

    Resolves a relative path against the workspace root, mirroring the
    behaviour of ``tests/phase2b_test_guidance.py``. Use this for
    prompt-only analyses (no GT videos required).
    """
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(_WORKSPACE, csv_path)
    df = pd.read_csv(csv_path).dropna(subset=["caption"])
    n = min(num_samples, len(df))
    if shuffle:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df = df.head(n).reset_index(drop=True)
    return df


def resolve_video_uri(videos_root: str, video_filename: str) -> str:
    """Join the local OpenVid videos directory with a filename row."""
    return os.path.join(videos_root, video_filename)


def load_video_tensor(
    video_uri: str, num_frames: int, height: int, width: int,
) -> Optional[torch.Tensor]:
    """Load a video → ``[T, C, H, W]`` tensor in [-1, 1].

    Resize / temporal-bucket via the same helper used by training so the
    encoded latents match the BMV loss / predictor input distribution.
    Returns ``None`` if the video cannot be decoded.
    """
    from finetrainers import functional as FF

    try:
        try:
            from decord import VideoReader, cpu
            reader = VideoReader(video_uri, ctx=cpu(0))
            total = len(reader)
            frames = reader.get_batch(list(range(total))).asnumpy()
        except Exception:
            import imageio.v3 as iio
            frames = list(iio.imiter(video_uri))
            total = len(frames)
    except Exception as e:
        print(f"  [skip] failed to load {video_uri}: {e}", flush=True)
        return None

    if total < 2:
        return None

    if not torch.is_tensor(frames):
        try:
            frames = torch.from_numpy(frames)
        except TypeError:
            import numpy as np
            frames = torch.from_numpy(np.array(frames))

    frames = frames.permute(0, 3, 1, 2).contiguous().float() / 127.5 - 1.0
    bucket = (num_frames, height, width)
    frames, _ = FF.resize_to_nearest_bucket_video(
        frames, [bucket], resize_mode="bicubic",
    )
    return frames


# ---------------------------------------------------------------------------
# Pipeline + LoRA + predictor
# ---------------------------------------------------------------------------


def _load_lamo_predictor(
    ckpt_path: str,
    hidden_channels: int,
    num_res_blocks: int,
    use_se: bool,
    use_prev_delta: bool,
    use_prompt_cond: bool,
    prompt_text_dim: int,
    prompt_cond_dim: int,
    device: torch.device,
    dtype: torch.dtype,
):
    from finetrainers.models.wan.physical_motion_lamo import (
        TemporalConsistencyPredictor,
    )
    from safetensors.torch import load_file

    predictor = TemporalConsistencyPredictor(
        in_channels=16,
        hidden_channels=hidden_channels,
        num_res_blocks=num_res_blocks,
        use_se=use_se,
        use_prev_delta=use_prev_delta,
        use_prompt_cond=use_prompt_cond,
        prompt_text_dim=prompt_text_dim,
        prompt_cond_dim=prompt_cond_dim,
    )
    state = load_file(ckpt_path)
    phys = {
        k[len("physical_lamo."):]: v
        for k, v in state.items() if k.startswith("physical_lamo.")
    }
    if not phys:
        phys = state
    missing, unexpected = predictor.load_state_dict(phys, strict=False)
    if missing:
        print(f"  [predictor] {len(missing)} missing keys (first 3): {missing[:3]}",
              flush=True)
    if unexpected:
        print(f"  [predictor] {len(unexpected)} unexpected keys (first 3): "
              f"{unexpected[:3]}", flush=True)
    return predictor.to(device=device, dtype=dtype).eval()


def build_lamo_pipeline_and_predictor(
    model_path: str,
    lora_path: str,
    predictor_ckpt: str,
    lora_weight_name: str,
    lora_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_scale: Optional[float],
    predictor_hidden_channels: int,
    predictor_num_res_blocks: int,
    predictor_use_se: bool,
    predictor_use_prev_delta: bool,
    predictor_use_prompt_cond: bool,
    prompt_text_dim: int,
    prompt_cond_dim: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Returns ``(pipe, predictor, lora_scale, lora_name)``.

    LoRA is loaded but disabled by default so callers explicitly toggle
    baseline (LoRA off) and lamo (LoRA on) paths. The predictor is
    returned separately and never attached to the pipeline by this
    function — both interpretability scripts run their own loops outside
    of ``pipe.__call__``.
    """
    from diffusers import CogVideoXDPMScheduler
    from finetrainers.models.cogvideox.pipeline_cogvideox_lamo import (
        CogVideoXPipelineLaMo,
    )

    print(f"  Loading CogVideoX pipeline from {model_path} ...", flush=True)
    pipe = CogVideoXPipelineLaMo.from_pretrained(
        model_path, torch_dtype=dtype,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
    )
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if lora_path is not None:
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path) or "."
            weight_name = os.path.basename(lora_path)
        else:
            lora_dir = lora_path
            weight_name = lora_weight_name
        scale = lora_scale if lora_scale is not None else lora_alpha / lora_rank
        print(f"  Loading LoRA from {lora_dir} (weight={weight_name}, scale={scale:.4f}) ...",
              flush=True)
        pipe.load_lora_weights(
            lora_dir, weight_name=weight_name, adapter_name=lora_name,
        )
        pipe.set_adapters([lora_name], [scale])
        pipe.disable_lora()
    else:
        scale = 0.0

    predictor = None
    if predictor_ckpt is not None:
        print(f"  Loading lamo predictor from {predictor_ckpt} ...", flush=True)
        predictor = _load_lamo_predictor(
            predictor_ckpt,
            hidden_channels=predictor_hidden_channels,
            num_res_blocks=predictor_num_res_blocks,
            use_se=predictor_use_se,
            use_prev_delta=predictor_use_prev_delta,
            use_prompt_cond=predictor_use_prompt_cond,
            prompt_text_dim=prompt_text_dim,
            prompt_cond_dim=prompt_cond_dim,
            device=device, dtype=dtype,
        )

    return pipe, predictor, scale, lora_name


# ---------------------------------------------------------------------------
# VAE / prompt encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_video_to_latents(pipe, video_tensor: torch.Tensor) -> torch.Tensor:
    """``[T, C, H, W]`` in [-1, 1] → ``[1, T_lat, 16, H_lat, W_lat]`` (training-scaled).

    Mirrors ``CogVideoXLatentEncodeProcessor`` in ``base_specification.py``:
    permute to ``[B, C, T, H, W]`` for the VAE, sample from the latent
    distribution, permute back to ``[B, T_lat, C, H, W]``, and apply
    ``vae.config.scaling_factor`` (``invert_scale_latents`` is False for the
    standard CogVideoX-5B VAE).
    """
    vae = pipe.vae
    video = video_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
    video = video.permute(0, 2, 1, 3, 4).contiguous()
    z = vae.encode(video).latent_dist.sample()
    z = z.permute(0, 2, 1, 3, 4)
    if not getattr(vae.config, "invert_scale_latents", False):
        z = z * vae.config.scaling_factor
    return z


@torch.no_grad()
def encode_prompt_full_and_pooled(
    pipe, prompt: str, max_sequence_length: int = 226,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(prompt_embeds [1, L, D], prompt_pooled [1, D])``.

    Pool method is mean-over-tokens (canonical for T5; matches lamo training
    and pipeline). No CFG / negative prompt is involved here — both scripts
    work in single-step / no-CFG modes by design.
    """
    from finetrainers.models.wan.physical_motion_lamo import pool_prompt_embeds

    device = pipe._execution_device
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    prompt_pooled = pool_prompt_embeds(prompt_embeds)
    return prompt_embeds, prompt_pooled


# ---------------------------------------------------------------------------
# CLI argument helpers (shared knobs)
# ---------------------------------------------------------------------------


def add_common_args(parser):
    """Add the model / LoRA / predictor / data args used by both scripts."""
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to CogVideoX-5b-Diffusers")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="lamo pytorch_lora_weights.safetensors (file or dir). "
                             "Required for the BMV script; optional for motion guidance.")
    parser.add_argument("--predictor_ckpt", type=str, default=None,
                        help="lamo predictor.safetensors. "
                             "Required for motion guidance script.")
    parser.add_argument("--lora_weight_name", type=str,
                        default="pytorch_lora_weights.safetensors")
    parser.add_argument("--lora_name", type=str, default="lora_adapter")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_scale", type=float, default=None,
                        help="Override LoRA scale (defaults to alpha/rank)")

    parser.add_argument("--predictor_hidden_channels", type=int, default=256)
    parser.add_argument("--predictor_num_res_blocks", type=int, default=8)
    parser.add_argument("--predictor_use_se", type=int, default=1)
    parser.add_argument("--predictor_use_prev_delta", type=int, default=0)
    parser.add_argument("--predictor_use_prompt_cond", type=int, default=1)
    parser.add_argument("--prompt_text_dim", type=int, default=4096)
    parser.add_argument("--prompt_cond_dim", type=int, default=128)

    parser.add_argument(
        "--prompt_csv", type=str,
        default=None,
        help="Local CSV with a 'caption' column (path relative to the "
             "workspace root or absolute).",
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Randomly sample prompts using --seed (default). Use "
             "--no_shuffle for deterministic head-N ordering.",
    )
    parser.add_argument(
        "--no_shuffle", dest="shuffle", action="store_false",
        help="Disable random sampling -- iterate the CSV in order.",
    )
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of prompts to draw (configurable).")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--work_dir", type=str, default="outputs/interpret_lamo")


def parse_dtype(s: str) -> torch.dtype:
    return {"float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32}[s]
