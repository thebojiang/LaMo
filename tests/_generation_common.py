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

import json
import os
from typing import Optional, Union

import imageio
import numpy as np
import torch


RESOLUTION_MAP = {
    "cogvideox": (720, 480),
    "lamo": (720, 480),
}


def load_prompt_json(json_path, dtype=None, generate_type=None):
    """Load a simple prompt JSON file.

    Supported formats:
      - [{"caption": "..."}]
      - [{"en_caption": "..."}]
      - ["..."]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    prompts = []
    for row in rows:
        if isinstance(row, str):
            prompts.append(row)
        elif isinstance(row, dict):
            prompts.append(row.get("caption") or row.get("en_caption") or row.get("prompt") or "")
    return [p for p in prompts if p]


def _load_lamo_predictor(
    lora_path: Optional[str],
    physical_module_path: Optional[str],
    predictor_hidden_channels: int,
    predictor_num_res_blocks: int,
    predictor_use_se: bool,
    predictor_use_prev_delta: bool,
    predictor_use_prompt_cond: bool,
    prompt_text_dim: int,
    prompt_cond_dim: int,
):
    from finetrainers.models.wan.physical_motion_lamo import (
        TemporalConsistencyPredictor,
    )

    predictor = TemporalConsistencyPredictor(
        in_channels=16,
        hidden_channels=predictor_hidden_channels,
        num_res_blocks=predictor_num_res_blocks,
        use_se=predictor_use_se,
        use_prev_delta=predictor_use_prev_delta,
        use_prompt_cond=predictor_use_prompt_cond,
        prompt_text_dim=prompt_text_dim,
        prompt_cond_dim=prompt_cond_dim,
    )
    predictor_path = physical_module_path
    if predictor_path is None and lora_path:
        lora_dir = os.path.dirname(lora_path) if os.path.isfile(lora_path) else lora_path
        predictor_path = os.path.join(lora_dir, "predictor.safetensors")
    if not predictor_path or not os.path.isfile(predictor_path):
        raise FileNotFoundError(
            "LaMo generation requires predictor.safetensors. Set "
            "--physical_module_path or place predictor.safetensors next to the LoRA weights."
        )

    from safetensors.torch import load_file

    state = load_file(predictor_path)
    weights = {
        k[len("physical_lamo."):]: v
        for k, v in state.items()
        if k.startswith("physical_lamo.")
    }
    if not weights:
        weights = {
            k: v for k, v in state.items()
            if not k.startswith("physical_global_proj.")
        }
    if not weights:
        raise ValueError(
            f"No LaMo predictor weights found in {predictor_path}. Expected "
            "physical_lamo.* keys or unprefixed predictor keys."
        )
    missing, unexpected = predictor.load_state_dict(weights, strict=False)
    if len(missing) == len(predictor.state_dict()):
        raise ValueError(
            f"Predictor weights in {predictor_path} did not match the expected "
            "LaMo predictor architecture. If this is an older checkpoint, run "
            "eval/scripts/normalize_lamo_predictor_keys.py first."
        )
    if missing:
        print(f"Predictor load warning: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"Predictor load warning: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    print(f"Loaded prompt-conditioned temporal predictor from {predictor_path}")
    return predictor


def generate_video(
    model_path: str,
    prompts: list[str, dict],
    lora_path: Optional[str] = None,
    lora_name: str = "lora_adapter",
    lora_rank: int = 128,
    lora_alpha: int = 64,
    num_frames: int = 49,
    output_file: str = "./outputs/sample/",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    phys_guidance_scale: float = 6.0,
    generate_type: str = "lora",
    model_type: str = "lamo",
    fps: int = 16,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    start_index: int = 0,
    device: Optional[Union[int, torch.device]] = None,
    physical_module_path: Optional[str] = None,
    physical_latent_dim: Optional[int] = None,
    physical_motion_dim: int = 16,
    physical_token_dim: Optional[int] = None,
    physical_codebook_size: Optional[int] = None,
    physical_encoder_K: Optional[int] = None,
    physical_decoder_base_spatial: Optional[tuple] = None,
    physical_decoder_upsample_layers: Optional[int] = None,
    guidance_lambda: float = 15.0,
    guidance_step_ratio: float = 0.8,
    predictor_hidden_channels: int = 256,
    predictor_num_res_blocks: int = 8,
    predictor_use_se: bool = True,
    predictor_use_prev_delta: bool = False,
    predictor_use_prompt_cond: bool = True,
    prompt_text_dim: int = 4096,
    prompt_cond_dim: int = 128,
    per_prompt_seeds: Optional[list] = None,
    skip_existing: bool = False,
):
    if model_type not in RESOLUTION_MAP:
        raise ValueError(f"Unsupported model_type={model_type!r}. Use 'lamo' or 'cogvideox'.")
    if generate_type not in ("baseline", "lora"):
        raise ValueError("Only generate_type='baseline' or 'lora' is supported in the open-source build.")
    if per_prompt_seeds is not None and len(per_prompt_seeds) != len(prompts):
        raise ValueError(
            f"per_prompt_seeds length {len(per_prompt_seeds)} != prompts length {len(prompts)}"
        )

    os.makedirs(output_file, exist_ok=True)
    width, height = RESOLUTION_MAP[model_type]
    if device is not None:
        device_obj = torch.device(f"cuda:{device}") if isinstance(device, int) else device
    else:
        device_obj = torch.device("cuda")

    if model_type == "lamo":
        from finetrainers.models.cogvideox.pipeline_cogvideox_lamo import (
            CogVideoXPipelineLaMo,
        )

        pipe_cls = CogVideoXPipelineLaMo
    else:
        from diffusers import CogVideoXPipeline

        pipe_cls = CogVideoXPipeline

    pipe = pipe_cls.from_pretrained(model_path, torch_dtype=dtype).to(device_obj)

    if model_type == "lamo":
        predictor = _load_lamo_predictor(
            lora_path=lora_path,
            physical_module_path=physical_module_path,
            predictor_hidden_channels=predictor_hidden_channels,
            predictor_num_res_blocks=predictor_num_res_blocks,
            predictor_use_se=predictor_use_se,
            predictor_use_prev_delta=predictor_use_prev_delta,
            predictor_use_prompt_cond=predictor_use_prompt_cond,
            prompt_text_dim=prompt_text_dim,
            prompt_cond_dim=prompt_cond_dim,
        )
        pipe.set_temporal_predictor(
            predictor,
            guidance_lambda=guidance_lambda,
            guidance_step_ratio=guidance_step_ratio,
        )

    if lora_path and generate_type == "lora":
        lora_dir = lora_path if os.path.isdir(lora_path) else os.path.dirname(lora_path)
        pipe.load_lora_weights(
            lora_dir,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name=lora_name,
        )
        pipe.set_adapters([lora_name], [lora_alpha / lora_rank])
        print(f"LoRA loaded: {lora_dir} (adapter={lora_name}, scale={lora_alpha / lora_rank})")

    from diffusers import CogVideoXDPMScheduler

    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(device=device_obj)

    generation_kwargs = {"use_dynamic_cfg": True}
    for index, prompt in enumerate(prompts):
        global_idx = start_index + index
        text_prompt = prompt.get("caption", "") if isinstance(prompt, dict) else str(prompt)
        video_path = os.path.join(output_file, f"video_{global_idx}.mp4")
        if skip_existing and os.path.isfile(video_path) and os.path.getsize(video_path) > 0:
            print(f"Skipping existing generated video at {video_path} (idx {global_idx})")
            continue
        cur_seed = per_prompt_seeds[index] if per_prompt_seeds is not None else seed
        video = pipe(
            prompt=text_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(cur_seed),
            **generation_kwargs,
        ).frames[0]
        frames = [np.array(img) for img in video]
        imageio.mimsave(video_path, frames, fps=fps, codec="libx264")
        print(f"Saved generated video at {video_path} (idx {global_idx})")
