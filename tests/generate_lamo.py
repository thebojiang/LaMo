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

import argparse
import os

import torch

from _generation_common import generate_video, load_prompt_json


def main():
    parser = argparse.ArgumentParser(description="Generate videos with LaMo.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="<PATH_TO_COGVIDEOX_BASE_MODEL>")
    parser.add_argument("--lora_path", type=str, default="<PATH_TO_LORA_CHECKPOINT>")
    parser.add_argument("--physical_module_path", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="./outputs/lamo/")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--guidance_lambda", type=float, default=15.0)
    parser.add_argument("--guidance_step_ratio", type=float, default=0.8)
    parser.add_argument("--predictor_hidden_channels", type=int, default=256)
    parser.add_argument("--predictor_num_res_blocks", type=int, default=8)
    parser.add_argument("--predictor_use_se", type=int, default=1)
    parser.add_argument("--predictor_use_prev_delta", type=int, default=0)
    parser.add_argument("--predictor_use_prompt_cond", type=int, default=1)
    parser.add_argument("--prompt_text_dim", type=int, default=4096)
    parser.add_argument("--prompt_cond_dim", type=int, default=128)
    parser.add_argument("--lora_name", type=str, default="lora_adapter")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    prompts = load_prompt_json(args.prompt_path) if args.prompt_path and os.path.exists(args.prompt_path) else [args.prompt]
    if not any(prompts):
        raise ValueError("Set --prompt or --prompt_path.")

    generate_video(
        model_path=args.model_path,
        prompts=prompts,
        lora_path=args.lora_path,
        physical_module_path=args.physical_module_path,
        lora_name=args.lora_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_frames=args.num_frames,
        output_file=args.output_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generate_type="lora",
        model_type="lamo",
        fps=args.fps,
        seed=args.seed,
        dtype=dtype_map[args.dtype],
        guidance_lambda=args.guidance_lambda,
        guidance_step_ratio=args.guidance_step_ratio,
        predictor_hidden_channels=args.predictor_hidden_channels,
        predictor_num_res_blocks=args.predictor_num_res_blocks,
        predictor_use_se=bool(args.predictor_use_se),
        predictor_use_prev_delta=bool(args.predictor_use_prev_delta),
        predictor_use_prompt_cond=bool(args.predictor_use_prompt_cond),
        prompt_text_dim=args.prompt_text_dim,
        prompt_cond_dim=args.prompt_cond_dim,
    )


if __name__ == "__main__":
    main()
