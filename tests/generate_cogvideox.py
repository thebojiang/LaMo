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
    parser = argparse.ArgumentParser(description="Generate videos with CogVideoX.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="<PATH_TO_COGVIDEOX_BASE_MODEL>")
    parser.add_argument("--output_file", type=str, default="./outputs/cogvideox/")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
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
        num_frames=args.num_frames,
        output_file=args.output_file,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generate_type="baseline",
        model_type="cogvideox",
        fps=args.fps,
        seed=args.seed,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
