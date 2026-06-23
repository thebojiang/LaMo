# Copyright (c) 2026 Applied Intuition, Inc.
#
# This file is part of a modified version of finetrainers
# (https://github.com/huggingface/finetrainers), Copyright the
# finetrainers contributors, licensed under the Apache License, Version
# 2.0. Modifications by Applied Intuition, Inc.
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

import os
from typing import List, Union

import numpy as np
import wandb
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from PIL import Image


def save_model_card(
    args,
    repo_id: str,
    videos: Union[List[str], Union[List[Image.Image], List[np.ndarray]]],
    validation_prompts: List[str],
    fps: int = 30,
) -> None:
    widget_dict = []
    output_dir = str(args.output_dir)
    if videos is not None and len(videos) > 0:
        for i, (video, validation_prompt) in enumerate(zip(videos, validation_prompts)):
            if not isinstance(video, str):
                export_to_video(video, os.path.join(output_dir, f"final_video_{i}.mp4"), fps=fps)
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": video if isinstance(video, str) else f"final_video_{i}.mp4"},
                }
            )

    model_description = f"""
# LoRA Finetune

<Gallery />

## Model description

This is a lora finetune of model: `{args.pretrained_model_name_or_path}`.

The model was trained using [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
TODO
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.
"""
    if wandb.run.url:
        model_description += f"""
Find out the wandb run URL and training configurations [here]({wandb.run.url}).
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(args.output_dir, "README.md"))
