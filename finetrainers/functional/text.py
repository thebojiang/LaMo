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

import random
from typing import List, Union

import torch


def dropout_caption(caption: Union[str, List[str]], dropout_p: float = 0) -> Union[str, List[str]]:
    if random.random() >= dropout_p:
        return caption
    if isinstance(caption, str):
        return ""
    return [""] * len(caption)


def dropout_embeddings_to_zero(embed: torch.Tensor, dropout_p: float = 0) -> torch.Tensor:
    if random.random() >= dropout_p:
        return embed
    embed = torch.zeros_like(embed)
    return embed


def remove_prefix(text: str, prefixes: List[str]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text.removeprefix(prefix).strip()
    return text
