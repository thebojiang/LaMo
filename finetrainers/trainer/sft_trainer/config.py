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

import argparse
from typing import TYPE_CHECKING, List, Union

from ..config_utils import ConfigMixin


if TYPE_CHECKING:
    from ...args import BaseArgs


class SFTLowRankConfig(ConfigMixin):
    r"""
    Configuration class for SFT low rank training.

    Args:
        rank (int):
            Rank of the low rank approximation.
        lora_alpha (int):
            The lora_alpha parameter to compute scaling factor (lora_alpha / rank) for low-rank matrices.
        target_modules (`str` or `List[str]`):
            Target modules for the low rank approximation. Can be a regex string or a list of regex strings.
    """

    rank: int = 64
    lora_alpha: int = 64
    # Wan uses "blocks"; Flux/SD3 use "transformer_blocks" or "single_transformer_blocks"
    target_modules: Union[str, List[str]] = "(transformer_blocks|single_transformer_blocks|blocks).*(to_q|to_k|to_v|to_out.0)"

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=["(transformer_blocks|single_transformer_blocks|blocks).*(to_q|to_k|to_v|to_out.0)"],
        )

    def validate_args(self, args: "BaseArgs"):
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.rank = argparse_args.rank
        mapped_args.lora_alpha = argparse_args.lora_alpha
        mapped_args.target_modules = (
            argparse_args.target_modules[0] if len(argparse_args.target_modules) == 1 else argparse_args.target_modules
        )


class SFTFullRankConfig(ConfigMixin):
    def add_args(self, parser: argparse.ArgumentParser):
        pass

    def validate_args(self, args: "BaseArgs"):
        pass

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        pass
