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
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..args import BaseArgs


class ConfigMixin:
    def add_args(self, parser: argparse.ArgumentParser):
        raise NotImplementedError("ConfigMixin::add_args should be implemented by subclasses.")

    def validate_args(self, args: "BaseArgs"):
        raise NotImplementedError("ConfigMixin::map_args should be implemented by subclasses.")

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        raise NotImplementedError("ConfigMixin::validate_args should be implemented by subclasses.")
