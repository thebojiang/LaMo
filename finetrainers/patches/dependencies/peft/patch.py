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

import functools

from peft.tuners.tuners_utils import BaseTunerLayer

from ...utils import DisableTensorToDtype


def patch_peft_move_adapter_to_device_of_base_layer() -> None:
    _perform_patch_move_adapter_to_device_of_base_layer()


def _perform_patch_move_adapter_to_device_of_base_layer() -> None:
    BaseTunerLayer._move_adapter_to_device_of_base_layer = _patched_move_adapter_to_device_of_base_layer(
        BaseTunerLayer._move_adapter_to_device_of_base_layer
    )


def _patched_move_adapter_to_device_of_base_layer(func) -> None:
    # TODO(aryan): This is really unsafe probably and may break things. It works for now, but revisit and refactor.
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with DisableTensorToDtype():
            return func(self, *args, **kwargs)

    return wrapper
