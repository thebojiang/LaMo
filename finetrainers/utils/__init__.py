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

import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .activation_checkpoint import apply_activation_checkpointing
from .data import determine_batch_size, should_perform_precomputation
from .diffusion import (
    _enable_vae_memory_optimizations,
    default_flow_shift,
    get_scheduler_alphas,
    get_scheduler_sigmas,
    prepare_loss_weights,
    prepare_sigmas,
    prepare_target,
    resolution_dependent_timestep_flow_shift,
)
from .file import delete_files, find_files, string_to_filename
from .hub import save_model_card
from .memory import bytes_to_gigabytes, free_memory, get_memory_statistics, make_contiguous
from .model import resolve_component_cls
from .state_checkpoint import PTDCheckpointManager
from .torch import (
    align_device_and_dtype,
    clip_grad_norm_,
    enable_determinism,
    expand_tensor_dims,
    get_device_info,
    set_requires_grad,
    synchronize_device,
    unwrap_model,
)


def get_parameter_names(obj: Any, method_name: Optional[str] = None) -> Set[str]:
    if method_name is not None:
        obj = getattr(obj, method_name)
    return {name for name, _ in inspect.signature(obj).parameters.items()}


def get_non_null_items(
    x: Union[List[Any], Tuple[Any], Dict[str, Any]]
) -> Union[List[Any], Tuple[Any], Dict[str, Any]]:
    if isinstance(x, dict):
        return {k: v for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return type(x)(v for v in x if v is not None)
