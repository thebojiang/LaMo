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

from enum import Enum
from typing import Any

from .accelerate import AccelerateParallelBackend
from .utils import apply_ddp_ptd, apply_fsdp2_ptd, dist_max, dist_mean


ParallelBackendType = Any


class ParallelBackendEnum(str, Enum):
    ACCELERATE = "accelerate"
    PTD = "ptd"


def get_parallel_backend_cls(backend: ParallelBackendEnum) -> ParallelBackendType:
    if backend == ParallelBackendEnum.ACCELERATE:
        return AccelerateParallelBackend
    if backend == ParallelBackendEnum.PTD:
        from .ptd import PytorchDTensorParallelBackend

        return PytorchDTensorParallelBackend
    raise ValueError(f"Unknown parallel backend: {backend}")
