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
from typing import Type

from .models import ModelSpecification
from .models.cogvideox import CogVideoXModelSpecification
from .models.cogvideox import LaMoCogVideoXModelSpecification


class ModelType(str, Enum):
    COGVIDEOX = "cogvideox"
    LAMO = "lamo"


class TrainingType(str, Enum):
    LORA = "lora"
    FULL_FINETUNE = "full-finetune"


SUPPORTED_MODEL_CONFIGS = {
    ModelType.COGVIDEOX: {
        TrainingType.LORA: CogVideoXModelSpecification,
        TrainingType.FULL_FINETUNE: CogVideoXModelSpecification,
    },
    ModelType.LAMO: {
        TrainingType.LORA: LaMoCogVideoXModelSpecification,
        TrainingType.FULL_FINETUNE: LaMoCogVideoXModelSpecification,
    },
}


def _get_model_specifiction_cls(model_name: str, training_type: str) -> Type[ModelSpecification]:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if training_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Training type {training_type} not supported for model {model_name}. "
            f"Supported training types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][training_type]
