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
from typing import Any, Dict, List


class ProcessorMixin:
    def __init__(self) -> None:
        self._forward_parameter_names = inspect.signature(self.forward).parameters.keys()
        self.output_names: List[str] = None
        self.input_names: Dict[str, Any] = None

    def __call__(self, *args, **kwargs) -> Any:
        shallow_copy_kwargs = dict(kwargs.items())
        if self.input_names is not None:
            for k, v in self.input_names.items():
                shallow_copy_kwargs[v] = shallow_copy_kwargs.pop(k)
        acceptable_kwargs = {k: v for k, v in shallow_copy_kwargs.items() if k in self._forward_parameter_names}
        return self.forward(*args, **acceptable_kwargs)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("ProcessorMixin::forward method should be implemented by the subclass.")
