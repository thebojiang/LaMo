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

import torch


class DisableTensorToDtype:
    def __enter__(self):
        self.original_to = torch.Tensor.to

        def modified_to(tensor, *args, **kwargs):
            # remove dtype from args if present
            args = [arg if not isinstance(arg, torch.dtype) else None for arg in args]
            if "dtype" in kwargs:
                kwargs.pop("dtype")
            return self.original_to(tensor, *args, **kwargs)

        torch.Tensor.to = modified_to

    def __exit__(self, *args, **kwargs):
        torch.Tensor.to = self.original_to
