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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast

from .base import ProcessorMixin


class CLIPPooledProcessor(ProcessorMixin):
    r"""
    Processor for the Llama family of models. This processor is used to encode text inputs and return the embeddings
    and attention masks for the input text.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the embeddings of the input
            text and the second output is the attention mask for the input text.
    """

    def __init__(self, output_names: List[str] = None, input_names: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        self.output_names = output_names
        self.input_names = input_names

        assert len(output_names) == 1
        if input_names is not None:
            assert len(input_names) <= 3

    def forward(
        self,
        tokenizer: Union[CLIPTokenizer, CLIPTokenizerFast],
        text_encoder: CLIPTextModel,
        caption: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode the input text and return the embeddings and attention mask for the input text.

        Args:
            tokenizer (`Union[LlamaTokenizer, LlamaTokenizerFast]`):
                The tokenizer used to tokenize the input text.
            text_encoder (`LlamaModel`):
                The text encoder used to encode the input text.
            caption (`Union[str, List[str]]`):
                The input text to be encoded.
        """
        if isinstance(caption, str):
            caption = [caption]

        device = text_encoder.device
        dtype = text_encoder.dtype

        text_inputs = tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False).pooler_output
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return {self.output_names[0]: prompt_embeds}
