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

from typing import List, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast

from .base import ProcessorMixin


class T5Processor(ProcessorMixin):
    r"""
    Processor for the T5 family of models. This processor is used to encode text inputs and return the embeddings
    and attention masks for the input text.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor should return. The first output is the embeddings of the input
            text and the second output is the attention mask for the input text.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()

        self.output_names = output_names

        assert len(self.output_names) == 2

    def forward(
        self,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        text_encoder: T5EncoderModel,
        caption: Union[str, List[str]],
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encode the input text and return the embeddings and attention mask for the input text.

        Args:
            tokenizer (`Union[T5Tokenizer, T5TokenizerFast]`):
                The tokenizer used to tokenize the input text.
            text_encoder (`T5EncoderModel`):
                The text encoder used to encode the input text.
            caption (`Union[str, List[str]]`):
                The input text to be encoded.
            max_sequence_length (`int`):
                The maximum sequence length of the input text.
        """
        if isinstance(caption, str):
            caption = [caption]

        device = text_encoder.device
        dtype = text_encoder.dtype

        batch_size = len(caption)
        text_inputs = tokenizer(
            caption,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

        return {
            self.output_names[0]: prompt_embeds,
            self.output_names[1]: prompt_attention_mask,
        }
