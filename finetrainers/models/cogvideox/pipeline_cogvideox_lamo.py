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

# LaMo: CogVideoX inference pipeline with prompt-conditioned temporal-
# consistency guidance.
#
# LaMo prompt-conditioning:
#   - The temporal predictor is conditioned on the *positive* prompt, mean-
#     pooled to a single C-dim vector.  We therefore pre-compute the pooled
#     positive prompt embedding once (before the CFG concat), cache it, and
#     pass it into ``apply_classifier_guidance`` at every denoising step.
#
# Self-contained — does not import from any previous LaMo pipeline version.
# Only depends on ``physical_motion_lamo``.

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import CogVideoXPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.schedulers import CogVideoXDPMScheduler
from diffusers.utils import is_torch_xla_available

from ..wan.physical_motion_lamo import (
    TemporalConsistencyPredictor,
    apply_classifier_guidance,
    pool_prompt_embeds,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class CogVideoXPipelineLaMo(CogVideoXPipeline):
    """CogVideoX pipeline with prompt-conditioned temporal-consistency
    classifier guidance (lamo).

    Usage::

        pipe = CogVideoXPipelineLaMo.from_pretrained(...)
        pipe.set_temporal_predictor(
            predictor,
            guidance_lambda=15.0, guidance_step_ratio=0.8,
        )
        video = pipe(prompt="...", num_inference_steps=50).frames[0]

    Set ``guidance_lambda <= 0`` to disable guidance (returns to a plain
    CogVideoX inference path while keeping the predictor loaded).
    """

    def set_temporal_predictor(
        self,
        predictor: TemporalConsistencyPredictor,
        guidance_lambda: float = 15.0,
        guidance_step_ratio: float = 0.8,
    ) -> None:
        self._lamo_predictor = predictor
        self._lamo_guidance_lambda = guidance_lambda
        self._lamo_guidance_step_ratio = guidance_step_ratio

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        predictor = getattr(self, "_lamo_predictor", None)
        if predictor is None:
            return super().__call__(
                prompt=prompt, negative_prompt=negative_prompt,
                height=height, width=width, num_frames=num_frames,
                num_inference_steps=num_inference_steps, timesteps=timesteps,
                guidance_scale=guidance_scale, use_dynamic_cfg=use_dynamic_cfg,
                num_videos_per_prompt=num_videos_per_prompt, eta=eta,
                generator=generator, latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type, return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        guidance_lambda = self._lamo_guidance_lambda
        guidance_step_ratio = self._lamo_guidance_step_ratio

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt, height, width, negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds, negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Pool the *positive* prompt only.  We deliberately exclude the
        # negative prompt from the predictor's condition: the goal of motion
        # guidance is to nudge generation toward prompt-conditional motion,
        # not toward negative-prompt motion.  Pooling is mean-over-tokens
        # (T5 has no [CLS]), the canonical generic aggregation.
        prompt_pooled = pool_prompt_embeds(prompt_embeds).to(device=device)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        timesteps_sched, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps,
        )
        self._num_timesteps = len(timesteps_sched)

        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames = num_frames + additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, latent_channels,
            num_frames, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        num_warmup_steps = max(
            len(timesteps_sched) - num_inference_steps * self.scheduler.order, 0,
        )
        predictor.to(device=device, dtype=latents.dtype)
        prompt_pooled = prompt_pooled.to(dtype=latents.dtype)
        total_steps = len(timesteps_sched)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps_sched):
                if self.interrupt:
                    continue
                self._current_timestep = t

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t,
                )
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(
                            math.pi * (
                                (num_inference_steps - t.item()) / num_inference_steps
                            ) ** 5.0
                        )) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Prompt-conditioned classifier guidance: steer noise_pred via
                # the gradient of the temporal-consistency loss, with the
                # predictor seeing the (positive) prompt.
                pred_type = getattr(self.scheduler.config, "prediction_type", "v_prediction")
                noise_pred = apply_classifier_guidance(
                    predictor=predictor,
                    noise_pred=noise_pred,
                    latents=latents,
                    timestep=t,
                    alphas_cumprod=self.scheduler.alphas_cumprod.to(device),
                    guidance_lambda=guidance_lambda,
                    guidance_step_ratio=guidance_step_ratio,
                    current_step=i,
                    total_steps=total_steps,
                    prediction_type=pred_type,
                    prompt_pooled=prompt_pooled,
                )

                # Scheduler step (now using guided noise_pred).
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    step_output = self.scheduler.step(
                        noise_pred, t, latents,
                        **extra_step_kwargs, return_dict=True,
                    )
                    prev_sample = step_output.prev_sample
                else:
                    prev_sample = self.scheduler.step(
                        noise_pred, None, t,
                        timesteps_sched[i - 1] if i > 0 else None,
                        latents, **extra_step_kwargs, return_dict=False,
                    )[0]

                latents = prev_sample.to(prompt_embeds.dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals().get(k)
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs,
                    )
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds,
                    )

                if (
                    i == len(timesteps_sched) - 1
                    or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    )
                ):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if not output_type == "latent":
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(
                video=video, output_type=output_type,
            )
        else:
            video = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return CogVideoXPipelineOutput(frames=video)
