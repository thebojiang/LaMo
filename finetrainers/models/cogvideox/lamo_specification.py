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

# LaMo: Prompt-conditioned Temporal Consistency Predictor — CogVideoX
# specification.
#
# Differences from prompt-agnostic predecessor:
#   - Predictor accepts a mean-pooled prompt embedding via FiLM, so the
#     temporal-consistency guidance is conditioned on the prompt.
#   - Denoiser-side motion auxiliary loss is reduced to a single term:
#     BMV-relL2.  All other prompt-agnostic predecessor ablation losses are removed.
#   - Inference-time guidance is exclusively the gradient-based classifier
#     guidance (``apply_classifier_guidance``).
#
# Self-contained — does not import from any previous LaMo specification
# version.  Only depends on ``physical_motion_lamo``.

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import CogVideoXDDIMScheduler, CogVideoXTransformer3DModel

from ...logging import get_logger
from ...processors import ProcessorMixin
from ...typing import SchedulerType
from ..utils import DiagonalGaussianDistribution
from ..wan.physical_motion_lamo import (
    TemporalConsistencyPredictor,
    compute_denoiser_motion_loss,
    compute_predictor_loss,
    pool_prompt_embeds,
)
from .base_specification import CogVideoXModelSpecification
from .utils import prepare_rotary_positional_embeddings

logger = get_logger()

try:
    from utils.model_path_resolver import resolve_pretrained_model_path
except ImportError:
    resolve_pretrained_model_path = None


class LaMoCogVideoXModelSpecification(CogVideoXModelSpecification):
    """CogVideoX + LaMo (prompt-conditioned Temporal Consistency Predictor).

    Training (``enable_denoiser_motion_loss=False``, default):
        Train only the predictor on precomputed GT latent pairs (with the
        mean-pooled prompt as FiLM condition).  Denoiser is frozen.

    Training (``enable_denoiser_motion_loss=True``):
        Standard diffusion forward (noise → transformer → x̂₀) plus the
        BMV-relL2 motion loss on x̂₀.  Predictor also trains on GT latents
        simultaneously.

    Inference:
        The trained predictor is plugged into ``CogVideoXPipelineLaMo``
        which calls ``apply_classifier_guidance`` with the pooled positive
        prompt.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "THUDM/CogVideoX-5b",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        # --- lamo predictor architecture ---
        predictor_in_channels: int = 16,
        predictor_hidden_channels: int = 256,
        predictor_num_res_blocks: int = 8,
        predictor_use_se: bool = True,
        predictor_use_prev_delta: bool = False,
        predictor_use_prompt_cond: bool = True,
        prompt_text_dim: int = 4096,           # T5-XXL hidden size for CogVideoX-5b
        prompt_cond_dim: int = 128,            # FiLM bottleneck dimension
        prompt_dropout_prob: float = 0.1,      # classifier-free dropout during predictor training
        # --- predictor training ---
        predictor_lr: float = 1e-3,
        predictor_lr_scheduler: str = "cosine",
        predictor_noise_aug_prob: float = 0.5,
        predictor_noise_aug_scale: float = 1.0,
        predictor_cosine_weight: float = 0.5,
        # --- Denoiser motion auxiliary loss (BMV-relL2 only) ---
        enable_denoiser_motion_loss: bool = False,
        lambda_bmv_rel_l2: float = 0.0,
        bmv_tau: int = 2,
        # --- inference: classifier guidance defaults ---
        guidance_lambda: float = 15.0,
        guidance_step_ratio: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
            condition_model_processors=condition_model_processors,
            latent_model_processors=latent_model_processors,
            **kwargs,
        )

        # Predictor architecture
        self.predictor_in_channels = predictor_in_channels
        self.predictor_hidden_channels = predictor_hidden_channels
        self.predictor_num_res_blocks = predictor_num_res_blocks
        self.predictor_use_se = predictor_use_se
        self.predictor_use_prev_delta = predictor_use_prev_delta
        self.predictor_use_prompt_cond = predictor_use_prompt_cond
        self.prompt_text_dim = prompt_text_dim
        self.prompt_cond_dim = prompt_cond_dim
        self.prompt_dropout_prob = prompt_dropout_prob

        # Predictor training
        self.predictor_noise_aug_prob = predictor_noise_aug_prob
        self.predictor_noise_aug_scale = predictor_noise_aug_scale
        self.predictor_cosine_weight = predictor_cosine_weight

        # Denoiser-side BMV-relL2 motion loss
        self.enable_denoiser_motion_loss = enable_denoiser_motion_loss
        self.lambda_bmv_rel_l2 = lambda_bmv_rel_l2
        self.bmv_tau = bmv_tau

        # Inference guidance
        self.guidance_lambda = guidance_lambda
        self.guidance_step_ratio = guidance_step_ratio

        # Exposed as physical_lr so LaMoSFTTrainer picks it up for separate
        # optimizer LR.
        self.physical_lr = predictor_lr
        self.predictor_lr_scheduler = predictor_lr_scheduler

        # Runtime state
        self._predictor: Optional[TemporalConsistencyPredictor] = None
        self._init_logged = False

    # ------------------------------------------------------------------ #
    # Trainable parts                                                    #
    # ------------------------------------------------------------------ #

    def get_trainable_model_parts(
        self, transformer: torch.nn.Module,
    ) -> List[torch.nn.Module]:
        parts = [transformer]
        if self._predictor is not None:
            parts.append(self._predictor)
        return parts

    # ------------------------------------------------------------------ #
    # Weight path helpers                                                #
    # ------------------------------------------------------------------ #

    def resolve_weight_path(self, path: str) -> str:
        return path

    # ------------------------------------------------------------------ #
    # Load                                                                #
    # ------------------------------------------------------------------ #

    def load_diffusion_models(self, ckpt_path=None) -> Dict[str, torch.nn.Module]:
        out = super().load_diffusion_models()

        self._predictor = TemporalConsistencyPredictor(
            in_channels=self.predictor_in_channels,
            hidden_channels=self.predictor_hidden_channels,
            num_res_blocks=self.predictor_num_res_blocks,
            use_se=self.predictor_use_se,
            use_prev_delta=self.predictor_use_prev_delta,
            use_prompt_cond=self.predictor_use_prompt_cond,
            prompt_text_dim=self.prompt_text_dim,
            prompt_cond_dim=self.prompt_cond_dim,
        )
        n_params = sum(p.numel() for p in self._predictor.parameters())
        logger.info(
            "LaMo: created prompt-conditioned TemporalConsistencyPredictor "
            "(%d params, in_ch=%d, hidden=%d, blocks=%d, se=%s, prev_delta=%s, "
            "prompt_cond=%s, text_dim=%d, cond_dim=%d).",
            n_params, self.predictor_in_channels,
            self.predictor_hidden_channels, self.predictor_num_res_blocks,
            self.predictor_use_se, self.predictor_use_prev_delta,
            self.predictor_use_prompt_cond, self.prompt_text_dim,
            self.prompt_cond_dim,
        )

        if ckpt_path:
            resolved = self.resolve_weight_path(ckpt_path)
            if os.path.isdir(resolved):
                self._load_physical_weights(resolved)

        return out

    # ------------------------------------------------------------------ #
    # Save / load predictor weights                                      #
    # ------------------------------------------------------------------ #

    def _get_physical_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        if self._predictor is not None:
            for k, v in self._predictor.state_dict().items():
                state["physical_lamo." + k] = v
        return state

    def _save_physical_weights(
        self, directory: str, state_dict: Dict[str, torch.Tensor],
    ) -> None:
        if not state_dict:
            return
        from safetensors.torch import save_file

        path = os.path.join(directory, "predictor.safetensors")
        os.makedirs(directory, exist_ok=True)
        save_file(state_dict, path)
        logger.info("LaMo: saved predictor to %s", path)

    def _load_physical_weights(self, directory: str) -> None:
        if self._predictor is None:
            return
        path = os.path.join(directory, "predictor.safetensors")
        if not os.path.isfile(path):
            return
        from safetensors.torch import load_file

        state = load_file(path)
        phys = {
            k[len("physical_lamo."):]: v
            for k, v in state.items()
            if k.startswith("physical_lamo.")
        }
        if phys:
            self._predictor.load_state_dict(phys, strict=False)
            logger.info("LaMo: loaded predictor weights from %s", path)

    def load_physical_weights(self, directory: str) -> None:
        self._load_physical_weights(self.resolve_weight_path(directory))

    def _save_lora_weights(
        self, directory, transformer_state_dict=None, scheduler=None,
        *args, _extra_physical_state=None, **kwargs,
    ):
        super()._save_lora_weights(
            directory, transformer_state_dict, scheduler, *args, **kwargs,
        )
        state = self._get_physical_state_dict()
        if _extra_physical_state:
            state.update(_extra_physical_state)
        if state:
            self._save_physical_weights(directory, state)

    def _save_model(
        self, directory, transformer, transformer_state_dict=None, scheduler=None,
    ):
        super()._save_model(directory, transformer, transformer_state_dict, scheduler)
        state = self._get_physical_state_dict()
        if state:
            self._save_physical_weights(directory, state)

    # ------------------------------------------------------------------ #
    # Helper: pull pooled prompt out of condition_model_conditions       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_prompt_pooled(
        condition_model_conditions: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Mean-pool ``encoder_hidden_states`` to a per-sample prompt vector.

        Returns ``None`` when no prompt embeddings are available (e.g. if a
        downstream caller passes only latent conditions); the predictor will
        then fall back to its learned null embedding.
        """
        prompt_embeds = condition_model_conditions.get("encoder_hidden_states")
        if prompt_embeds is None:
            return None
        attn_mask = condition_model_conditions.get("prompt_attention_mask")
        return pool_prompt_embeds(prompt_embeds, attn_mask)

    # ------------------------------------------------------------------ #
    # Forward (training)                                                  #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDDIMScheduler,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        VAE_SPATIAL_SCALE_FACTOR = 8
        patch_size = self.transformer_config.patch_size
        patch_size_t = getattr(self.transformer_config, "patch_size_t", None)

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
        else:
            posterior = DiagonalGaussianDistribution(
                latent_model_conditions.pop("latents"), _dim=2,
            )
            latents = posterior.sample(generator=generator)
            del posterior

        if not getattr(self.vae_config, "invert_scale_latents", False):
            latents = latents * self.vae_config.scaling_factor

        if patch_size_t is not None:
            latents = self._pad_frames(latents, patch_size_t)

        dev = next(transformer.parameters()).device
        dtype = next(transformer.parameters()).dtype
        latents = latents.to(device=dev, dtype=dtype)

        if self._predictor is not None:
            self._predictor.to(device=dev, dtype=dtype)
            self._predictor.requires_grad_(True)

        training_step = kwargs.get("training_step", 0)

        # Pre-compute pooled prompt once; passed to both predictor-only and
        # joint-training paths.  Always detached from the diffusion graph
        # because the prompt embedding is produced by the (frozen) text
        # encoder upstream and we do not want predictor gradients to flow
        # into the text encoder.
        prompt_pooled = self._extract_prompt_pooled(condition_model_conditions)
        if prompt_pooled is not None:
            prompt_pooled = prompt_pooled.detach().to(device=dev, dtype=dtype)

        # Scheduler alphas for predictor noise augmentation.
        scheduler_alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
        if scheduler_alphas_cumprod is not None:
            scheduler_alphas_cumprod = scheduler_alphas_cumprod.to(
                device=dev, dtype=torch.float32,
            )

        if self.enable_denoiser_motion_loss:
            # ============================================================
            # Joint training: diffusion + BMV-relL2 + predictor
            # ============================================================
            if not self._init_logged:
                logger.info(
                    "LaMo: joint training "
                    "(enable_denoiser_motion_loss=True, lambda_bmv_rel_l2=%.4f, "
                    "noise_aug_prob=%.2f, prompt_cond=%s).",
                    self.lambda_bmv_rel_l2, self.predictor_noise_aug_prob,
                    self.predictor_use_prompt_cond,
                )
                self._init_logged = True

            # Standard diffusion forward (matches base CogVideoX spec).
            rope_base_height = self.transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
            rope_base_width = self.transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR

            timesteps = (sigmas.flatten() * 1000.0).long()
            noise = torch.zeros_like(latents).normal_(generator=generator)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            B, T, C, H, W = latents.shape
            ofs_emb = (
                None
                if getattr(self.transformer_config, "ofs_embed_dim", None) is None
                else latents.new_full((B,), fill_value=2.0)
            )

            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    height=H * VAE_SPATIAL_SCALE_FACTOR,
                    width=W * VAE_SPATIAL_SCALE_FACTOR,
                    num_frames=T,
                    vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    attention_head_dim=self.transformer_config.attention_head_dim,
                    device=dev,
                    base_height=rope_base_height,
                    base_width=rope_base_width,
                )
                if self.transformer_config.use_rotary_positional_embeddings
                else None
            )

            latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
            latent_model_conditions["image_rotary_emb"] = image_rotary_emb
            latent_model_conditions["ofs"] = ofs_emb

            velocity = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                return_dict=False,
            )[0]
            pred = scheduler.get_velocity(velocity, noisy_latents, timesteps)
            target = latents

            # BMV-relL2 motion loss on x̂₀ — gradients flow to transformer.
            dn_motion_loss, dn_motion_logs = compute_denoiser_motion_loss(
                pred_x0=pred,
                target_x0=target,
                sigmas=sigmas,
                lambda_bmv_rel_l2=self.lambda_bmv_rel_l2,
                bmv_tau=self.bmv_tau,
            )

            # Predictor loss on GT latents (detached from diffusion graph).
            predictor_loss, predictor_logs = compute_predictor_loss(
                predictor=self._predictor,
                latents=latents.detach(),
                prompt_pooled=prompt_pooled,
                scheduler_alphas_cumprod=scheduler_alphas_cumprod,
                noise_aug_prob=self.predictor_noise_aug_prob,
                noise_aug_scale=self.predictor_noise_aug_scale,
                generator=generator,
                cosine_weight=self.predictor_cosine_weight,
                prompt_dropout_prob=self.prompt_dropout_prob,
            )

            total_physical_loss = predictor_loss + dn_motion_loss

            logs = {**predictor_logs, **dn_motion_logs}
            logs["lamo_denoiser_motion"] = 1
            logs["lamo_training_step"] = training_step

            return pred, target, sigmas, {
                "physical_loss": total_physical_loss,
                "physical_logs": logs,
            }

        else:
            # ============================================================
            # Predictor-only training (transformer frozen)
            # ============================================================
            for p in transformer.parameters():
                p.requires_grad_(False)

            if not self._init_logged:
                logger.info(
                    "LaMo: predictor-only training "
                    "(noise_aug_prob=%.2f, noise_aug_scale=%.2f, prompt_cond=%s, "
                    "prompt_dropout_prob=%.2f).",
                    self.predictor_noise_aug_prob, self.predictor_noise_aug_scale,
                    self.predictor_use_prompt_cond, self.prompt_dropout_prob,
                )
                self._init_logged = True

            loss, logs = compute_predictor_loss(
                predictor=self._predictor,
                latents=latents,
                prompt_pooled=prompt_pooled,
                scheduler_alphas_cumprod=scheduler_alphas_cumprod,
                noise_aug_prob=self.predictor_noise_aug_prob,
                noise_aug_scale=self.predictor_noise_aug_scale,
                generator=generator,
                cosine_weight=self.predictor_cosine_weight,
                prompt_dropout_prob=self.prompt_dropout_prob,
            )

            logs["lamo_denoiser_motion"] = 0
            logs["lamo_training_step"] = training_step

            dummy = latents.detach()
            return dummy, dummy, sigmas.to(device=dev), {
                "physical_loss": loss,
                "physical_logs": logs,
            }
