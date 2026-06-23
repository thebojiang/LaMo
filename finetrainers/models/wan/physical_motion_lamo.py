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

# LaMo: Prompt-conditioned Temporal Consistency Predictor.
#
# Inheritance map vs prompt-agnostic predecessor:
#   - Predictor inputs: z_t (and optionally Δz_{t-1}) + pooled prompt embedding,
#     injected into every ResBlock via FiLM (Feature-wise Linear Modulation).
#     Identity at init: γ-head zero-init + (1+γ) shift means the predictor
#     starts as the prompt-agnostic predecessor prompt-agnostic predictor.
#   - Denoiser auxiliary motion losses: only ``bmv_rel_l2`` is kept (the
#     scale-normalised L² on the τ-lag bulk motion vector).  All other prompt-agnostic predecessor
#     ablation losses (cosine / PCA / CKA / Sobolev / spatial / gated / ...)
#     are dropped to keep the configuration surface small.
#   - Inference-time guidance: only ``apply_classifier_guidance`` is provided
#     (the gradient-based steering of ``noise_pred`` through a temporal-
#     consistency loss); ``apply_temporal_guidance`` (z0_hat correction) is
#     intentionally not exposed.
#
# Why prompt-conditioning?  In prompt-agnostic predecessor the predictor is a pure conv-net on the
# noisy latent and is therefore prompt-agnostic; at inference the gradient of
# the consistency loss pulls ``noise_pred`` toward whatever motion *prior* the
# predictor learned during training.  Empirically this raises Physical
# Coherence (PC) but lowers Semantic Adherence (SA), because the prior can
# overrule the prompt-implied trajectory.  Conditioning the predictor on the
# prompt ties the gradient to the prompt-conditional motion distribution, so
# motion guidance no longer fights the prompt.
#
# Self-contained — does not import from any previous LaMo version.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ChannelSEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration.

    Captures inter-channel dependencies that are critical when the 3D causal
    VAE encodes temporal/motion information across the latent channels.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(-2, -1), keepdim=True)       # [B, C, 1, 1]
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class FiLMResBlock(nn.Module):
    """Conv3x3 → GroupNorm → FiLM → GELU → Conv3x3 → GroupNorm → (SE) + residual.

    FiLM (Feature-wise Linear Modulation) injects a per-channel ``(γ, β)``
    pair derived from a global condition vector after the first GroupNorm.
    The FiLM head is zero-initialised so that at the start of training
    ``γ ≡ 0, β ≡ 0`` and the block is identical to the prompt-agnostic predecessor ResBlock under
    the convention ``y = (1 + γ) · x + β``.

    When ``cond_dim is None`` the FiLM head is not created and the block
    behaves exactly as the prompt-agnostic predecessor ResBlock — used when prompt-conditioning is
    disabled, so the lamo predictor degenerates to prompt-agnostic predecessor cleanly.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int = 8,
        use_se: bool = False,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.se = ChannelSEBlock(channels) if use_se else None

        if cond_dim is not None:
            self.film = nn.Linear(cond_dim, 2 * channels)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.gn1(self.conv1(x))
        if self.film is not None and cond is not None:
            gamma_beta = self.film(cond)                       # [N, 2C]
            gamma, beta = gamma_beta.chunk(2, dim=-1)          # [N, C], [N, C]
            x = (1.0 + gamma).unsqueeze(-1).unsqueeze(-1) * x \
                + beta.unsqueeze(-1).unsqueeze(-1)
        x = F.gelu(x)
        x = self.gn2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        x = F.gelu(x + residual)
        return x


# ---------------------------------------------------------------------------
# Prompt pooling helper (mean-pool with optional attention mask)
# ---------------------------------------------------------------------------


def pool_prompt_embeds(
    prompt_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-pool [B, L, D] T5/CLIP-style prompt embeddings to [B, D].

    Padding tokens are masked out when ``attention_mask`` is provided; without
    a mask the function falls back to a plain ``mean`` over the sequence axis.
    Mean-pooling is the canonical way to derive a single prompt vector from
    T5 (which lacks a dedicated [CLS] token) and is the standard generic
    aggregation used in prompt-conditioned ControlNets and FiLM CNNs.
    """
    if prompt_embeds.dim() != 3:
        raise ValueError(
            f"prompt_embeds must be [B, L, D]; got {tuple(prompt_embeds.shape)}",
        )
    if attention_mask is None:
        return prompt_embeds.mean(dim=1)
    mask = attention_mask.to(dtype=prompt_embeds.dtype).unsqueeze(-1)  # [B, L, 1]
    summed = (prompt_embeds * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


# ---------------------------------------------------------------------------
# Prompt-conditioned Temporal Consistency Predictor
# ---------------------------------------------------------------------------


class TemporalConsistencyPredictor(nn.Module):
    """Predicts the next-frame latent change Δz from a single latent frame
    z_t, optionally conditioned on the previous frame change Δz_{t-1} and on
    a *pooled prompt embedding*.

    Architecture
    ------------
        Conv3x3(C_in → H)
        →  N × FiLMResBlock(H, cond_dim=D_cond)
        →  Conv3x3(H → C)

    where ``C_in = 2·in_channels`` if ``use_prev_delta`` else ``in_channels``.
    The output projection is zero-initialised so the predictor starts as the
    constant-zero map (``Δz_pred ≡ 0``) — a safe identity-skip behaviour.

    Prompt pathway
    --------------
    Pooled prompt vector ``p ∈ ℝ^{D_text}`` is mapped through a single linear
    bottleneck

        p_cond = GELU(prompt_proj(p))      ∈ ℝ^{D_cond}

    and fed to every FiLM head as the global condition.  A learned
    ``null_prompt`` parameter (zero-initialised) replaces ``p`` when
    classifier-free training drops the prompt — the FiLM head does not
    receive a hard "no signal" tensor, only a learnable null embedding the
    block can specialise to.

    When ``use_prompt_cond=False``, no FiLM heads / prompt-proj are created
    and the predictor is exactly the prompt-agnostic predecessor prompt-agnostic predictor.

    Parameters
    ----------
    in_channels:        VAE latent channels (16 for CogVideoX-5b).
    hidden_channels:    Feature width.
    num_res_blocks:     Number of FiLM ResBlocks.
    num_groups:         GroupNorm group count.
    use_se:             Enable Squeeze-and-Excitation in each ResBlock.
    use_prev_delta:     Concat Δz_{t-1} as extra input channels.
    use_prompt_cond:    Enable prompt FiLM conditioning.
    prompt_text_dim:    Dimension of the pooled prompt vector p (e.g. 4096
                        for T5-XXL).
    prompt_cond_dim:    Bottleneck dimension D_cond fed to every FiLM head.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        num_groups: int = 8,
        use_se: bool = True,
        use_prev_delta: bool = False,
        use_prompt_cond: bool = True,
        prompt_text_dim: int = 4096,
        prompt_cond_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_prev_delta = use_prev_delta
        self.use_prompt_cond = use_prompt_cond
        self.prompt_text_dim = prompt_text_dim
        self.prompt_cond_dim = prompt_cond_dim

        proj_in_channels = 2 * in_channels if use_prev_delta else in_channels
        self.input_proj = nn.Conv2d(proj_in_channels, hidden_channels, 3, padding=1)

        cond_dim = prompt_cond_dim if use_prompt_cond else None
        self.blocks = nn.ModuleList([
            FiLMResBlock(
                hidden_channels, num_groups=num_groups,
                use_se=use_se, cond_dim=cond_dim,
            )
            for _ in range(num_res_blocks)
        ])
        self.output_proj = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

        if use_prompt_cond:
            self.prompt_proj = nn.Linear(prompt_text_dim, prompt_cond_dim)
            # Standard near-zero init for prompt projection — combined with
            # zero FiLM heads, the early-training output is identical to the
            # prompt-agnostic predecessor baseline regardless of prompt content.
            nn.init.normal_(self.prompt_proj.weight, std=0.02)
            nn.init.zeros_(self.prompt_proj.bias)
            # Learned null embedding for classifier-free training (replaces
            # dropped prompts).  Zero-init: at start, identical to "no prompt".
            self.null_prompt = nn.Parameter(torch.zeros(prompt_text_dim))
        else:
            self.prompt_proj = None
            self.null_prompt = None

        self._zero_init_output()

    def _zero_init_output(self) -> None:
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def _project_prompt(
        self,
        prompt_pooled: Optional[torch.Tensor],
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Map a pooled prompt vector to the FiLM condition vector.

        Returns ``None`` if prompt conditioning is disabled (so each FiLM
        head will short-circuit and behave like the prompt-agnostic predecessor ResBlock).
        Accepts ``[B, D_text]`` or ``[N, D_text]`` (the caller is expected
        to broadcast across the per-frame-pair axis).  When
        ``prompt_pooled is None`` the learned null embedding is broadcast.
        """
        if not self.use_prompt_cond:
            return None
        if prompt_pooled is None:
            p = self.null_prompt.to(device=device, dtype=dtype)
            p = p.unsqueeze(0).expand(n_samples, -1)
        else:
            p = prompt_pooled.to(device=device, dtype=dtype)
            if p.dim() != 2:
                raise ValueError(
                    "prompt_pooled must be [B, D_text] or [N, D_text]; got "
                    f"{tuple(p.shape)}",
                )
            if p.shape[0] != n_samples:
                if n_samples % p.shape[0] != 0:
                    raise ValueError(
                        f"Cannot broadcast prompt batch {p.shape[0]} to "
                        f"{n_samples} samples (n_samples must be a multiple "
                        f"of prompt batch).",
                    )
                repeat = n_samples // p.shape[0]
                p = p.unsqueeze(1).expand(-1, repeat, -1).reshape(n_samples, -1)
        return F.gelu(self.prompt_proj(p))

    def forward(
        self,
        z_t: torch.Tensor,
        prompt_pooled: Optional[torch.Tensor] = None,
        prev_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t:            [N, C, H, W] single latent frame
            prompt_pooled:  [B, D_text] or [N, D_text]; ignored when prompt
                            conditioning is disabled.  When None and prompt
                            conditioning is enabled, the learned null
                            embedding is used.
            prev_delta:     [N, C, H, W] previous frame change Δz_{t-1};
                            required when ``use_prev_delta=True``.
        Returns:
            Δz_pred         [N, C, H, W] predicted frame-to-frame change.
        """
        z_t = z_t.to(dtype=self.input_proj.weight.dtype)
        if self.use_prev_delta:
            if prev_delta is None:
                prev_delta = torch.zeros_like(z_t)
            else:
                prev_delta = prev_delta.to(dtype=z_t.dtype)
            x = torch.cat([z_t, prev_delta], dim=1)
        else:
            x = z_t

        cond = self._project_prompt(
            prompt_pooled, n_samples=x.shape[0],
            device=x.device, dtype=x.dtype,
        )

        h = F.gelu(self.input_proj(x))
        for block in self.blocks:
            h = block(h, cond=cond)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Training: predictor loss on GT latent pairs (MSE + cosine)
# ---------------------------------------------------------------------------


def _build_prev_delta_seq(
    delta_z_true: torch.Tensor, B: int, T_minus_1: int,
    C: int, H: int, W: int,
) -> torch.Tensor:
    """Construct the [B*(T-1), C, H, W] prev_delta tensor where the entry
    for each ``z_t`` is ``Δz_{t-1}`` (zero for the very first pair)."""
    prev_delta_seq = torch.zeros(
        B, T_minus_1, C, H, W,
        device=delta_z_true.device, dtype=delta_z_true.dtype,
    )
    if T_minus_1 > 1:
        prev_delta_seq[:, 1:] = delta_z_true.reshape(B, T_minus_1, C, H, W)[:, :-1]
    return prev_delta_seq.reshape(-1, C, H, W)


def _expand_prompt_to_pairs(
    prompt_pooled: Optional[torch.Tensor], B: int, T_minus_1: int,
) -> Optional[torch.Tensor]:
    """Broadcast ``[B, D]`` prompt to ``[B*(T-1), D]``.

    Returns ``None`` if ``prompt_pooled is None`` (caller will fall back to
    the predictor's learned null embedding when prompt conditioning is on).
    """
    if prompt_pooled is None:
        return None
    if prompt_pooled.dim() != 2 or prompt_pooled.shape[0] != B:
        raise ValueError(
            f"prompt_pooled must be [B={B}, D]; got {tuple(prompt_pooled.shape)}",
        )
    return (
        prompt_pooled.unsqueeze(1)
        .expand(B, T_minus_1, prompt_pooled.shape[-1])
        .reshape(B * T_minus_1, prompt_pooled.shape[-1])
    )


def _apply_prompt_dropout(
    prompt_per_pair: Optional[torch.Tensor],
    prompt_dropout_prob: float,
    use_prompt_cond: bool,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Randomly zero-out the prompt for a Bernoulli fraction of frame-pairs.

    A zero prompt is interpreted by the predictor as "use the learned null
    embedding" — see ``TemporalConsistencyPredictor._project_prompt``.  We
    return ``None`` for pairs that are dropped only when *all* pairs in the
    batch are dropped; otherwise we keep a per-row mask.
    """
    if prompt_per_pair is None or not use_prompt_cond:
        return prompt_per_pair
    if prompt_dropout_prob <= 0.0:
        return prompt_per_pair
    N = prompt_per_pair.shape[0]
    keep_mask = (
        torch.rand(N, 1, device=prompt_per_pair.device, generator=generator)
        > prompt_dropout_prob
    ).to(dtype=prompt_per_pair.dtype)
    # Where keep_mask == 0, the row becomes zero → predictor treats it as
    # "drop"; combined with the learned null embedding being zero-init, the
    # initial behaviour matches the unconditional path.
    return prompt_per_pair * keep_mask


def compute_predictor_loss(
    predictor: TemporalConsistencyPredictor,
    latents: torch.Tensor,
    prompt_pooled: Optional[torch.Tensor] = None,
    scheduler_alphas_cumprod: Optional[torch.Tensor] = None,
    noise_aug_prob: float = 0.5,
    noise_aug_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
    cosine_weight: float = 0.5,
    prompt_dropout_prob: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Standalone predictor-only loss on GT latent pairs (transformer frozen).

    L = MSE(Δz_pred, Δz_true) + cosine_weight · (1 − cos(Δz_pred, Δz_true))

    Optionally applies a *diffusion-aligned* noise augmentation to the input
    ``z_t``: with probability ``noise_aug_prob``, a sample is replaced by
    ``z_t + σ · ε`` where ``σ = √(1 − ᾱ_t) · scale`` for a uniformly sampled
    diffusion timestep.  This makes the predictor robust to the kind of
    partial noise present in the denoiser's mid-step ``x̂₀``.

    Prompt conditioning
    -------------------
    ``prompt_pooled`` is optional; pass ``[B, D_text]`` to condition the
    predictor on the (mean-pooled) prompt embedding.  ``prompt_dropout_prob``
    governs classifier-free-style training: each frame-pair is independently
    swapped to a zero prompt with this probability so that the predictor
    learns both conditional and unconditional behaviours.
    """
    B, T, C, H, W = latents.shape
    if T < 2:
        zero = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        return zero, {"lamo_predictor_loss": 0.0, "lamo_predictor_num_pairs": 0}

    T_minus_1 = T - 1
    z_curr = latents[:, :-1].reshape(-1, C, H, W)        # [B*(T-1), C, H, W]
    z_next = latents[:, 1:].reshape(-1, C, H, W)
    delta_z_true = z_next - z_curr
    N = z_curr.shape[0]

    # --- prev_delta input (only when predictor uses it) ---
    prev_delta_input = None
    if predictor.use_prev_delta:
        prev_delta_input = _build_prev_delta_seq(
            delta_z_true, B, T_minus_1, C, H, W,
        )

    # --- per-pair prompt (broadcast B → B*(T-1)) + classifier-free dropout ---
    prompt_per_pair = _expand_prompt_to_pairs(prompt_pooled, B, T_minus_1)
    prompt_per_pair = _apply_prompt_dropout(
        prompt_per_pair, prompt_dropout_prob,
        predictor.use_prompt_cond, generator=generator,
    )

    # --- diffusion-aligned noise augmentation on z_t ---
    if noise_aug_prob > 0:
        aug_mask = torch.rand(N, 1, 1, 1, device=z_curr.device, generator=generator) < noise_aug_prob
        if scheduler_alphas_cumprod is not None:
            num_timesteps = scheduler_alphas_cumprod.shape[0]
            sampled_t = torch.randint(
                0, num_timesteps, (N,),
                device=z_curr.device, generator=generator,
            )
            alpha_bar = scheduler_alphas_cumprod[sampled_t].float()
            sigma = (1.0 - alpha_bar).sqrt() * noise_aug_scale
            sigma = sigma.view(N, 1, 1, 1)
        else:
            sigma = torch.rand(N, 1, 1, 1, device=z_curr.device, generator=generator) * noise_aug_scale
        noise = torch.randn_like(z_curr)
        z_input = torch.where(aug_mask, z_curr + sigma * noise, z_curr)
    else:
        z_input = z_curr

    delta_z_pred = predictor(
        z_input, prompt_pooled=prompt_per_pair, prev_delta=prev_delta_input,
    )

    mse_loss = F.mse_loss(delta_z_pred, delta_z_true)
    flat_pred = delta_z_pred.reshape(N, -1)
    flat_true = delta_z_true.reshape(N, -1)
    cos_sim = F.cosine_similarity(flat_pred, flat_true, dim=-1)
    cos_loss = (1.0 - cos_sim).mean()

    loss = mse_loss + cosine_weight * cos_loss

    with torch.no_grad():
        pred_norm = flat_pred.norm(dim=-1)
        true_norm = flat_true.norm(dim=-1)
        norm_ratio = (pred_norm / true_norm.clamp(min=1e-8)).mean()
        ss_res = (flat_pred - flat_true).pow(2).sum()
        ss_tot = (flat_true - flat_true.mean(dim=0, keepdim=True)).pow(2).sum()
        r_squared = 1.0 - ss_res / ss_tot.clamp(min=1e-8)

    logs = {
        "lamo_mse_loss": mse_loss.detach().item(),
        "lamo_cos_loss": cos_loss.detach().item(),
        "lamo_predictor_loss": loss.detach().item(),
        "lamo_predictor_num_pairs": N,
        "lamo_pred_norm": delta_z_pred.detach().norm().item(),
        "lamo_true_norm": delta_z_true.detach().norm().item(),
        "lamo_cos_sim": cos_sim.detach().mean().item(),
        "lamo_norm_ratio": norm_ratio.item(),
        "lamo_r_squared": r_squared.item(),
        "lamo_prompt_cond": int(predictor.use_prompt_cond),
        "lamo_prompt_dropout_prob": float(prompt_dropout_prob),
    }
    return loss, logs


# ---------------------------------------------------------------------------
# Training: BMV-relL2 motion loss on denoiser's predicted x̂₀
# ---------------------------------------------------------------------------
#
# Definition.  For each consecutive frame pair we form the τ-lag bulk motion
# vector  μ = mean_{H,W}(z_{t+τ} − z_t) ∈ ℝ^C  and impose
#
#       L_BMV-relL2 = E_{b,t}[ ‖μ_pred − μ_true‖² / sg[‖μ_true‖²] ]
#
# Expanding with r = ‖μ_p‖/‖μ_t‖ and θ = ∠(μ_p, μ_t):
#       L = (r − 1)²  +  2 r (1 − cos θ)
# i.e. the loss is exactly *magnitude error* + *direction error* with no
# extra hyper-parameters.  τ=2 (default) whitens the CogVideoX VAE's MA(1)
# encoder noise (corr(Δ_2 z_t, Δ_2 z_{t+1}) = 0 for an MA(1) process), so we
# do not need a sign-mask hack.  The stop-grad denominator makes the loss
# scale-invariant per frame-pair.
#
# It is weighted by w(σ) = mean_b((1 − σ²)_+) so the gradient vanishes at
# high noise where x̂₀ is uninformative — same schedule used by every x̂₀-
# side motion loss in prompt-agnostic predecessor.


def _compute_motion_bmv_rel_l2(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    tau: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale-normalised L² on the τ-lag bulk motion vector (parameter-free
    apart from τ).  Returns 0 when ``T ≤ τ``."""
    B, T, C, H, W = pred_x0.shape
    if T <= tau:
        return torch.tensor(0.0, device=pred_x0.device, dtype=pred_x0.dtype)

    dz_pred = pred_x0[:, tau:] - pred_x0[:, :-tau]      # [B, T-τ, C, H, W]
    dz_true = target_x0[:, tau:] - target_x0[:, :-tau]
    mu_pred = dz_pred.mean(dim=(-2, -1)).reshape(-1, C).float()
    mu_true = dz_true.mean(dim=(-2, -1)).reshape(-1, C).float()

    num = (mu_pred - mu_true).pow(2).sum(dim=-1)            # [N]
    den = mu_true.pow(2).sum(dim=-1).detach().clamp(min=eps)  # stop-grad
    return (num / den).mean().to(dtype=pred_x0.dtype)


def compute_denoiser_motion_loss(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    sigmas: torch.Tensor,
    lambda_bmv_rel_l2: float = 0.0,
    bmv_tau: int = 2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """lamo denoiser-side motion auxiliary loss — BMV-relL2 only.

    L = w(σ) · λ_relL2 · E[ ‖μ_pred − μ_true‖² / sg[‖μ_true‖²] ]

    Args:
        pred_x0:            [B, T, C, H, W] denoiser's predicted clean latents
        target_x0:          [B, T, C, H, W] ground-truth clean latents
        sigmas:             per-sample noise levels (flattens to ≥ B)
        lambda_bmv_rel_l2:  scalar weight (0 = disabled)
        bmv_tau:            temporal lag τ (default 2, whitens MA(1))

    Returns:
        (scalar loss, dict of scalar logs).  Returns 0 / {} when disabled.
    """
    B, T, _, _, _ = pred_x0.shape
    tau = max(int(bmv_tau), 1)
    if lambda_bmv_rel_l2 <= 0 or T <= tau:
        zero = torch.tensor(0.0, device=pred_x0.device, dtype=pred_x0.dtype)
        return zero, {}

    sigma_val = sigmas.flatten()[:B].float()
    motion_weight = (1.0 - sigma_val.pow(2)).clamp(min=0).mean()

    bmv = _compute_motion_bmv_rel_l2(pred_x0, target_x0, tau=tau)
    loss = lambda_bmv_rel_l2 * motion_weight * bmv

    logs: Dict[str, float] = {
        "lamo_dn_bmv_rel_l2": bmv.detach().item(),
        "lamo_dn_bmv_weight": motion_weight.detach().item(),
        "lamo_dn_bmv_loss": loss.detach().item(),
        "lamo_dn_bmv_tau": int(tau),
    }
    return loss, logs


# ---------------------------------------------------------------------------
# Inference (recommended): Classifier guidance — steer noise_pred via gradient
# ---------------------------------------------------------------------------


def apply_classifier_guidance(
    predictor: TemporalConsistencyPredictor,
    noise_pred: torch.Tensor,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    guidance_lambda: float = 15.0,
    guidance_step_ratio: float = 0.8,
    current_step: int = 0,
    total_steps: int = 50,
    prediction_type: str = "v_prediction",
    prompt_pooled: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Modify ``noise_pred`` (post-CFG) using the gradient of a temporal-
    consistency loss, keeping the denoising trajectory inside the scheduler's
    mathematical framework.

    Pipeline
    --------
      1. Differentiably derive  x̂₀ = √ᾱ · x_t − √(1−ᾱ) · v   (or the ε-form).
      2. Compute  L = MSE(P(x̂₀_i, prompt) + x̂₀_i,  x̂₀_{i+1}).
      3. g = ∇_{noise_pred} L                (predictor frozen; only activations).
      4. noise_pred_guided = noise_pred − λ · g.

    The chain rule provides correct time-dependent attenuation: at low noise
    the gradient is naturally smaller because x̂₀ is already clean.

    The prompt is supplied *post-CFG-mix*: we steer toward the prompt-
    conditional motion distribution, so ``prompt_pooled`` should be the mean-
    pool of the *positive* prompt embedding (shape ``[B, D_text]``).
    Passing ``None`` falls back to the learned null embedding, which is the
    prompt-agnostic predecessor-style prompt-agnostic guidance behaviour.
    """
    start_step = int((1.0 - guidance_step_ratio) * total_steps)
    if guidance_lambda <= 0 or current_step < start_step:
        return noise_pred

    B, T, C, H, W = latents.shape
    if T < 2:
        return noise_pred

    t_idx = timestep.long().clamp(0, len(alphas_cumprod) - 1)
    alpha_prod_t = alphas_cumprod[t_idx].float()
    while alpha_prod_t.dim() < noise_pred.dim():
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)

    eps = noise_pred.detach().clone().requires_grad_(True)

    # Per-pair prompt broadcast (B → B*(T-1)).
    if prompt_pooled is not None:
        prompt_per_pair = (
            prompt_pooled.detach()
            .to(device=latents.device)
            .unsqueeze(1)
            .expand(B, T - 1, prompt_pooled.shape[-1])
            .reshape(B * (T - 1), prompt_pooled.shape[-1])
        )
    else:
        prompt_per_pair = None

    with torch.enable_grad():
        x_t = latents.detach().float()
        if prediction_type == "v_prediction":
            pred_x0 = alpha_prod_t.sqrt() * x_t - (1 - alpha_prod_t).sqrt() * eps
        else:
            pred_x0 = (x_t - (1 - alpha_prod_t).sqrt() * eps) / alpha_prod_t.sqrt().clamp(min=1e-8)

        z_curr = pred_x0[:, :-1].reshape(-1, C, H, W)
        z_next = pred_x0[:, 1:].reshape(-1, C, H, W)

        prev_delta = None
        if predictor.use_prev_delta:
            delta_all = z_next - z_curr
            pd_seq = torch.zeros(
                B, T - 1, C, H, W,
                device=latents.device, dtype=pred_x0.dtype,
            )
            if T > 2:
                pd_seq[:, 1:] = delta_all.reshape(B, T - 1, C, H, W)[:, :-1]
            prev_delta = pd_seq.reshape(-1, C, H, W)

        delta_pred = predictor(
            z_curr, prompt_pooled=prompt_per_pair, prev_delta=prev_delta,
        )
        delta_pred = delta_pred.to(dtype=z_curr.dtype)

        loss = F.mse_loss(z_curr + delta_pred, z_next)
        grad = torch.autograd.grad(loss, eps)[0]

    return noise_pred - guidance_lambda * grad.to(dtype=noise_pred.dtype)
