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

"""
LaMo SFT Trainer: LaMo training extensions.

Subclasses SFTTrainer to add:
- Adapter and physical weight loading (adapter_weight_dir)
- OpenVid dataset support (dataset_type openvid, _data_root)
- Physical loss and get_trainable_model_parts (LaMo)
- Checkpoint after_save_callback for project-specific integrations
- report_to-aware trackers, run_validation gate
- Extra logging and distributed barriers for stability
"""

import functools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from peft import get_peft_model_state_dict
from tqdm import tqdm

from ... import data, logging, optimizer, parallel, utils
from ...config import TrainingType
from ...state import TrainState

from .trainer import SFTTrainer


logger = logging.get_logger()


def _sync_physical_log_dict(
    accumulated: Dict[str, float],
    device: torch.device,
    use_dist: bool,
    dp_cp_mesh,
) -> Dict[str, float]:
    """Average per-key physical log scalars across distributed ranks (same as global_physical_loss)."""
    if not accumulated:
        return {}
    if use_dist and dp_cp_mesh is not None:
        out: Dict[str, float] = {}
        for k, v in accumulated.items():
            t = parallel.dist_mean(
                torch.tensor([v], device=device, dtype=torch.float32),
                dp_cp_mesh,
            )
            out[k] = t.item() if hasattr(t, "item") else float(t)
        return out
    return dict(accumulated)


class LaMoSFTTrainer(SFTTrainer):
    """SFTTrainer for LaMo, OpenVid, and adapter loading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._after_checkpoint_save_callback = None

    def _init_distributed(self) -> None:
        logger.info("[LaMoSFTTrainer.__init__] _init_distributed()...")
        super()._init_distributed()
        logger.info("[LaMoSFTTrainer.__init__] _init_distributed() done.")

    def run(self) -> None:
        try:
            logger.info("[run] _prepare_models()...")
            self._prepare_models()
            logger.info("[run] _prepare_trainable_parameters()...")
            self._prepare_trainable_parameters()
            logger.info("[run] _prepare_for_training()...")
            self._prepare_for_training()
            logger.info("[run] _prepare_dataset()...")
            self._prepare_dataset()
            logger.info("[run] _prepare_checkpointing()...")
            self._prepare_checkpointing()
            logger.info("[run] _train()...")
            self._train()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.state.parallel_backend.destroy()
            raise e

    def _prepare_models(self) -> None:
        logger.info("Initializing models")
        logger.info("[_prepare_models] Loading diffusion models (transformer, scheduler, etc.)...")
        diffusion_components = self.model_specification.load_diffusion_models()
        logger.info("[_prepare_models] Diffusion models loaded.")
        self._set_components(diffusion_components)

        if self.state.parallel_backend.pipeline_parallel_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet. This will be supported in the future."
            )

    def _prepare_trainable_parameters(self) -> None:
        super()._prepare_trainable_parameters()
        adapter_weight_dir = getattr(self.args, "adapter_weight_dir", None)
        if adapter_weight_dir:
            self._load_adapter_and_physical_weights(adapter_weight_dir)

    def _load_adapter_and_physical_weights(self, adapter_weight_dir: str) -> None:
        """Load LoRA and (for LaMo) physical module from a local exported weight dir."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        weight_dir = self.model_specification.resolve_weight_path(adapter_weight_dir)
        if not weight_dir or not os.path.isdir(weight_dir):
            return
        lora_path = None
        for name in ("adapter_model.safetensors", "pytorch_lora_weights.safetensors"):
            p = os.path.join(weight_dir, name)
            if os.path.isfile(p):
                lora_path = p
                break
        if lora_path and (
            self.args.training_type == TrainingType.LORA
        ):
            state_dict = load_file(lora_path)
            set_peft_model_state_dict(self.transformer, state_dict, adapter_name="default")
            logger.info("Loaded LoRA weights from %s into transformer.", weight_dir)
        if hasattr(self.model_specification, "load_physical_weights"):
            self.model_specification.load_physical_weights(weight_dir)

    def _prepare_for_training(self) -> None:
        # Same as base but use get_trainable_model_parts when available (LaMo physical modules).
        parallel_backend = self.state.parallel_backend
        world_mesh = parallel_backend.get_mesh()
        model_specification = self.model_specification

        if parallel_backend.context_parallel_enabled:
            raise NotImplementedError(
                "Context parallelism is not supported yet. This will be supported in the future."
            )

        if parallel_backend.tensor_parallel_enabled:
            model_specification.apply_tensor_parallel(
                backend=parallel.ParallelBackendEnum.PTD,
                device_mesh=parallel_backend.get_mesh()["tp"],
                transformer=self.transformer,
            )

        if self.args.gradient_checkpointing:
            utils.apply_activation_checkpointing(
                self.transformer, checkpointing_type="full"
            )

        if parallel_backend.data_sharding_enabled:
            if self.args.parallel_backend == "accelerate":
                raise NotImplementedError(
                    "Data sharding is not supported with Accelerate yet."
                )
            if parallel_backend.data_replication_enabled:
                logger.info("Applying HSDP to the model")
            else:
                logger.info("Applying FSDP to the model")
            dp_mesh_names = (
                ("dp_replicate", "dp_shard_cp")
                if (
                    parallel_backend.data_replication_enabled
                    or parallel_backend.context_parallel_enabled
                )
                else ("dp_shard_cp",)
            )
            parallel.apply_fsdp2_ptd(
                model=self.transformer,
                dp_mesh=world_mesh[dp_mesh_names],
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_backend.pipeline_parallel_enabled,
                cpu_offload=False,
            )
        elif parallel_backend.data_replication_enabled:
            logger.info("Applying DDP to the model")
            if world_mesh.ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")
            parallel_backend.apply_ddp(self.transformer, world_mesh)

        self._move_components_to_device()

        if False:
            for name, params in self.transformer.named_parameters():
                if "phys" in name:
                    params.requires_grad_(True)

        if self.args.training_type == TrainingType.LORA:
            for name, params in self.transformer.named_parameters():
                if "physical_global_proj" in name or "phys_spatial_proj" in name:
                    params.requires_grad_(True)

        model_parts = (
            self.model_specification.get_trainable_model_parts(self.transformer)
            if hasattr(self.model_specification, "get_trainable_model_parts")
            else [self.transformer]
        )
        self.state.num_trainable_parameters = sum(
            p.numel() for m in model_parts for p in m.parameters() if p.requires_grad
        )

        logger.info("Initializing optimizer and lr scheduler")
        self.state.train_state = TrainState()
        self.optimizer = optimizer.get_optimizer(
            parallel_backend=self.args.parallel_backend,
            name=self.args.optimizer,
            model_parts=model_parts,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            fused=False,
        )
        self.lr_scheduler = optimizer.get_lr_scheduler(
            parallel_backend=self.args.parallel_backend,
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
        )

        # LaMo: override physical module optimizer with independent learning rate & scheduler
        self._physical_separate_lr = False
        phys_lr = getattr(self.model_specification, "physical_lr", None)
        if phys_lr is not None and len(model_parts) > 1:
            if isinstance(self.optimizer, optimizer.OptimizerWrapper) and len(self.optimizer.optimizers) > 1:
                phys_opt = self.optimizer.optimizers[-1]
                for pg in phys_opt.param_groups:
                    pg["lr"] = phys_lr
                    pg["initial_lr"] = phys_lr

                if isinstance(self.lr_scheduler, optimizer.SchedulerWrapper) and len(self.lr_scheduler.schedulers) > 1:
                    phys_sched_name = getattr(self.model_specification, "predictor_lr_scheduler", None)
                    if phys_sched_name and phys_sched_name != self.args.lr_scheduler:
                        phys_lambda_fn = optimizer.get_lr_scheduler_lambda(
                            phys_sched_name,
                            num_warmup_steps=self.args.lr_warmup_steps,
                            num_training_steps=self.args.train_steps,
                        )
                        import torch as _torch
                        self.lr_scheduler.schedulers[-1] = _torch.optim.lr_scheduler.LambdaLR(
                            phys_opt, phys_lambda_fn, last_epoch=-1,
                        )
                        logger.info(
                            "Physical module LR scheduler overridden: %s (main: %s)",
                            phys_sched_name, self.args.lr_scheduler,
                        )
                    else:
                        phys_sched = self.lr_scheduler.schedulers[-1]
                        phys_sched.base_lrs = [phys_lr] * len(phys_sched.base_lrs)

                self._physical_separate_lr = True
                logger.info(
                    "Physical module independent optimizer: lr=%.2e (main lr=%.2e)",
                    phys_lr, self.args.lr,
                )

        self.optimizer, self.lr_scheduler = parallel_backend.prepare_optimizer(
            self.optimizer, self.lr_scheduler
        )

        self._init_logging()
        self._init_trackers()
        self._init_directories_and_repositories()

    def _prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        with open(self.args.dataset_config, "r") as file:
            dataset_configs = json.load(file)["datasets"]
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")

        datasets = []
        for config in dataset_configs:
            data_root = config.pop("data_root", None)
            dataset_file = config.pop("dataset_file", None)
            dataset_type = config.pop("dataset_type")
            caption_options = config.pop("caption_options", {})

            if data_root is not None and dataset_file is not None and dataset_type != "openvid":
                raise ValueError(
                    "Both data_root and dataset_file cannot be provided in the same dataset config (except for dataset_type 'openvid')."
                )

            if dataset_type == "openvid":
                if dataset_file is None:
                    raise ValueError("dataset_type 'openvid' requires dataset_file (path to OpenVid CSV).")
                dataset_name_or_root = dataset_file
                data_root_for_openvid = data_root or str(Path(dataset_file).parent)
            else:
                dataset_name_or_root = data_root or dataset_file
                data_root_for_openvid = None

            dataset = data.initialize_dataset(
                dataset_name_or_root,
                dataset_type,
                streaming=True,
                infinite=True,
                model_name=self.args.model_name,
                enable_precomputation=self.args.enable_precomputation,
                precompute_root=self.args.precomputation_dir,
                _caption_options=caption_options,
                _data_root=data_root_for_openvid,
            )

            if not dataset._precomputable_once and self.args.precomputation_once:
                raise ValueError(
                    f"Dataset {dataset_name_or_root} does not support precomputing all embeddings at once."
                )

            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            preprocessing_type = "video" if dataset_type == "openvid" else dataset_type
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, preprocessing_type, config)
            datasets.append(dataset)

        dataset = data.combine_datasets(
            datasets, buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True
        )
        dataloader = self.state.parallel_backend.prepare_dataloader(
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.pin_memory,
        )

        self.dataset = dataset
        self.dataloader = dataloader

    def _prepare_checkpointing(self) -> None:
        parallel_backend = self.state.parallel_backend

        def save_model_hook(state_dict: Dict[str, Any], step) -> None:
            if parallel_backend.is_main_process:
                # PTDCheckpointManager invokes this callback once per trainable
                # model part. For LaMo specs whose `get_trainable_model_parts`
                # returns extra non-transformer modules (e.g. the LaMo predictor),
                # the corresponding state_dict must NOT be routed to
                # `_save_model` / `_save_lora_weights` (which would try to
                # strict-load it into `CogVideoXTransformer3DModel`). The LaMo
                # spec already persists those weights (with the correct version
                # prefix) on the transformer pass via `_save_physical_weights`,
                # so we detect and skip them here.
                if hasattr(self.model_specification, "get_trainable_model_parts"):
                    extra_parts = self.model_specification.get_trainable_model_parts(self.transformer)[1:]
                    sd_keys = set(state_dict.keys())
                    for _part in extra_parts:
                        if _part is None:
                            continue
                        part_keys = set(_part.state_dict().keys())
                        if part_keys and sd_keys == part_keys:
                            parallel_backend.wait_for_everyone()
                            return

                _PHYS_PREFIXES = ("projection.", "dynamics.", "encoder.", "recon_decoder.", "prior.")
                is_physical_state = any(
                    any(k.startswith(pfx) for pfx in _PHYS_PREFIXES) for k in state_dict.keys()
                )
                if is_physical_state:
                    if hasattr(self.model_specification, "_get_physical_state_dict"):
                        # LaMo models: _save_lora_weights / _save_model already saves
                        # physical weights with correct version prefix — skip here to
                        # avoid overwriting that file with un-prefixed keys.
                        pass
                    elif hasattr(self.model_specification, "_save_physical_weights"):
                        output_dir = os.path.join(
                            self.args.output_dir, f"{self.trainer_prefix}_{step}"
                        )
                        self.model_specification._save_physical_weights(output_dir, state_dict)
                elif self.args.training_type == TrainingType.LORA:
                    # Extract physical_global_proj weights before PEFT filtering
                    # (PEFT only keeps LoRA adapter params; physical_global_proj is a
                    # non-LoRA trainable module in LaMo transformers)
                    phys_proj_state = {}
                    for k, v in state_dict.items():
                        for _pname in ("physical_global_proj", "phys_spatial_proj"):
                            if _pname in k:
                                idx = k.index(_pname)
                                phys_proj_state[k[idx:]] = v
                                break

                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    if not state_dict and not phys_proj_state:
                        pass
                    else:
                        output_dir = os.path.join(
                            self.args.output_dir, f"{self.trainer_prefix}_{step}"
                        )
                        self.model_specification._save_lora_weights(
                            output_dir, state_dict, self.scheduler,
                            _extra_physical_state=phys_proj_state if phys_proj_state else None,
                        )
                elif False:
                    self.model_specification._save_lora_weights(
                        self.args.output_dir, step, self.transformer, state_dict, self.scheduler
                    )
                elif self.args.training_type == TrainingType.FULL_FINETUNE:
                    self.model_specification._save_model(
                        self.args.output_dir, self.transformer, state_dict, self.scheduler
                    )
            parallel_backend.wait_for_everyone()

        enable_state_checkpointing = self.args.checkpointing_steps > 0
        model_parts_for_ckpt = (
            self.model_specification.get_trainable_model_parts(self.transformer)
            if hasattr(self.model_specification, "get_trainable_model_parts")
            else [self.transformer]
        )
        self.checkpointer = utils.PTDCheckpointManager(
            dataloader=self.dataloader,
            model_parts=model_parts_for_ckpt,
            optimizers=self.optimizer,
            schedulers=self.lr_scheduler,
            states={"train_state": self.state.train_state},
            checkpointing_steps=self.args.checkpointing_steps,
            checkpointing_limit=self.args.checkpointing_limit,
            output_dir=self.args.output_dir,
            enable=enable_state_checkpointing,
            _callback_fn=save_model_hook,
            after_save_callback=getattr(self, "_after_checkpoint_save_callback", None),
        )

        resume_from_checkpoint = self.args.resume_from_checkpoint
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = -1
        if resume_from_checkpoint is not None:
            self.checkpointer.load(resume_from_checkpoint)

    def set_after_checkpoint_save_callback(self, callback) -> None:
        """Set a callback (checkpoint_dir_path: str, step: int) -> None to run after each checkpoint save."""
        self._after_checkpoint_save_callback = callback

    def _train(self) -> None:
        logger.info("Starting training")

        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        device = parallel_backend.device

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        global_batch_size = self.args.batch_size * parallel_backend._dp_degree
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-replica batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=train_state.step,
            desc="Training steps",
            disable=not parallel_backend.is_local_main_process,
        )

        generator = torch.Generator(device=device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        patch_size = 1
        if (
            getattr(self.transformer.config, "patch_size", None) is not None
            and getattr(self.transformer.config, "patch_size_t", None) is not None
        ):
            patch_size = self.transformer.config.patch_size * self.transformer.config.patch_size_t
        elif isinstance(getattr(self.transformer.config, "patch_size", None), int):
            patch_size = self.transformer.config.patch_size
        elif isinstance(getattr(self.transformer.config, "patch_size", None), (list, tuple)):
            patch_size = math.prod(self.transformer.config.patch_size)

        scheduler_sigmas = utils.get_scheduler_sigmas(self.scheduler)
        scheduler_sigmas = (
            scheduler_sigmas.to(device=device, dtype=torch.float32)
            if scheduler_sigmas is not None
            else None
        )
        scheduler_alphas = utils.get_scheduler_alphas(self.scheduler)
        scheduler_alphas = (
            scheduler_alphas.to(device=device, dtype=torch.float32)
            if scheduler_alphas is not None
            else None
        )
        timesteps_buffer = []

        self.transformer.train()
        data_iterator = iter(self.dataloader)

        processor_fn = {
            "condition": self.model_specification.prepare_conditions,
            "latent": functools.partial(
                self.model_specification.prepare_latents,
                compute_posterior=not self.args.precomputation_once,
            ),
        }

        if False:
            processor_fn["priori"] = self.model_specification.prepare_priori_or_quantify_priori
            processor_fn["quantify_priori"] = (
                self.model_specification.prepare_priori_or_quantify_priori
            )

        preprocessor = data.initialize_preprocessor(
            rank=parallel_backend.rank,
            num_items=self.args.precomputation_items if self.args.enable_precomputation else 1,
            processor_fn=processor_fn,
            save_dir=self.args.precomputation_dir,
            enable_precomputation=self.args.enable_precomputation,
            hash_save=self.args.hash_save,
            model_name=self.args.model_name,
        )
        precomputed_condition_iterator: Iterable[Dict[str, Any]] = None
        precomputed_latent_iterator: Iterable[Dict[str, Any]] = None
        sampler = data.ResolutionSampler(
            batch_size=self.args.batch_size,
            dim_keys=self.model_specification._resolution_dim_keys,
        )
        requires_gradient_step = True
        accumulated_loss = 0.0
        if False:
            accumulated_aux_loss = 0.0
        accumulated_physical_loss = 0.0
        accumulated_physical_logs: Dict[str, float] = {}

        if parallel_backend.data_replication_enabled or parallel_backend.data_sharding_enabled:
            logger.info(
                f"rank {parallel_backend.rank} / world_size {parallel_backend.world_size} entering training loop"
            )
            parallel_backend.wait_for_everyone()

        while (
            train_state.step < self.args.train_steps
            and train_state.observed_data_samples < self.args.max_data_samples
        ):
            if preprocessor.requires_data:
                if False:
                    (
                        precomputed_condition_iterator,
                        precomputed_latent_iterator,
                        priori_iterator,
                        quantify_priori_iterator,
                    ) = self._prepare_data(preprocessor, data_iterator)
                else:
                    precomputed_condition_iterator, precomputed_latent_iterator = self._prepare_data(
                        preprocessor, data_iterator
                    )

            try:
                condition_item = next(precomputed_condition_iterator)
                latent_item = next(precomputed_latent_iterator)
                if False:
                    priori_item = next(priori_iterator)
                    quantify_priori_item = next(quantify_priori_iterator)
                    sampler.consume(
                        condition_item,
                        latent_item,
                        priori_item,
                        quantify_priori_item,
                    )
                else:
                    sampler.consume(condition_item, latent_item)
            except StopIteration:
                if requires_gradient_step:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    requires_gradient_step = False
                logger.info("Data exhausted. Exiting training loop.")
                break

            if sampler.is_ready:
                if False:
                    (
                        condition_batch,
                        latent_batch,
                        priori_batch,
                        quantify_priori_batch,
                    ) = sampler.get_batch()
                    priori_conditions = self.model_specification.collate_conditions(
                        priori_batch
                    )
                    quantify_priori_conditions = (
                        self.model_specification.collate_conditions(quantify_priori_batch)
                    )
                else:
                    condition_batch, latent_batch = sampler.get_batch()
                condition_model_conditions = self.model_specification.collate_conditions(
                    condition_batch
                )
                latent_model_conditions = self.model_specification.collate_latents(
                    latent_batch
                )
            else:
                continue

            if False:
                condition_model_conditions = {
                    **condition_model_conditions,
                    **priori_conditions,
                    **quantify_priori_conditions,
                }

            if (
                (
                    parallel_backend.data_replication_enabled
                    or parallel_backend.data_sharding_enabled
                )
                and train_state.step == 0
            ):
                parallel_backend.wait_for_everyone()

            train_state.step += 1
            train_state.observed_data_samples += (
                self.args.batch_size * parallel_backend._dp_degree
            )

            lmc_latents = latent_model_conditions["latents"]
            train_state.observed_num_tokens += (
                math.prod(lmc_latents.shape[:-1]) // patch_size
            )

            logger.debug(
                f"Starting training step ({train_state.step}/{self.args.train_steps})"
            )

            utils.align_device_and_dtype(
                latent_model_conditions, device, self.args.transformer_dtype
            )
            utils.align_device_and_dtype(
                condition_model_conditions, device, self.args.transformer_dtype
            )
            latent_model_conditions = utils.make_contiguous(latent_model_conditions)
            condition_model_conditions = utils.make_contiguous(
                condition_model_conditions
            )

            sigmas = utils.prepare_sigmas(
                scheduler=self.scheduler,
                sigmas=scheduler_sigmas,
                batch_size=self.args.batch_size,
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                flow_weighting_scheme=self.args.flow_weighting_scheme,
                flow_logit_mean=self.args.flow_logit_mean,
                flow_logit_std=self.args.flow_logit_std,
                flow_mode_scale=self.args.flow_mode_scale,
                device=device,
                generator=self.state.generator,
            )
            sigmas = utils.expand_tensor_dims(
                sigmas, latent_model_conditions["latents"].ndim
            )

            forward_result = self.model_specification.forward(
                transformer=self.transformer,
                scheduler=self.scheduler,
                condition_model_conditions=condition_model_conditions,
                latent_model_conditions=latent_model_conditions,
                sigmas=sigmas,
                compute_posterior=not self.args.precomputation_once,
                training_step=train_state.step,
            )
            pred = forward_result[0]
            target = forward_result[1]
            sigmas = forward_result[2]
            forward_aux = forward_result[3] if len(forward_result) > 3 else None

            if (
                False
                and self.transformer.whether_classifier
            ):
                pred, aux_loss = pred[0], pred[1]

            timesteps = (sigmas * 1000.0).long()
            weights = utils.prepare_loss_weights(
                scheduler=self.scheduler,
                alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                sigmas=sigmas,
                flow_weighting_scheme=self.args.flow_weighting_scheme,
            )
            pred_for_loss = pred[0] if isinstance(pred, (list, tuple)) else pred
            weights = utils.expand_tensor_dims(weights, pred_for_loss.ndim)

            loss = weights.float() * (pred_for_loss.float() - target.float()).pow(2)
            loss = loss.mean(list(range(1, loss.ndim)))
            loss = loss.mean()

            if (
                False
                and self.transformer.whether_classifier
            ):
                if aux_loss > 1.0:
                    loss = loss + ((1 + aux_loss) / (1 + aux_loss.detach() + 1e-5)) * 0.10
                else:
                    loss = (
                        loss
                        + ((1 + aux_loss) / (1 + aux_loss.detach() + 1e-5))
                        * 0.05
                        * aux_loss.detach()
                    )
                accumulated_aux_loss += aux_loss.detach().item()

            if forward_aux is not None and "physical_loss" in forward_aux:
                if self._physical_separate_lr:
                    lambda_phys = 1.0
                else:
                    lambda_phys = getattr(
                        self.model_specification, "lambda_phys", 0.01
                    )
                loss = loss + lambda_phys * forward_aux["physical_loss"]
                accumulated_physical_loss += forward_aux["physical_loss"].detach().item()
                pl = forward_aux.get("physical_logs")
                if isinstance(pl, dict):
                    for k, v in pl.items():
                        fv = (
                            float(v)
                            if not isinstance(v, torch.Tensor)
                            else v.detach().float().item()
                        )
                        accumulated_physical_logs[k] = (
                            accumulated_physical_logs.get(k, 0.0) + fv
                        )

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.detach().item()
            requires_gradient_step = True

            model_parts = (
                self.model_specification.get_trainable_model_parts(self.transformer)
                if hasattr(self.model_specification, "get_trainable_model_parts")
                else [self.transformer]
            )
            if self._physical_separate_lr and len(model_parts) > 1:
                main_params = [p for m in model_parts[:-1] for p in m.parameters()]
                phys_params = list(model_parts[-1].parameters())
                grad_norm = utils.torch._clip_grad_norm_while_handling_failing_dtensor_cases(
                    main_params,
                    self.args.max_grad_norm,
                    foreach=True,
                    pp_mesh=(
                        parallel_backend.get_mesh("pp")
                        if parallel_backend.pipeline_parallel_enabled
                        else None
                    ),
                )
                torch.nn.utils.clip_grad_norm_(phys_params, self.args.max_grad_norm)
            else:
                grad_norm = utils.torch._clip_grad_norm_while_handling_failing_dtensor_cases(
                    [p for m in model_parts for p in m.parameters()],
                    self.args.max_grad_norm,
                    foreach=True,
                    pp_mesh=(
                        parallel_backend.get_mesh("pp")
                        if parallel_backend.pipeline_parallel_enabled
                        else None
                    ),
                )

            logs = {}

            if train_state.step % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if grad_norm is not None:
                    logs["grad_norm"] = (
                        grad_norm
                        if isinstance(grad_norm, float)
                        else grad_norm.detach().item()
                    )
                if (
                    parallel_backend.data_replication_enabled
                    or parallel_backend.data_sharding_enabled
                    or parallel_backend.context_parallel_enabled
                ):
                    dp_cp_mesh = parallel_backend.get_mesh("dp_cp")
                    global_avg_loss, global_max_loss = (
                        parallel.dist_mean(
                            torch.tensor([accumulated_loss], device=device),
                            dp_cp_mesh,
                        ),
                        parallel.dist_max(
                            torch.tensor([accumulated_loss], device=device),
                            dp_cp_mesh,
                        ),
                    )
                    if (
                        False
                        and self.transformer.whether_classifier
                    ):
                        global_aux_loss = parallel.dist_mean(
                            torch.tensor([accumulated_aux_loss], device=device),
                            dp_cp_mesh,
                        )
                        accumulated_aux_loss = 0.0
                    else:
                        global_aux_loss = None
                    global_physical_loss = parallel.dist_mean(
                        torch.tensor([accumulated_physical_loss], device=device),
                        dp_cp_mesh,
                    )
                    global_physical_logs = _sync_physical_log_dict(
                        accumulated_physical_logs, device, True, dp_cp_mesh
                    )
                else:
                    global_avg_loss = global_max_loss = accumulated_loss
                    if (
                        False
                        and self.transformer.whether_classifier
                    ):
                        global_aux_loss = accumulated_aux_loss
                        accumulated_aux_loss = 0.0
                    else:
                        global_aux_loss = None
                    global_physical_loss = accumulated_physical_loss
                    global_physical_logs = _sync_physical_log_dict(
                        accumulated_physical_logs, device, False, None
                    )

                accumulated_physical_loss = 0.0
                accumulated_physical_logs = {}
                logs["global_avg_loss"] = global_avg_loss
                if global_aux_loss is not None:
                    logs["global_aux_loss"] = (
                        global_aux_loss
                        if isinstance(global_aux_loss, float)
                        else global_aux_loss.detach().item()
                    )
                logs["global_max_loss"] = global_max_loss
                logs["global_physical_loss"] = (
                    global_physical_loss
                    if isinstance(global_physical_loss, float)
                    else global_physical_loss.detach().item()
                )
                if global_physical_logs:
                    for _k, _v in global_physical_logs.items():
                        logs[f"global_phys_{_k}"] = _v
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_aux_losses.append(
                    global_aux_loss if global_aux_loss is not None else 0.0
                )
                train_state.global_max_losses.append(global_max_loss)
                accumulated_loss = 0.0
                requires_gradient_step = False

            progress_bar.update(1)

            def _to_float(x):
                if x is None:
                    return None
                return x if isinstance(x, float) else x.detach().item()

            if logs:
                postfix_short = {}
                if "grad_norm" in logs:
                    postfix_short["grad_norm"] = _to_float(logs["grad_norm"])
                if "global_avg_loss" in logs:
                    postfix_short["avg_loss"] = _to_float(logs["global_avg_loss"])
                if "global_max_loss" in logs:
                    postfix_short["max_loss"] = _to_float(logs["global_max_loss"])
                if "global_physical_loss" in logs:
                    postfix_short["phys_loss"] = _to_float(
                        logs["global_physical_loss"]
                    )
                for _lk, _lv in logs.items():
                    if _lk.startswith("global_phys_"):
                        postfix_short[_lk.replace("global_phys_", "p_", 1)] = (
                            _to_float(_lv)
                        )
                if "global_aux_loss" in logs:
                    postfix_short["aux_loss"] = _to_float(logs["global_aux_loss"])
                progress_bar.set_postfix(postfix_short)
                _gn = _to_float(logs.get("grad_norm"))
                _ga = _to_float(logs.get("global_avg_loss"))
                _gm = _to_float(logs.get("global_max_loss"))
                _gp = _to_float(logs.get("global_physical_loss"))
                _phys_extra = ""
                _phys_items = sorted(
                    (k, v) for k, v in logs.items() if k.startswith("global_phys_")
                )
                if _phys_items:
                    _phys_extra = " | " + " ".join(
                        f"{k[len('global_phys_'):]}={float(v):.6g}"
                        for k, v in _phys_items
                    )
                logger.info(
                    "step %s | grad_norm=%s avg_loss=%s max_loss=%s phys_loss=%s%s",
                    train_state.step,
                    f"{_gn:.4f}" if _gn is not None else "n/a",
                    f"{_ga:.4f}" if _ga is not None else "n/a",
                    f"{_gm:.4f}" if _gm is not None else "n/a",
                    f"{_gp:.4f}" if _gp is not None else "n/a",
                    _phys_extra,
                )
            else:
                progress_bar.set_postfix(logs)

            timesteps_buffer.extend(
                [
                    (train_state.step, t)
                    for t in timesteps.detach().cpu().numpy().tolist()
                ]
            )

            if train_state.step % self.args.logging_steps == 0:
                timesteps_buffer = []
                logs["observed_data_samples"] = train_state.observed_data_samples
                logs["observed_num_tokens"] = train_state.observed_num_tokens
                parallel_backend.log(logs, step=train_state.step)
                train_state.log_steps.append(train_state.step)

            self.checkpointer.save(
                step=train_state.step,
                _device=device,
                _is_main_process=parallel_backend.is_main_process,
            )

            if train_state.step % self.args.validation_steps == 0:
                self._validate(step=train_state.step, final_validation=False)

        self.checkpointer.save(
            train_state.step,
            force=True,
            _device=device,
            _is_main_process=parallel_backend.is_main_process,
        )
        parallel_backend.wait_for_everyone()
        self._validate(step=train_state.step, final_validation=True)

        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(
            f"Memory after training end: {json.dumps(memory_statistics, indent=4)}"
        )

        if parallel_backend.is_main_process and self.args.push_to_hub:
            from huggingface_hub import upload_folder

            upload_folder(
                repo_id=self.state.repo_id,
                folder_path=self.args.output_dir,
                ignore_patterns=[f"{self.checkpointer._prefix}_*"],
            )

        parallel_backend.destroy()

    def _validate(self, step: int, final_validation: bool = False) -> None:
        if not getattr(self.args, "run_validation", True) or self.args.validation_dataset_file is None:
            return
        super()._validate(step=step, final_validation=final_validation)

    def _init_trackers(self) -> None:
        if getattr(self.args, "report_to", None) == "wandb":
            trackers = ["wandb"]
        else:
            trackers = []
        experiment_name = self.args.tracker_name or "finetrainers-experiment"
        self.state.parallel_backend.initialize_trackers(
            trackers,
            experiment_name=experiment_name,
            config=self._get_training_info(),
            log_dir=self.args.logging_dir,
        )
