# Copyright (c) 2026 Applied Intuition, Inc.
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

import os
import sys
import traceback

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from finetrainers import BaseArgs, LaMoSFTTrainer, TrainingType, get_logger
from finetrainers.config import _get_model_specifiction_cls
from finetrainers.trainer.sft_trainer.config import SFTFullRankConfig, SFTLowRankConfig


logger = get_logger()


def main():
    try:
        import multiprocessing

        # set_start_method() can only be called once; cluster/torchrun may have already set it.
        method = multiprocessing.get_start_method(allow_none=True)
        if method is None:
            multiprocessing.set_start_method("fork")
        elif method != "fork":
            logger.info(
                "Multiprocessing start method already set to %r (e.g. by cluster launcher). "
                "Using it as-is; 'fork' can reduce dataloader memory but is not required.",
                method,
            )
    except Exception as e:
        logger.warning(
            'Could not set multiprocessing start method to "fork": %s. '
            "Training will continue; dataloader workers may use more memory.",
            e,
        )

    try:
        args = BaseArgs()
        argv = [y.strip() for x in sys.argv for y in x.split()]
        training_type_index = argv.index("--training_type")
        if training_type_index == -1:
            raise ValueError("Training type not provided in command line arguments.")

        training_type = argv[training_type_index + 1]
        training_cls = None
        if training_type == TrainingType.LORA:
            training_cls = SFTLowRankConfig
        elif training_type == TrainingType.FULL_FINETUNE:
            training_cls = SFTFullRankConfig
        else:
            raise ValueError(f"Training type {training_type} not supported.")

        training_config = training_cls()
        args.extend_args(training_config.add_args, training_config.map_args, training_config.validate_args)
        args = args.parse_args()
        logger.info("[startup] Args parsed.")

        from utils.model_path_resolver import resolve_pretrained_model_path
        logger.info("[startup] Resolving pretrained model path...")
        pretrained_path = resolve_pretrained_model_path(args.pretrained_model_name_or_path)
        logger.info("[startup] Resolved pretrained path -> %s", pretrained_path)

        logger.info("[startup] Building model specification...")
        model_specification_cls = _get_model_specifiction_cls(args.model_name, args.training_type)
        spec_kwargs = dict(
            pretrained_model_name_or_path=pretrained_path,
            tokenizer_id=args.tokenizer_id,
            tokenizer_2_id=args.tokenizer_2_id,
            tokenizer_3_id=args.tokenizer_3_id,
            text_encoder_id=args.text_encoder_id,
            text_encoder_2_id=args.text_encoder_2_id,
            text_encoder_3_id=args.text_encoder_3_id,
            transformer_id=args.transformer_id,
            vae_id=args.vae_id,
            text_encoder_dtype=args.text_encoder_dtype,
            text_encoder_2_dtype=args.text_encoder_2_dtype,
            text_encoder_3_dtype=args.text_encoder_3_dtype,
            transformer_dtype=args.transformer_dtype,
            vae_dtype=args.vae_dtype,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        if args.model_name == "lamo":
            spec_kwargs["predictor_in_channels"] = getattr(args, "predictor_in_channels", 16)
            spec_kwargs["predictor_hidden_channels"] = getattr(args, "predictor_hidden_channels", 256)
            spec_kwargs["predictor_num_res_blocks"] = getattr(args, "predictor_num_res_blocks", 8)
            spec_kwargs["predictor_use_se"] = bool(getattr(args, "predictor_use_se", 1))
            spec_kwargs["predictor_use_prev_delta"] = bool(getattr(args, "predictor_use_prev_delta", 0))
            spec_kwargs["predictor_use_prompt_cond"] = bool(getattr(args, "predictor_use_prompt_cond", 1))
            spec_kwargs["prompt_text_dim"] = getattr(args, "prompt_text_dim", 4096)
            spec_kwargs["prompt_cond_dim"] = getattr(args, "prompt_cond_dim", 128)
            spec_kwargs["prompt_dropout_prob"] = getattr(args, "prompt_dropout_prob", 0.1)
            spec_kwargs["predictor_lr"] = getattr(args, "predictor_lr", 1e-3)
            spec_kwargs["predictor_lr_scheduler"] = getattr(args, "predictor_lr_scheduler", "cosine")
            spec_kwargs["predictor_noise_aug_prob"] = getattr(args, "predictor_noise_aug_prob", 0.5)
            spec_kwargs["predictor_noise_aug_scale"] = getattr(args, "predictor_noise_aug_scale", 1.0)
            spec_kwargs["predictor_cosine_weight"] = getattr(args, "predictor_cosine_weight", 0.5)
            spec_kwargs["enable_denoiser_motion_loss"] = bool(getattr(args, "enable_denoiser_motion_loss", 0))
            spec_kwargs["lambda_bmv_rel_l2"] = getattr(args, "lambda_bmv_rel_l2", 0.0)
            spec_kwargs["bmv_tau"] = getattr(args, "bmv_tau", 2)
            spec_kwargs["guidance_lambda"] = getattr(args, "guidance_lambda", 15.0)
            spec_kwargs["guidance_step_ratio"] = getattr(args, "guidance_step_ratio", 0.8)
        model_specification = model_specification_cls(**spec_kwargs)

        if args.training_type in [TrainingType.LORA, TrainingType.FULL_FINETUNE]:
            logger.info("[startup] Creating LaMoSFTTrainer (distributed init)...")
            trainer = LaMoSFTTrainer(args, model_specification)
            if getattr(args, "output_dir_sync_path", None):
                raise ValueError(
                    "--output_dir_sync_path is not supported in the "
                    "open-source release. Use --output_dir with a local path "
                    "such as <PATH_TO_OUTPUT_DIR>."
                )
            logger.info("[startup] LaMoSFTTrainer created. Starting run()...")
        else:
            raise ValueError(f"Training type {args.training_type} not supported.")

        trainer.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
        raise
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
