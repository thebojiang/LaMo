#!/usr/bin/env python3

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

"""Internal torchrun entrypoint for local VBench evaluation.

Users should launch ``eval/scripts/run_vbench_eval.sh``. This helper exists
because torchrun starts one Python file per rank.
"""

from __future__ import annotations

import os
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.cluster_vbench_eval import main


def _get_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def _get_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def build_config() -> dict:
    cfg = {
        "name": os.environ["RUN_TAG"],
        "model_type": os.environ.get("MODEL_TYPE", "lamo"),
        "generate_type": os.environ.get("GENERATE_TYPE", "baseline"),
        "pretrained_model_path": os.environ["BASE_MODEL"],
        "checkpoint_path": os.environ.get("CHECKPOINT_PATH", ""),
        "result_path": os.environ["RESULT_PATH"],
        "prompt_info_json": os.environ["PROMPT_INFO_JSON"],
        "vbench_root": os.environ["VBENCH_ROOT"],
        "vbench_pretrained_path": os.environ.get("VBENCH_PRETRAINED_PATH", ""),
        "num_gpus": _get_int("NUM_GPUS", "1"),
        "workers_per_rank": _get_int("WORKERS_PER_RANK", "1"),
        "num_samples": _get_int("NUM_SAMPLES", "5"),
        "num_frames": _get_int("NUM_FRAMES", "49"),
        "fps": _get_int("FPS", "8"),
        "num_inference_steps": _get_int("NUM_INFERENCE_STEPS", "50"),
        "guidance_scale": _get_float("GUIDANCE_SCALE", "6.0"),
        "phys_guidance_scale": _get_float("PHYS_GUIDANCE_SCALE", "3.0"),
        "seed": _get_int("SEED", "42"),
        "guidance_lambda": _get_float("GUIDANCE_LAMBDA", "15.0"),
        "guidance_step_ratio": _get_float("GUIDANCE_STEP_RATIO", "0.8"),
        "predictor_hidden_channels": _get_int("PREDICTOR_HIDDEN_CHANNELS", "256"),
        "predictor_num_res_blocks": _get_int("PREDICTOR_NUM_RES_BLOCKS", "8"),
        "predictor_use_se": _get_int("PREDICTOR_USE_SE", "1"),
        "predictor_use_prev_delta": _get_int("PREDICTOR_USE_PREV_DELTA", "0"),
        "predictor_use_prompt_cond": _get_int("PREDICTOR_USE_PROMPT_COND", "1"),
        "prompt_text_dim": _get_int("PROMPT_TEXT_DIM", "4096"),
        "prompt_cond_dim": _get_int("PROMPT_COND_DIM", "128"),
        "generation_timeout_seconds": _get_int("GENERATION_TIMEOUT_SECONDS", "72000"),
        "generation_sync_timeout_seconds": _get_int("GENERATION_SYNC_TIMEOUT_SECONDS", "72000"),
        "vbench_eval_timeout_seconds": _get_int("VBENCH_EVAL_TIMEOUT_SECONDS", "14400"),
        "vbench_eval_gpus": _get_int("VBENCH_EVAL_GPUS", "1"),
        "skip_existing": _get_int("SKIP_EXISTING", "1"),
        "skip_detectron2_install": _get_int("SKIP_DETECTRON2_INSTALL", "0"),
        "skip_vbench_pretrained_download": _get_int("SKIP_VBENCH_PRETRAINED_DOWNLOAD", "0"),
    }

    if os.environ.get("DIMENSIONS"):
        cfg["dimensions"] = os.environ["DIMENSIONS"]
    if os.environ.get("EVAL_LIMIT"):
        cfg["limit"] = int(os.environ["EVAL_LIMIT"])
    if os.environ.get("VBENCH_EVAL_SCRIPT"):
        cfg["vbench_eval_script"] = os.environ["VBENCH_EVAL_SCRIPT"]
    if os.environ.get("VBENCH_DIST_BACKEND"):
        cfg["vbench_dist_backend"] = os.environ["VBENCH_DIST_BACKEND"]
    if os.environ.get("DETECTRON2_WHEEL_PATH"):
        cfg["detectron2_wheel_path"] = os.environ["DETECTRON2_WHEEL_PATH"]
    if os.environ.get("DETECTRON2_INSTALL_SPEC"):
        cfg["detectron2_install_spec"] = os.environ["DETECTRON2_INSTALL_SPEC"]

    if os.environ.get("LORA_NAME"):
        cfg["lora_name"] = os.environ["LORA_NAME"]
    if os.environ.get("LORA_RANK"):
        cfg["lora_rank"] = int(os.environ["LORA_RANK"])
    if os.environ.get("LORA_ALPHA"):
        cfg["lora_alpha"] = int(os.environ["LORA_ALPHA"])

    return cfg


if __name__ == "__main__":
    main(build_config())
