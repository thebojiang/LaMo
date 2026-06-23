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

"""Internal torchrun entrypoint for local VideoPhy evaluation.

Users should launch ``eval/scripts/run_videophy_eval.sh``.  This helper exists
because torchrun starts one Python file per rank.
"""

from __future__ import annotations

import os
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.cluster_eval import main


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
        "checkpoint_path": os.environ["CHECKPOINT_PATH"],
        "result_path": os.environ["RESULT_PATH"],
        "input_csv": os.environ["INPUT_CSV"],
        "videocon_checkpoint": os.environ["VIDEOCON_CHECKPOINT"],
        "videophy_repo_path": os.environ["VIDEOPHY_REPO_PATH"],
        "num_gpus": _get_int("NUM_GPUS", "1"),
        "workers_per_rank": _get_int("WORKERS_PER_RANK", "1"),
        "num_frames": _get_int("NUM_FRAMES", "49"),
        "fps": _get_int("FPS", "8"),
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
        "entailment_eval_timeout_seconds": _get_int("ENTAILMENT_EVAL_TIMEOUT_SECONDS", "7200"),
        "skip_existing": _get_int("SKIP_EXISTING", "1"),
    }

    if os.environ.get("EVAL_LIMIT"):
        cfg["limit"] = int(os.environ["EVAL_LIMIT"])

    return cfg


if __name__ == "__main__":
    main(build_config())
