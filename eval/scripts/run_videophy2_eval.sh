#!/usr/bin/env bash

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

set -euo pipefail

# Public local VideoPhy2 evaluation entrypoint. Supports single-GPU and
# single-node multi-GPU generation before running VideoPhy2 metrics on rank 0.
#
# Common overrides:
#   BASE_MODEL=/path/to/CogVideoX-2b-Diffusers
#   CHECKPOINT_PATH=/path/to/LaMo-CogVideoX-2b
#   MODEL_TYPE=lamo|cogvideox
#   GENERATE_TYPE=baseline|lora
#   INPUT_CSV=/path/to/videophy2.csv
#   VIDEOPHY2_CHECKPOINT=/path/to/videophy_2_auto
#   RUN_TAG=name or RESULT_PATH=/path/to/output
#   RESUME_GENERATION=1
#   WORKERS_PER_RANK=1
#   EVAL_LIMIT=N
#   CONDA_ENV=lamo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" && -z "${NUM_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
elif [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

if [[ -z "${NUM_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "all" ]]; then
    IFS=',' read -r -a CUDA_VISIBLE_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS="${#CUDA_VISIBLE_DEVICE_LIST[@]}"
  else
    NUM_GPUS=1
  fi
fi
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "[run_videophy2] NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "all" ]]; then
  IFS=',' read -r -a CUDA_VISIBLE_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "$NUM_GPUS" -gt "${#CUDA_VISIBLE_DEVICE_LIST[@]}" ]]; then
    echo "[run_videophy2] NUM_GPUS=$NUM_GPUS exceeds CUDA_VISIBLE_DEVICES count (${#CUDA_VISIBLE_DEVICE_LIST[@]}): $CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
fi
export NUM_GPUS

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

RUN_TAG="${RUN_TAG:-videophy2_eval_$(date +%Y%m%d_%H%M%S)}"
RESULT_PATH="${RESULT_PATH:-outputs/eval/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-outputs/eval/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"

MODEL_TYPE="${MODEL_TYPE:-lamo}"
GENERATE_TYPE="${GENERATE_TYPE:-baseline}"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/pretrain_models/CogVideoX-2b-Diffusers}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/pretrain_models/LaMo-CogVideoX-2b}"
INPUT_CSV="${INPUT_CSV:-${REPO_ROOT}/eval/benchmark_data/videophy2/videophy2.csv}"
VIDEOPHY2_CHECKPOINT="${VIDEOPHY2_CHECKPOINT:-${REPO_ROOT}/pretrain_models/videophy_2_auto}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_EXE="${CONDA_EXE:-conda}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
CLEAN_EVAL_WORK="${CLEAN_EVAL_WORK:-1}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"
GENERATION_TIMEOUT_SECONDS="${GENERATION_TIMEOUT_SECONDS:-72000}"
GENERATION_SYNC_TIMEOUT_SECONDS="${GENERATION_SYNC_TIMEOUT_SECONDS:-$GENERATION_TIMEOUT_SECONDS}"
ENTAILMENT_EVAL_TIMEOUT_SECONDS="${ENTAILMENT_EVAL_TIMEOUT_SECONDS:-21600}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-1}"

if [[ "$RESUME_GENERATION" == "1" ]]; then
  CLEAN_EVAL_WORK=0
  SKIP_EXISTING=1
fi

mkdir -p "$RESULT_PATH" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run_videophy2] repo:        $REPO_ROOT"
echo "[run_videophy2] run_tag:     $RUN_TAG"
echo "[run_videophy2] gpu:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} NUM_GPUS=$NUM_GPUS WORKERS_PER_RANK=$WORKERS_PER_RANK"
echo "[run_videophy2] model_type:  $MODEL_TYPE"
echo "[run_videophy2] gen_type:    $GENERATE_TYPE"
echo "[run_videophy2] base_model:  $BASE_MODEL"
echo "[run_videophy2] checkpoint:  $CHECKPOINT_PATH"
echo "[run_videophy2] input_csv:   $INPUT_CSV"
echo "[run_videophy2] evaluator:   $VIDEOPHY2_CHECKPOINT"
echo "[run_videophy2] result_path: $RESULT_PATH"
echo "[run_videophy2] log_file:    $LOG_FILE"
echo "[run_videophy2] timeout:     generation=${GENERATION_TIMEOUT_SECONDS}s sync=${GENERATION_SYNC_TIMEOUT_SECONDS}s entailment=${ENTAILMENT_EVAL_TIMEOUT_SECONDS}s"
echo "[run_videophy2] resume:      RESUME_GENERATION=$RESUME_GENERATION CLEAN_EVAL_WORK=$CLEAN_EVAL_WORK SKIP_EXISTING=$SKIP_EXISTING"
date

for path in "$BASE_MODEL" "$CHECKPOINT_PATH" "$INPUT_CSV" "$VIDEOPHY2_CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "[run_videophy2] missing required path: $path" >&2
    exit 1
  fi
done

if [[ "$MODEL_TYPE" != "lamo" && "$MODEL_TYPE" != "cogvideox" ]]; then
  echo "[run_videophy2] MODEL_TYPE must be lamo or cogvideox; got: $MODEL_TYPE" >&2
  exit 1
fi
if [[ "$GENERATE_TYPE" == "lora" ]]; then
  if [[ ! -f "$CHECKPOINT_PATH/pytorch_lora_weights.safetensors" && \
        ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" && \
        ! -f "$CHECKPOINT_PATH/transformer/pytorch_lora_weights.safetensors" && \
        ! -f "$CHECKPOINT_PATH/transformer/adapter_model.safetensors" ]]; then
    echo "[run_videophy2] missing LoRA weights under: $CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ "$MODEL_TYPE" == "lamo" && \
        ! -f "$CHECKPOINT_PATH/predictor.safetensors" && \
        ! -f "$CHECKPOINT_PATH/transformer/predictor.safetensors" ]]; then
    echo "[run_videophy2] missing predictor.safetensors under: $CHECKPOINT_PATH" >&2
    exit 1
  fi
elif [[ "$GENERATE_TYPE" == "baseline" ]]; then
  if [[ ! -f "$CHECKPOINT_PATH/transformer/diffusion_pytorch_model.safetensors" ]]; then
    echo "[run_videophy2] missing transformer checkpoint under: $CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ "$MODEL_TYPE" == "lamo" && ! -f "$CHECKPOINT_PATH/predictor.safetensors" ]]; then
    echo "[run_videophy2] missing predictor.safetensors under: $CHECKPOINT_PATH" >&2
    exit 1
  fi
else
  echo "[run_videophy2] GENERATE_TYPE must be baseline or lora; got: $GENERATE_TYPE" >&2
  exit 1
fi

export RUN_TAG
export MODEL_TYPE
export GENERATE_TYPE
export RESULT_PATH
export BASE_MODEL
export CHECKPOINT_PATH
export INPUT_CSV
export VIDEOPHY2_CHECKPOINT
export GENERATION_TIMEOUT_SECONDS
export GENERATION_SYNC_TIMEOUT_SECONDS
export ENTAILMENT_EVAL_TIMEOUT_SECONDS
export SKIP_EXISTING
export WORKERS_PER_RANK

if command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1 || [[ -x "$NVIDIA_SMI_BIN" ]]; then
  "$NVIDIA_SMI_BIN" || true
fi

if [[ "$CLEAN_EVAL_WORK" == "1" ]]; then
  echo "[run_videophy2] removing stale eval_work_videophy2 markers and generated files"
  rm -rf "$REPO_ROOT/eval_work_videophy2"
else
  echo "[run_videophy2] preserving eval_work_videophy2 generated files and removing stale markers"
  find "$REPO_ROOT/eval_work_videophy2" -maxdepth 1 -type f -name ".cluster_videophy2_*" -delete 2>/dev/null || true
  find "$REPO_ROOT/eval_work_videophy2" -maxdepth 1 -type f -name ".cluster_eval_gen_rank_*" -delete 2>/dev/null || true
fi

ENTRYPOINT="$REPO_ROOT/eval/scripts/_run_videophy2_eval_entry.py"
if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCH_CMD=(python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" "$ENTRYPOINT")
else
  export WORLD_SIZE=1
  export RANK=0
  export LOCAL_RANK=0
  LAUNCH_CMD=(python -u "$ENTRYPOINT")
fi

echo "[run_videophy2] launch:      ${LAUNCH_CMD[*]}"
if [[ -n "$CONDA_ENV" ]]; then
  if ! command -v "$CONDA_EXE" >/dev/null 2>&1 && [[ ! -x "$CONDA_EXE" ]]; then
    echo "[run_videophy2] CONDA_ENV=$CONDA_ENV but conda executable was not found: $CONDA_EXE" >&2
    exit 1
  fi
  "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" "${LAUNCH_CMD[@]}"
else
  "${LAUNCH_CMD[@]}"
fi

echo "[run_videophy2] done"
echo "[run_videophy2] summary: $RESULT_PATH/result_summary_videophy2.txt"
date
