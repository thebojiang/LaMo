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

# Public local VBench evaluation entrypoint. Supports single-GPU and
# single-node multi-GPU generation before running VBench metrics on rank 0.
#
# Common overrides:
#   BASE_MODEL=/path/to/CogVideoX-2b-Diffusers
#   CHECKPOINT_PATH=/path/to/LaMo-CogVideoX-2b
#   MODEL_TYPE=lamo|cogvideox
#   GENERATE_TYPE=baseline|lora|full-finetune
#   PROMPT_INFO_JSON=/path/to/VBench_full_info.json
#   VBENCH_PRETRAINED_PATH=/path/to/vbench_pretrained
#   RUN_TAG=name or RESULT_PATH=/path/to/output
#   RESUME_GENERATION=1
#   WORKERS_PER_RANK=1
#   EVAL_LIMIT=N
#   NUM_SAMPLES=5
#   DIMENSIONS=subject_consistency
#   VBENCH_EVAL_GPUS=1
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
  echo "[run_vbench] NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "all" ]]; then
  IFS=',' read -r -a CUDA_VISIBLE_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "$NUM_GPUS" -gt "${#CUDA_VISIBLE_DEVICE_LIST[@]}" ]]; then
    echo "[run_vbench] NUM_GPUS=$NUM_GPUS exceeds CUDA_VISIBLE_DEVICES count (${#CUDA_VISIBLE_DEVICE_LIST[@]}): $CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
fi
export NUM_GPUS

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

RUN_TAG="${RUN_TAG:-vbench_eval_$(date +%Y%m%d_%H%M%S)}"
RESULT_PATH="${RESULT_PATH:-outputs/eval/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-outputs/eval/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"

MODEL_TYPE="${MODEL_TYPE:-lamo}"
GENERATE_TYPE="${GENERATE_TYPE:-baseline}"
BASE_MODEL="${BASE_MODEL:-${REPO_ROOT}/pretrain_models/CogVideoX-2b-Diffusers}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/pretrain_models/LaMo-CogVideoX-2b}"
PROMPT_INFO_JSON="${PROMPT_INFO_JSON:-${REPO_ROOT}/eval/benchmark_data/vbench/VBench_full_info.json}"
VBENCH_ROOT="${VBENCH_ROOT:-${REPO_ROOT}/eval/vbench}"
VBENCH_PRETRAINED_PATH="${VBENCH_PRETRAINED_PATH:-${REPO_ROOT}/pretrain_models/vbench_pretrained}"
CONDA_ENV="${CONDA_ENV:-}"
CONDA_EXE="${CONDA_EXE:-conda}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
CLEAN_EVAL_WORK="${CLEAN_EVAL_WORK:-1}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"
GENERATION_TIMEOUT_SECONDS="${GENERATION_TIMEOUT_SECONDS:-72000}"
GENERATION_SYNC_TIMEOUT_SECONDS="${GENERATION_SYNC_TIMEOUT_SECONDS:-$GENERATION_TIMEOUT_SECONDS}"
VBENCH_EVAL_TIMEOUT_SECONDS="${VBENCH_EVAL_TIMEOUT_SECONDS:-14400}"
VBENCH_EVAL_GPUS="${VBENCH_EVAL_GPUS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
WORKERS_PER_RANK="${WORKERS_PER_RANK:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SKIP_DETECTRON2_INSTALL="${SKIP_DETECTRON2_INSTALL:-0}"
SKIP_VBENCH_PRETRAINED_DOWNLOAD="${SKIP_VBENCH_PRETRAINED_DOWNLOAD:-0}"

for numeric_var in VBENCH_EVAL_GPUS WORKERS_PER_RANK NUM_SAMPLES; do
  numeric_value="${!numeric_var}"
  if ! [[ "$numeric_value" =~ ^[0-9]+$ ]] || [[ "$numeric_value" -lt 1 ]]; then
    echo "[run_vbench] $numeric_var must be a positive integer, got: $numeric_value" >&2
    exit 1
  fi
done

if [[ "$RESUME_GENERATION" == "1" ]]; then
  CLEAN_EVAL_WORK=0
  SKIP_EXISTING=1
fi

mkdir -p "$RESULT_PATH" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run_vbench] repo:        $REPO_ROOT"
echo "[run_vbench] run_tag:     $RUN_TAG"
echo "[run_vbench] gpu:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} NUM_GPUS=$NUM_GPUS WORKERS_PER_RANK=$WORKERS_PER_RANK VBENCH_EVAL_GPUS=$VBENCH_EVAL_GPUS"
echo "[run_vbench] model_type:  $MODEL_TYPE"
echo "[run_vbench] gen_type:    $GENERATE_TYPE"
echo "[run_vbench] base_model:  $BASE_MODEL"
echo "[run_vbench] checkpoint:  ${CHECKPOINT_PATH:-<none>}"
echo "[run_vbench] prompts:     $PROMPT_INFO_JSON"
echo "[run_vbench] vbench_root: $VBENCH_ROOT"
echo "[run_vbench] vbench_ckpt: ${VBENCH_PRETRAINED_PATH:-<runtime download/cache>}"
echo "[run_vbench] result_path: $RESULT_PATH"
echo "[run_vbench] log_file:    $LOG_FILE"
echo "[run_vbench] samples:     NUM_SAMPLES=$NUM_SAMPLES DIMENSIONS=${DIMENSIONS:-<standard>}"
echo "[run_vbench] timeout:     generation=${GENERATION_TIMEOUT_SECONDS}s sync=${GENERATION_SYNC_TIMEOUT_SECONDS}s vbench_eval=${VBENCH_EVAL_TIMEOUT_SECONDS}s"
echo "[run_vbench] resume:      RESUME_GENERATION=$RESUME_GENERATION CLEAN_EVAL_WORK=$CLEAN_EVAL_WORK SKIP_EXISTING=$SKIP_EXISTING"
echo "[run_vbench] dist:        VBENCH_DIST_BACKEND=${VBENCH_DIST_BACKEND:-<auto>}"
date

for path in "$BASE_MODEL" "$PROMPT_INFO_JSON" "$VBENCH_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[run_vbench] missing required path: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$VBENCH_ROOT/evaluate.py" ]]; then
  echo "[run_vbench] missing VBench evaluate.py under: $VBENCH_ROOT" >&2
  exit 1
fi
if [[ -n "$CHECKPOINT_PATH" && ! -e "$CHECKPOINT_PATH" ]]; then
  echo "[run_vbench] missing checkpoint path: $CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ "$MODEL_TYPE" != "lamo" && "$MODEL_TYPE" != "cogvideox" ]]; then
  echo "[run_vbench] MODEL_TYPE must be lamo or cogvideox; got: $MODEL_TYPE" >&2
  exit 1
fi
if [[ "$GENERATE_TYPE" == "lora" ]]; then
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "[run_vbench] GENERATE_TYPE=lora requires CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$CHECKPOINT_PATH/pytorch_lora_weights.safetensors" && \
        ! -f "$CHECKPOINT_PATH/adapter_model.safetensors" && \
        ! -f "$CHECKPOINT_PATH/transformer/pytorch_lora_weights.safetensors" && \
        ! -f "$CHECKPOINT_PATH/transformer/adapter_model.safetensors" ]]; then
    echo "[run_vbench] missing LoRA weights under: $CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ "$MODEL_TYPE" == "lamo" && \
        ! -f "$CHECKPOINT_PATH/predictor.safetensors" && \
        ! -f "$CHECKPOINT_PATH/transformer/predictor.safetensors" ]]; then
    echo "[run_vbench] missing predictor.safetensors under: $CHECKPOINT_PATH" >&2
    exit 1
  fi
elif [[ "$GENERATE_TYPE" == "baseline" || "$GENERATE_TYPE" == "full-finetune" ]]; then
  if [[ "$GENERATE_TYPE" == "full-finetune" && -z "$CHECKPOINT_PATH" ]]; then
    echo "[run_vbench] GENERATE_TYPE=full-finetune requires CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ -n "$CHECKPOINT_PATH" ]]; then
    if [[ ! -f "$CHECKPOINT_PATH/transformer/diffusion_pytorch_model.safetensors" ]]; then
      echo "[run_vbench] missing transformer checkpoint under: $CHECKPOINT_PATH" >&2
      exit 1
    fi
    if [[ "$MODEL_TYPE" == "lamo" && ! -f "$CHECKPOINT_PATH/predictor.safetensors" ]]; then
      echo "[run_vbench] missing predictor.safetensors under: $CHECKPOINT_PATH" >&2
      exit 1
    fi
  fi
else
  echo "[run_vbench] GENERATE_TYPE must be baseline, full-finetune, or lora; got: $GENERATE_TYPE" >&2
  exit 1
fi

if [[ ! -d "$VBENCH_PRETRAINED_PATH" && "$SKIP_VBENCH_PRETRAINED_DOWNLOAD" != "1" ]]; then
  echo "[run_vbench] warning: VBench pretrained cache not found: $VBENCH_PRETRAINED_PATH"
  echo "[run_vbench] warning: run eval/scripts/download_vbench_pretrained.sh first, or set SKIP_VBENCH_PRETRAINED_DOWNLOAD=1 to let VBench fetch missing assets at runtime."
fi

export RUN_TAG
export MODEL_TYPE
export GENERATE_TYPE
export RESULT_PATH
export BASE_MODEL
export CHECKPOINT_PATH
export PROMPT_INFO_JSON
export VBENCH_ROOT
export VBENCH_PRETRAINED_PATH
export GENERATION_TIMEOUT_SECONDS
export GENERATION_SYNC_TIMEOUT_SECONDS
export VBENCH_EVAL_TIMEOUT_SECONDS
export VBENCH_EVAL_GPUS
export SKIP_EXISTING
export WORKERS_PER_RANK
export NUM_SAMPLES
export SKIP_DETECTRON2_INSTALL
export SKIP_VBENCH_PRETRAINED_DOWNLOAD
if [[ -n "${VBENCH_DIST_BACKEND:-}" ]]; then
  export VBENCH_DIST_BACKEND
fi

if command -v "$NVIDIA_SMI_BIN" >/dev/null 2>&1 || [[ -x "$NVIDIA_SMI_BIN" ]]; then
  "$NVIDIA_SMI_BIN" || true
fi

if [[ "$CLEAN_EVAL_WORK" == "1" ]]; then
  echo "[run_vbench] removing stale eval_work_vbench markers and generated files"
  rm -rf "$REPO_ROOT/eval_work_vbench"
else
  echo "[run_vbench] preserving eval_work_vbench generated files and removing stale markers"
  find "$REPO_ROOT/eval_work_vbench" -maxdepth 1 -type f -name ".cluster_vbench_*" -delete 2>/dev/null || true
fi

ENTRYPOINT="$REPO_ROOT/eval/scripts/_run_vbench_eval_entry.py"
if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCH_CMD=(python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" "$ENTRYPOINT")
else
  export WORLD_SIZE=1
  export RANK=0
  export LOCAL_RANK=0
  LAUNCH_CMD=(python -u "$ENTRYPOINT")
fi

echo "[run_vbench] launch:      ${LAUNCH_CMD[*]}"
if [[ -n "$CONDA_ENV" ]]; then
  if ! command -v "$CONDA_EXE" >/dev/null 2>&1 && [[ ! -x "$CONDA_EXE" ]]; then
    echo "[run_vbench] CONDA_ENV=$CONDA_ENV but conda executable was not found: $CONDA_EXE" >&2
    exit 1
  fi
  "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" "${LAUNCH_CMD[@]}"
else
  "${LAUNCH_CMD[@]}"
fi

echo "[run_vbench] done"
echo "[run_vbench] summary: $RESULT_PATH/result_summary.txt"
date
