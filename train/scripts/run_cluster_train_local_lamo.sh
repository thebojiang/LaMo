#!/bin/bash

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

# 本地测试 cluster_train_eval.py 的 LaMo 训练。
#
# ========== 使用方式 ==========
# 默认从 train/configs/model_config_train_eval.yaml 读取 config。
# 1. 确保 train/configs/model_config_train_eval.yaml 中有 num_gpus、variant、dataset_config、checkpoint_path
# 2. 进入仓库根目录: cd <PATH_TO_LAMO_REPO>
# 3. 运行: bash train/scripts/run_cluster_train_local_lamo.sh
# 本地 GPU 数可用 NUM_GPUS 覆盖（如 NUM_GPUS=2），会覆盖 yaml 里的 num_gpus。
# 若不用 yaml，可设置 CLUSTER_TRAIN_CONFIG='{"num_gpus": 2, "variant": "lamo", "dataset_config": "path/to/training.json"}' 直接传 JSON。
#
# CUDA_VISIBLE_DEVICES=0,1 bash train/scripts/run_cluster_train_local_lamo.sh
# =====================================================

set -e -x

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="${FINETRAINERS_LOG_LEVEL:-DEBUG}"

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
  echo "[run_train] NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "all" ]]; then
  IFS=',' read -r -a CUDA_VISIBLE_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
  if [[ "$NUM_GPUS" -gt "${#CUDA_VISIBLE_DEVICE_LIST[@]}" ]]; then
    echo "[run_train] NUM_GPUS=$NUM_GPUS exceeds CUDA_VISIBLE_DEVICES count (${#CUDA_VISIBLE_DEVICE_LIST[@]}): $CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
fi
export NUM_GPUS

# 可选：覆盖 dataset_config。
# 例如: export CLUSTER_TRAIN_DATASET_CONFIG="<PATH_TO_TRAINING_DATASET_CONFIG>"
DATASET_CONFIG_OVERRIDE="${CLUSTER_TRAIN_DATASET_CONFIG:-}"

# 配置来源（优先级从高到低）：
# 1) CLUSTER_TRAIN_CONFIG：已设置则用该 JSON，不再读 yaml
# 2) MODEL_CONFIG_YAML：yaml 文件路径（默认 train/configs/model_config_train_eval.yaml）
# 3) 脚本内默认：num_gpus + variant=lamo
MODEL_CONFIG_YAML="${MODEL_CONFIG_YAML:-train/configs/model_config_train_eval.yaml}"

CLUSTER_TRAIN_CONFIG_WAS_SET=0
if [[ -n "${CLUSTER_TRAIN_CONFIG:-}" ]]; then
  CLUSTER_TRAIN_CONFIG_WAS_SET=1
fi

export MODEL_CONFIG_YAML
export CLUSTER_TRAIN_CONFIG_WAS_SET
# 仅当未提供 JSON 且找不到 yaml 时再补默认 JSON。
if [[ "$CLUSTER_TRAIN_CONFIG_WAS_SET" == "0" && ! -f "$MODEL_CONFIG_YAML" ]]; then
  if [ -n "$DATASET_CONFIG_OVERRIDE" ]; then
    CLUSTER_TRAIN_CONFIG="{\"num_gpus\": $NUM_GPUS, \"variant\": \"lamo\", \"dataset_config\": \"$DATASET_CONFIG_OVERRIDE\"}"
  else
    CLUSTER_TRAIN_CONFIG="{\"num_gpus\": $NUM_GPUS, \"variant\": \"lamo\"}"
  fi
fi
export CLUSTER_TRAIN_CONFIG

if [[ "$CLUSTER_TRAIN_CONFIG_WAS_SET" == "1" ]]; then
  CONFIG_SOURCE="CLUSTER_TRAIN_CONFIG"
elif [[ -f "$MODEL_CONFIG_YAML" ]]; then
  CONFIG_SOURCE="$MODEL_CONFIG_YAML"
else
  CONFIG_SOURCE="default JSON"
fi
echo "Running cluster_train_eval.main (config from: $CONFIG_SOURCE)"
python -c "
import json, os, sys

def load_config():
    raw = os.environ.get('CLUSTER_TRAIN_CONFIG', '')
    if raw and os.environ.get('CLUSTER_TRAIN_CONFIG_WAS_SET', '0') == '1':
        return json.loads(raw)
    yaml_path = os.environ.get('MODEL_CONFIG_YAML', '')
    if yaml_path and os.path.isfile(yaml_path):
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data
    if raw:
        return json.loads(raw)
    return {}

cfg = load_config()
# 本地可覆盖 num_gpus（与 CUDA_VISIBLE_DEVICES 张数一致）
num_gpus = int(os.environ.get('NUM_GPUS', cfg.get('num_gpus', 1)))
cfg['num_gpus'] = num_gpus
dataset_config = os.environ.get('CLUSTER_TRAIN_DATASET_CONFIG')
if dataset_config:
    cfg['dataset_config'] = dataset_config

from train.cluster_train_eval import main
main(cfg)
"

echo "Done: cluster_train_eval.py ran successfully (num_gpus=$NUM_GPUS)."
