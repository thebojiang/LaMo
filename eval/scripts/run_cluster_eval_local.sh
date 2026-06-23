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

# 本地测试 cluster_eval.py 的 VideoPhy 评测。
#
# ========== 使用方式 ==========
# 默认从 eval/configs/model_config_eval.yaml 读取 config。
# 1. 确保 eval/configs/model_config_eval.yaml 中有 checkpoint_path、input_csv、videocon_checkpoint
# 2. 进入仓库根目录: cd <PATH_TO_LAMO_REPO>
# 3. 运行: bash eval/scripts/run_cluster_eval_local.sh
# 本地单进程跑，无需多卡；可用 CUDA_VISIBLE_DEVICES 指定生成/评测用的 GPU。
# 若不用 yaml，可设置 CLUSTER_EVAL_CONFIG='{"checkpoint_path":"<PATH_TO_LORA_OR_MODEL>", "input_csv":"<PATH_TO_INPUT_CSV>", "videocon_checkpoint":"<PATH_TO_VIDEOPHY_EVALUATOR>"}' 直接传 JSON。
#
# CUDA_VISIBLE_DEVICES=0 bash eval/scripts/run_cluster_eval_local.sh
# =====================================================

set -e -x
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

# 本地单进程跑评测，不触发“非主进程等待”逻辑
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 配置来源（优先级从高到低）：
# 1) CLUSTER_EVAL_CONFIG：已设置则用该 JSON，不再读 yaml
# 2) MODEL_CONFIG_EVAL_YAML：yaml 文件路径（默认 eval/configs/model_config_eval.yaml）
MODEL_CONFIG_EVAL_YAML="${MODEL_CONFIG_EVAL_YAML:-eval/configs/model_config_eval.yaml}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# 仅当未提供 JSON 时，由 Python 从 yaml 读 config
export CLUSTER_EVAL_CONFIG
export MODEL_CONFIG_EVAL_YAML

echo "Running cluster_eval.main (config from: ${MODEL_CONFIG_EVAL_YAML:-CLUSTER_EVAL_CONFIG})"
python -c "
import json, os, sys

def load_config():
    raw = os.environ.get('CLUSTER_EVAL_CONFIG', '')
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    yaml_path = os.environ.get('MODEL_CONFIG_EVAL_YAML', '')
    if yaml_path and os.path.isfile(yaml_path):
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data
    return {}

cfg = load_config()
if not cfg:
    print('No config: set CLUSTER_EVAL_CONFIG or ensure MODEL_CONFIG_EVAL_YAML exists', file=sys.stderr)
    sys.exit(1)

# 本地可减小 world_size / limit 做快速测试
if os.environ.get('EVAL_WORLD_SIZE'):
    cfg['world_size'] = int(os.environ['EVAL_WORLD_SIZE'])
if os.environ.get('EVAL_LIMIT'):
    cfg['limit'] = int(os.environ['EVAL_LIMIT'])

from eval.cluster_eval import main
main(cfg)
"

echo "Done: cluster_eval.py ran successfully."
