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

"""
Unified local training and evaluation entry point for LaMo / CogVideoX.

统一入口脚本，支持三种模式，由 config 中的 `mode` 参数指定：
  - mode: "train"      → 仅训练（等价于 cluster_train.py）
  - mode: "eval"       → 仅评测（等价于 cluster_eval.py）
  - mode: "train_eval" → 先训练后评测（一次性完成）

Public entry config: train/configs/model_config_train_eval.yaml.

训练模式 (mode: train / train_eval):
  - variant: "lamo"
  - training_type: "lora" | "full-finetune"
  - checkpoint_path: 本地 checkpoint 输出路径（previous field name）
  - num_gpus: 训练使用的 GPU 数（如 16 = 2 机 × 8 卡）
  - 参见 cluster_train.py 的完整参数说明

评测模式 (mode: eval / train_eval):
  - eval_benchmark: "videophy" (默认) | "videophy2" | "vbench" - 选择评测基准
  - model_type: "lamo" | "cogvideox"
  - generate_type: "baseline" | "lora"
  - checkpoint_path: 训练产出的本地权重路径（previous field name）
  - result_path: 本地评测结果目录（previous field name）
  - eval_num_gpus: 评测使用的 GPU 数（默认 8 = 1 机 × 8 卡）
  - 参见 cluster_eval.py / cluster_videophy2_eval.py / cluster_vbench_eval.py 的完整参数说明

train_eval 模式说明：
  - 训练阶段使用 num_gpus（如 16）个 GPU
  - 评测阶段使用 eval_num_gpus（默认 8）个 GPU
  - 多余的 rank（如 rank 8-15）在评测阶段会自动退出
单进程启动示例：
  # 训练 lamo 模型
  python -c "from train.cluster_train_eval import main; main({'mode': 'train', 'variant': 'lamo', 'num_gpus': 2})"

  # 训练 + 评测（先用 16 GPU 训练，再用 8 GPU 评测）
  python -c "from train.cluster_train_eval import main; main({'mode': 'train_eval', 'num_gpus': 16, 'eval_num_gpus': 8, ...})"
"""

import json
import logging
import os
import shutil
import subprocess
import sys

try:
    import yaml
except ImportError:
    yaml = None

# Ensure project root is on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# -----------------------------------------------------------------------------
# Environment variables: shared by both train and eval
# -----------------------------------------------------------------------------
os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("FINETRAINERS_LOG_LEVEL", "DEBUG")

# Fix: Prevent transformers from using broken Apex FusedLayerNorm on cluster
sys.modules["apex"] = None
sys.modules["apex.normalization"] = None
logging.info("Blocked apex.normalization to prevent FusedLayerNorm issues")

logging.getLogger("urllib3").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Training constants
# -----------------------------------------------------------------------------
_MODEL_REF = os.environ.get("LAMO_BASE_MODEL", "THUDM/CogVideoX-2b")
_COGVIDEOX_2B_MODEL = os.environ.get("LAMO_COGVIDEOX_2B_MODEL", "THUDM/CogVideoX-2b")
_COGVIDEOX_5B_MODEL = os.environ.get("LAMO_COGVIDEOX_5B_MODEL", "THUDM/CogVideoX-5b")
_COGVIDEOX_OPENVID_32K_CONFIG = "train/configs/datasets/openvid/training_openvid_32k.json"
_COGVIDEOX_OPENVID_64K_CONFIG = os.environ.get("LAMO_DATASET_CONFIG", "")


# Training experiment params
COMMON_EXPERIMENT_PARAMS = {
    "lamo": {
        "model_name": "lamo",
        "pretrained_model_name_or_path": _COGVIDEOX_2B_MODEL,
        "no_validation": True,
        "tracker_name": "finetrainers-cogvideox-lamo",
        "output_dir": "./outputs/checkpoints/cluster",
        "logging_dir": "./outputs/logs/lamo",
        "predictor_in_channels": "16",
        "predictor_num_res_blocks": "8",
        "predictor_hidden_channels": "256",
        "predictor_use_se": "1",
        "predictor_use_prev_delta": "0",
        "predictor_use_prompt_cond": "1",
        "prompt_text_dim": "4096",
        "prompt_cond_dim": "128",
        "prompt_dropout_prob": "0.1",
        "predictor_lr": "5e-4",
        "predictor_cosine_weight": "0.3",
        "predictor_noise_aug_prob": "0.5",
        "predictor_noise_aug_scale": "0.3",
        "predictor_lr_scheduler": "cosine",
        "lambda_bmv_rel_l2": "0.4",
        "bmv_tau": "2",
        "enable_denoiser_motion_loss": "1",
        "guidance_lambda": "0.0",
        "guidance_step_ratio": "0.0",
    },
}

LORA_PARAMS = {
    "lora_rank": 128,
    "lora_alpha": 64,
}

# -----------------------------------------------------------------------------
# Evaluation constants
# -----------------------------------------------------------------------------
_WAN_1_3B_MODEL = _MODEL_REF

MODEL_DEFAULTS = {
    "cogvideox": {
        "num_frames": 49,
        "fps": 8,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
    },
    "lamo": {
        "num_frames": 49,
        "fps": 8,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
    },
}

_SINGLE_NODE_MAX_GPUS = 8
_DEFAULT_EVAL_NUM_GPUS = 8  # 评测默认使用 8 GPU（1 机）


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_subprocess_streaming(cmd, cwd, env):
    """
    Run subprocess while re-printing stdout/stderr to the current process in real time.

    When a distributed worker (e.g. RANK 0) spawns a new torchrun subprocess,
    the cluster platform's rank-based log collection may not capture the child
    process tree's output. By piping through the parent and re-printing, the
    output appears as RANK 0's own logs and is properly captured.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode


# =============================================================================
# Training helpers (from cluster_train.py)
# =============================================================================

def _resolve_eval_num_gpus(cfg: dict) -> int:
    """
    解析评测模式的 GPU 数量。优先级：
    1. cfg["eval_num_gpus"] - 评测专用配置
    2. cfg["num_gpus"] - 通用配置（向后兼容）
    3. 环境变量 WORLD_SIZE（集群已分配的 GPU 数，但限制为 <=8）
    4. 默认 8（单机 8 卡）
    
    注意：评测模式默认限制为单机（<=8 GPU），除非显式指定 eval_num_gpus > 8
    """
    # 显式指定的 eval_num_gpus 优先
    v = cfg.get("eval_num_gpus")
    if v is not None:
        return int(v)
    
    # 如果只指定了 num_gpus，评测模式限制为最多 8（单机）
    v = cfg.get("num_gpus")
    if v is not None:
        v = int(v)
        # 评测默认不超过 8 GPU，避免训练用 16 GPU 时评测也用 16
        return min(v, _DEFAULT_EVAL_NUM_GPUS)
    
    # 从环境变量获取（集群已分配），但限制为单机
    world_size = os.environ.get("WORLD_SIZE")
    if world_size:
        return min(int(world_size), _DEFAULT_EVAL_NUM_GPUS)
    
    return _DEFAULT_EVAL_NUM_GPUS


def _is_distributed_worker_env() -> bool:
    """Return True when this process is already a torchrun/cluster worker."""
    if os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
        return True
    world_size = os.environ.get("WORLD_SIZE")
    return bool(world_size and world_size != "1")


def _default_training_argv(
    num_gpus: int = 8,
    variant: str = "lamo",
    dataset_config_override: str = None,
    config_overrides: dict = None,
):
    """Build training argv for the open-source lamo model."""
    if variant != "lamo":
        raise ValueError("Only variant='lamo' is supported in the open-source release.")

    p = dict(COMMON_EXPERIMENT_PARAMS["lamo"])
    if isinstance(config_overrides, dict):
        p.update({key: value for key, value in config_overrides.items() if value is not None})
    dataset_config = dataset_config_override or p.get("dataset_config")
    if not dataset_config:
        raise ValueError(
            "dataset_config is required for open-source training. Pass it in "
            "train/configs/model_config_train_eval.yaml, CLUSTER_TRAIN_EVAL_CONFIG, or "
            "LAMO_DATASET_CONFIG."
        )
    p["dataset_config"] = dataset_config
    if p.get("training_type") == "lora":
        p = {**LORA_PARAMS, **p}

    argv = [
        "--parallel_backend", "ptd",
        "--pp_degree", "1",
        "--dp_degree", str(num_gpus),
        "--dp_shards", "1",
        "--cp_degree", "1",
        "--tp_degree", "1",
        "--model_name", "lamo",
        "--pretrained_model_name_or_path", p.get("pretrained_model_name_or_path", _COGVIDEOX_2B_MODEL),
        "--dataset_config", p["dataset_config"],
        "--dataset_shuffle_buffer_size", "10",
        "--dataloader_num_workers", "0",
        "--flow_weighting_scheme", "logit_normal",
        "--training_type", p["training_type"],
        "--seed", "42",
        "--batch_size", str(p["batch_size"]),
        "--train_steps", str(p["train_steps"]),
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing",
        "--checkpointing_steps", str(p["checkpointing_steps"]),
        "--checkpointing_limit", str(p["checkpointing_limit"]),
        "--enable_slicing",
        "--enable_tiling",
        "--optimizer", "adamw",
        "--lr", str(p["lr"]),
        "--lr_scheduler", p.get("lr_scheduler", "constant_with_warmup"),
        "--lr_warmup_steps", str(p["lr_warmup_steps"]),
        "--lr_num_cycles", "1",
        "--beta1", "0.9",
        "--beta2", "0.99",
        "--weight_decay", "1e-4",
        "--epsilon", "1e-8",
        "--max_grad_norm", "1.0",
        "--validation_steps", "50000",
        "--init_timeout", "3600",
        "--nccl_timeout", "3600",
        "--report_to", str(p.get("report_to", "wandb")),
    ]
    if p.get("no_validation"):
        argv.append("--no_validation")
    argv.extend([
        "--tracker_name", p["tracker_name"],
        "--output_dir", p["output_dir"],
        "--logging_dir", p["logging_dir"],
        "--predictor_in_channels", p.get("predictor_in_channels", "16"),
        "--predictor_num_res_blocks", p.get("predictor_num_res_blocks", "8"),
        "--predictor_hidden_channels", p.get("predictor_hidden_channels", "256"),
        "--predictor_use_se", p.get("predictor_use_se", "1"),
        "--predictor_use_prev_delta", p.get("predictor_use_prev_delta", "0"),
        "--predictor_use_prompt_cond", p.get("predictor_use_prompt_cond", "1"),
        "--prompt_text_dim", p.get("prompt_text_dim", "4096"),
        "--prompt_cond_dim", p.get("prompt_cond_dim", "128"),
        "--prompt_dropout_prob", p.get("prompt_dropout_prob", "0.1"),
        "--predictor_lr", p.get("predictor_lr", "5e-4"),
        "--predictor_lr_scheduler", p.get("predictor_lr_scheduler", "cosine"),
        "--predictor_noise_aug_prob", p.get("predictor_noise_aug_prob", "0.5"),
        "--predictor_noise_aug_scale", p.get("predictor_noise_aug_scale", "0.3"),
        "--predictor_cosine_weight", p.get("predictor_cosine_weight", "0.3"),
        "--lambda_bmv_rel_l2", p.get("lambda_bmv_rel_l2", "0.0"),
        "--bmv_tau", p.get("bmv_tau", "2"),
        "--enable_denoiser_motion_loss", p.get("enable_denoiser_motion_loss", "0"),
        "--guidance_lambda", p.get("guidance_lambda", "15.0"),
        "--guidance_step_ratio", p.get("guidance_step_ratio", "0.8"),
    ])
    if p.get("training_type") == "lora":
        argv.extend([
            "--rank", str(p.get("lora_rank", 128)),
            "--lora_alpha", str(p.get("lora_alpha", 64)),
            "--target_modules", p.get("lora_target_modules", "transformer_blocks.*(to_q|to_k|to_v|to_out.0)"),
        ])
    return [str(item) for item in argv]

def _resolve_num_gpus(cfg: dict) -> int:
    """Resolve num_gpus from cfg, environment, or train/configs/model_config_train_eval.yaml."""
    v = cfg.get("num_gpus")
    if v is not None:
        return int(v)
    v = os.environ.get("NUM_GPUS")
    if v:
        return int(v)
    if yaml:
        for path in (
            "train/configs/model_config_train_eval.yaml",
            os.path.join(_REPO_ROOT, "train", "configs", "model_config_train_eval.yaml"),
        ):
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f) or {}
                    v = data.get("num_gpus")
                    if v is not None:
                        return int(v)
                except Exception:
                    pass
    return 8


def _run_train_as_launcher(cfg: dict, exit_after: bool = True) -> int:
    """
    Launch training with torch.distributed.run.
    
    Args:
        cfg: 配置字典
        exit_after: 是否在训练完成后退出。train_eval 模式下设为 False，以便继续评测。
    
    Returns:
        训练的返回码（仅当 exit_after=False 时返回）
    """
    num_gpus = _resolve_num_gpus(cfg)
    script = os.path.join(_REPO_ROOT, "train", "cluster_train_eval.py")

    if num_gpus <= _SINGLE_NODE_MAX_GPUS:
        env = {**os.environ, "CLUSTER_TRAIN_EVAL_CONFIG": json.dumps(cfg)}
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1", "--nproc_per_node", str(num_gpus),
            script,
        ]
        code = subprocess.run(cmd, cwd=_REPO_ROOT, env=env).returncode
        if exit_after:
            sys.exit(code)
        return code

    env = {**os.environ, "CLUSTER_TRAIN_EVAL_CONFIG": json.dumps(cfg)}
    nnodes = int(os.environ.get("NNODES", (num_gpus + _SINGLE_NODE_MAX_GPUS - 1) // _SINGLE_NODE_MAX_GPUS))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    nproc_per_node = int(os.environ.get("NPROC_PER_NODE", _SINGLE_NODE_MAX_GPUS))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes", str(nnodes), "--nproc_per_node", str(nproc_per_node),
        "--node_rank", str(node_rank), "--master_addr", master_addr, "--master_port", master_port,
        script,
    ]
    code = subprocess.run(cmd, cwd=_REPO_ROOT, env=env).returncode
    if exit_after:
        sys.exit(code)
    return code


def _run_training(cfg: dict, is_train_eval_mode: bool = False):
    """
    执行训练逻辑（对应 cluster_train.main 的核心部分）。
    
    Args:
        cfg: 配置字典
        is_train_eval_mode: 是否为 train_eval 模式（训练后还需评测）
    """
    if not _is_distributed_worker_env():
        # 单进程入口：用 torchrun 拉起训练
        # train_eval 模式下不退出，以便继续评测
        _run_train_as_launcher(cfg, exit_after=not is_train_eval_mode)
        return

    argv = cfg.get("argv")
    if argv is not None and isinstance(argv, (list, tuple)):
        argv = [str(x) for x in argv]
    else:
        num_gpus = _resolve_num_gpus(cfg)
        variant = cfg.get("variant", "lamo")
        dataset_config_override = cfg.get("dataset_config")
        argv = _default_training_argv(
            num_gpus,
            variant,
            dataset_config_override,
            config_overrides=cfg,
        )
    
    pretrained_override = cfg.get("pretrained_model_name_or_path")
    if pretrained_override:
        try:
            idx = argv.index("--pretrained_model_name_or_path")
            argv[idx + 1] = str(pretrained_override)
        except ValueError:
            argv = argv + ["--pretrained_model_name_or_path", str(pretrained_override)]
    
    sync_path = cfg.get("output_dir_sync_path")
    if sync_path:
        raise ValueError(
            "output_dir_sync_path is not supported in the open-source "
            "release. Use a local output_dir/checkpoint_path such as "
            "<PATH_TO_OUTPUT_DIR>."
        )
    
    adapter_weight_dir = cfg.get("adapter_weight_dir")
    if adapter_weight_dir:
        argv = argv + ["--adapter_weight_dir", str(adapter_weight_dir)]
    
    resume_from_checkpoint = cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint is not None:
        argv = argv + ["--resume_from_checkpoint", str(resume_from_checkpoint)]
    
    sys.argv = [sys.argv[0]] + argv
    from train.train import main as train_main
    train_main()


# =============================================================================
# Evaluation helpers (from cluster_eval.py)
# =============================================================================

def _is_main_rank() -> bool:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size <= 1 or rank == 0


def _get_rank_and_world_size():
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, max(1, world_size)


def _inject_variant_model_params(cfg: dict) -> dict:
    """Merge model-architecture params from COMMON_EXPERIMENT_PARAMS into eval config.

    COMMON_EXPERIMENT_PARAMS carries architecture params (e.g. physical_encoder_K,
    physical_decoder_base_spatial) that must match between training and eval.
    Explicit values already present in *cfg* take precedence.
    """
    variant = cfg.get("variant") or cfg.get("model_type")
    if not variant or variant not in COMMON_EXPERIMENT_PARAMS:
        return cfg
    merged = dict(cfg)
    for key, value in COMMON_EXPERIMENT_PARAMS[variant].items():
        if key not in merged or merged[key] is None:
            merged[key] = value
    return merged


def _resolve_eval_benchmark(cfg: dict) -> str:
    """Return normalized benchmark name, one of {"videophy", "videophy2", "vbench"}."""
    bench = str(cfg.get("eval_benchmark") or "videophy").strip().lower()
    if bench in ("vbench", "vbench_standard", "vbench-standard"):
        return "vbench"
    if bench in ("videophy2", "videophy_2", "videophy-2", "vp2"):
        return "videophy2"
    return "videophy"


def _resolve_eval_generate_type(cfg: dict) -> str:
    """Infer eval generate_type from explicit config or training_type."""
    raw = cfg.get("generate_type")
    if raw:
        value = str(raw).strip().lower()
        if value in {"full-finetune", "full_finetune", "fullfinetune"}:
            return "baseline"
        return value
    training_type = str(cfg.get("training_type") or "lora").strip().lower()
    return "lora" if training_type == "lora" else "baseline"


def _prepare_eval_workspace(cfg: dict, benchmark: str) -> None:
    """Match public eval launchers: clean eval work by default, resume only on request."""
    work_dir = _eval_work_dir(benchmark)
    resume = _as_bool(
        cfg.get("resume_generation", os.environ.get("RESUME_GENERATION")),
        default=False,
    )
    clean = _as_bool(
        cfg.get("clean_eval_work", os.environ.get("CLEAN_EVAL_WORK")),
        default=not resume,
    )
    if resume:
        cfg["skip_existing"] = True
        clean = False
    elif "skip_existing" not in cfg:
        # If the workspace is not cleaned for a non-resume run, force
        # regeneration so stale videos from a previous checkpoint cannot be
        # silently reused.
        cfg["skip_existing"] = clean

    if clean:
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        print(f"[cluster_train_eval] Cleaned eval workspace: {work_dir}", flush=True)
    else:
        os.makedirs(work_dir, exist_ok=True)
        _cleanup_eval_markers(work_dir, benchmark=benchmark)
        print(f"[cluster_train_eval] Preserved eval workspace and cleaned markers: {work_dir}", flush=True)


def _run_evaluation(cfg: dict):
    """
    执行评测逻辑（委托给对应基准的 cluster_*_eval.main）。

    默认使用 VideoPhy（cluster_eval.main）；通过 cfg["eval_benchmark"] 可切换：
      - "videophy"  -> cluster_eval.main (VideoPhy-1, videocon SA/PC)
      - "videophy2" -> cluster_videophy2_eval.main (VideoPhy-2 AutoEvaluator)
      - "vbench"    -> cluster_vbench_eval.main (VBench)

    评测默认使用单机 8 卡，即使训练配置了 16 卡（2 机）。
    可通过 eval_num_gpus 显式指定评测 GPU 数量。
    """
    eval_num_gpus = _resolve_eval_num_gpus(cfg)
    benchmark = _resolve_eval_benchmark(cfg)

    eval_cfg = {**cfg, "num_gpus": eval_num_gpus}
    eval_cfg["generate_type"] = _resolve_eval_generate_type(eval_cfg)
    resume_generation = _as_bool(
        eval_cfg.get("resume_generation", os.environ.get("RESUME_GENERATION")),
        default=False,
    )
    clean_eval_work = _as_bool(
        eval_cfg.get("clean_eval_work", os.environ.get("CLEAN_EVAL_WORK")),
        default=not resume_generation,
    )
    if resume_generation:
        eval_cfg["skip_existing"] = True
    elif "skip_existing" not in eval_cfg:
        eval_cfg["skip_existing"] = clean_eval_work

    # Inject variant defaults so model-architecture params (physical_encoder_K,
    # physical_decoder_base_spatial, etc.) automatically match training.
    eval_cfg = _inject_variant_model_params(eval_cfg)

    # 如果集群分配了更多 GPU（如 16），但评测只需 8，需要限制 WORLD_SIZE
    current_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if current_world_size > eval_num_gpus:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if rank >= eval_num_gpus:
            print(f"[cluster_train_eval] Rank {rank} >= eval_num_gpus ({eval_num_gpus}), exiting (eval uses fewer GPUs than allocated)", flush=True)
            return
        os.environ["WORLD_SIZE"] = str(eval_num_gpus)
        print(f"[cluster_train_eval] Adjusted WORLD_SIZE from {current_world_size} to {eval_num_gpus} for evaluation", flush=True)

    print(f"[cluster_train_eval] Evaluation using {eval_num_gpus} GPUs, benchmark={benchmark}", flush=True)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        _prepare_eval_workspace(eval_cfg, benchmark)
    if benchmark == "vbench":
        os.environ["CLUSTER_VBENCH_EVAL_CONFIG"] = json.dumps(eval_cfg)
        from eval.cluster_vbench_eval import main as vbench_main
        vbench_main(eval_cfg)
    elif benchmark == "videophy2":
        os.environ["CLUSTER_VIDEOPHY2_EVAL_CONFIG"] = json.dumps(eval_cfg)
        from eval.cluster_videophy2_eval import main as videophy2_main
        videophy2_main(eval_cfg)
    else:
        os.environ["CLUSTER_EVAL_CONFIG"] = json.dumps(eval_cfg)
        from eval.cluster_eval import main as eval_main
        eval_main(eval_cfg)


# =============================================================================
# Eval config helpers
# =============================================================================

def _prepare_eval_config_after_training(cfg: dict, rank: int = 0) -> dict:
    """
    Resolve the evaluation checkpoint path from the training config.

    Priority:
    1. eval_checkpoint_path (explicitly specified)
    2. checkpoint_path/weight_{train_steps} (LoRA auto-construction)
    3. checkpoint_path as-is (full finetune)
    """
    eval_num_gpus = _resolve_eval_num_gpus(cfg)
    eval_cfg = dict(cfg)
    eval_cfg["mode"] = "eval"
    eval_cfg["eval_num_gpus"] = eval_num_gpus

    eval_ckpt = cfg.get("eval_checkpoint_path") or cfg.get("eval_checkpoint_path")
    generate_type = _resolve_eval_generate_type(cfg)
    eval_cfg["generate_type"] = generate_type

    if eval_ckpt:
        eval_cfg["checkpoint_path"] = eval_ckpt
        if rank == 0:
            print(f"[cluster_train_eval] Using eval_checkpoint_path: {eval_ckpt}", flush=True)
    else:
        base_ckpt = (cfg.get("checkpoint_path") or cfg.get("checkpoint_path") or "").rstrip("/")
        if base_ckpt:
            if generate_type == "lora":
                train_steps = cfg.get("train_steps") or COMMON_EXPERIMENT_PARAMS["lamo"].get("train_steps", 1000)
                auto_eval_ckpt = f"{base_ckpt}/weight_{train_steps}"
                if rank == 0:
                    print(f"[cluster_train_eval] Auto-constructed eval checkpoint (LoRA): {auto_eval_ckpt}", flush=True)
            else:
                auto_eval_ckpt = base_ckpt
                if rank == 0:
                    print(f"[cluster_train_eval] Auto-constructed eval checkpoint (full finetune): {auto_eval_ckpt}", flush=True)
            eval_cfg["checkpoint_path"] = auto_eval_ckpt

    return eval_cfg


def _cleanup_eval_markers(work_dir: str, benchmark: str = "videophy"):
    """Remove stale evaluation marker files from a previous run.

    Markers differ between benchmarks:
      - videophy:  ``.cluster_eval_*`` in ``eval_work/``
      - videophy2: ``.cluster_videophy2_*`` in ``eval_work_videophy2/``.
        Older runs may also have ``.cluster_eval_gen_rank_*_done`` markers
        from the shared generator, so clean those too.
      - vbench:    ``.cluster_vbench_*`` in ``eval_work_vbench/``
    """
    if benchmark == "vbench":
        markers = [
            ".cluster_vbench_download_done",
            ".cluster_vbench_generation_done",
            ".cluster_vbench_job_done",
            ".cluster_vbench_phase2_args.json",
        ]
        rank_marker_prefixes = [".cluster_vbench_gen_rank_"]
    elif benchmark == "videophy2":
        markers = [
            ".cluster_videophy2_download_done",
            ".cluster_videophy2_generation_done",
            ".cluster_videophy2_job_done",
            ".cluster_videophy2_phase2_args.json",
        ]
        # videophy_cluster_generate.py writes _eval_ prefixed rank markers
        # which cluster_videophy2_eval._promote_rank_markers copies; clean both.
        rank_marker_prefixes = [
            ".cluster_videophy2_gen_rank_",
            ".cluster_eval_gen_rank_",
        ]
    else:
        markers = [
            ".cluster_eval_download_done",
            ".cluster_eval_generation_done",
            ".cluster_eval_job_done",
            ".cluster_eval_phase2_args.json",
        ]
        rank_marker_prefixes = [".cluster_eval_gen_rank_"]
    import glob as glob_mod
    for m in markers:
        p = os.path.join(work_dir, m)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass
    for prefix in rank_marker_prefixes:
        for p in glob_mod.glob(os.path.join(work_dir, f"{prefix}*_done")):
            try:
                os.remove(p)
            except OSError:
                pass


def _eval_work_dir(benchmark: str) -> str:
    """Return the work directory used by the selected eval benchmark."""
    if benchmark == "vbench":
        return os.path.join(_REPO_ROOT, "eval_work_vbench")
    if benchmark == "videophy2":
        return os.path.join(_REPO_ROOT, "eval_work_videophy2")
    return os.path.join(_REPO_ROOT, "eval_work")


def _eval_summary_path(benchmark: str, cfg: dict = None) -> str:
    """Return the summary file path produced by the selected eval benchmark."""
    if benchmark == "vbench":
        result_path = (cfg or {}).get("result_path")
        if result_path:
            return os.path.join(os.path.abspath(result_path), "result_summary.txt")
        return os.path.join(_eval_work_dir("vbench"), "vbench_eval_results_summary.txt")
    if benchmark == "videophy2":
        return os.path.join(_eval_work_dir("videophy2"), "videophy2_eval_results_summary.txt")
    return os.path.join(_eval_work_dir("videophy"), "videophy_eval_results_summary.txt")


# =============================================================================
# Main entry point
# =============================================================================

def _launch_evaluation_after_training(cfg: dict) -> None:
    """
    训练完成后，由 RANK=0 启动评测 subprocess。
    
    这是为了处理集群平台直接以分布式方式调用 main() 的情况。
    训练完成后，分布式进程组已经销毁，需要启动新的 subprocess 来执行评测。
    """
    eval_num_gpus = _resolve_eval_num_gpus(cfg)
    script = os.path.join(_REPO_ROOT, "train", "cluster_train_eval.py")
    
    print(f"[cluster_train_eval] Launching evaluation with {eval_num_gpus} GPUs...", flush=True)
    
    # 构建评测配置
    eval_cfg = {**cfg, "mode": "eval", "num_gpus": eval_num_gpus}
    training_wandb = _read_training_wandb_info()
    if training_wandb.get("run_id"):
        eval_cfg["_training_wandb"] = training_wandb
    
    # 处理评测 checkpoint 路径
    eval_ckpt = cfg.get("eval_checkpoint_path") or cfg.get("eval_checkpoint_path")
    generate_type = _resolve_eval_generate_type(cfg)
    eval_cfg["generate_type"] = generate_type
    
    if eval_ckpt:
        eval_cfg["checkpoint_path"] = eval_ckpt
        print(f"[cluster_train_eval] Using eval_checkpoint_path: {eval_ckpt}", flush=True)
    else:
        base_ckpt = (cfg.get("checkpoint_path") or cfg.get("checkpoint_path") or "").rstrip("/")
        if base_ckpt:
            if generate_type == "lora":
                train_steps = cfg.get("train_steps") or COMMON_EXPERIMENT_PARAMS["lamo"].get("train_steps", 1000)
                auto_eval_ckpt = f"{base_ckpt}/weight_{train_steps}"
                print(f"[cluster_train_eval] Auto-constructed eval checkpoint (LoRA): {auto_eval_ckpt}", flush=True)
            else:
                auto_eval_ckpt = base_ckpt
                print(f"[cluster_train_eval] Auto-constructed eval checkpoint (full finetune): {auto_eval_ckpt}", flush=True)
            eval_cfg["checkpoint_path"] = auto_eval_ckpt
    
    # 启动评测 subprocess
    env = {**os.environ, "CLUSTER_TRAIN_EVAL_CONFIG": json.dumps(eval_cfg)}
    # 使用不同的 MASTER_PORT 避免与训练冲突
    eval_master_port = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
    env["MASTER_PORT"] = eval_master_port
    # 重置分布式环境变量，让评测 subprocess 重新初始化
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    env.pop("WORLD_SIZE", None)
    
    if eval_num_gpus <= _SINGLE_NODE_MAX_GPUS:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1", "--nproc_per_node", str(eval_num_gpus),
            script,
        ]
        eval_code = _run_subprocess_streaming(cmd, cwd=_REPO_ROOT, env=env)
    else:
        nnodes = (eval_num_gpus + _SINGLE_NODE_MAX_GPUS - 1) // _SINGLE_NODE_MAX_GPUS
        nproc_per_node = _SINGLE_NODE_MAX_GPUS
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", str(nnodes), "--nproc_per_node", str(nproc_per_node),
            "--node_rank", "0", "--master_addr", master_addr, "--master_port", eval_master_port,
            script,
        ]
        eval_code = _run_subprocess_streaming(cmd, cwd=_REPO_ROOT, env=env)
    
    if eval_code != 0:
        print(f"[cluster_train_eval] Evaluation failed with code {eval_code}", flush=True)
        sys.exit(eval_code)
    
    print(f"[cluster_train_eval] Evaluation completed successfully.", flush=True)
    summary_path = _eval_summary_path(_resolve_eval_benchmark(eval_cfg), eval_cfg)
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            text = f.read()
        print("[cluster_train_eval] === Evaluation result_summary ===", flush=True)
        print(text, flush=True)
        print("[cluster_train_eval] === End result_summary ===", flush=True)


def _run_train_eval_launcher(cfg: dict) -> None:
    """
    train_eval 模式的启动器：先训练后评测。
    
    流程：
    1. 用 num_gpus 个 GPU 运行训练（mode 临时设为 "train"）
    2. 训练完成后，用 eval_num_gpus 个 GPU 运行评测（mode 临时设为 "eval"）
    
    这样每个阶段的 worker 只做单一任务，避免分布式进程组的复用问题。
    """
    num_gpus = _resolve_num_gpus(cfg)
    eval_num_gpus = _resolve_eval_num_gpus(cfg)
    script = os.path.join(_REPO_ROOT, "train", "cluster_train_eval.py")
    
    print(f"[cluster_train_eval] Mode: train_eval (train with {num_gpus} GPUs, eval with {eval_num_gpus} GPUs)", flush=True)
    
    # ========== 阶段一：训练 ==========
    print(f"[cluster_train_eval] Phase 1: Training with {num_gpus} GPUs...", flush=True)
    train_cfg = {**cfg, "mode": "train"}  # 临时设为 train 模式
    
    if num_gpus <= _SINGLE_NODE_MAX_GPUS:
        env = {**os.environ, "CLUSTER_TRAIN_EVAL_CONFIG": json.dumps(train_cfg)}
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1", "--nproc_per_node", str(num_gpus),
            script,
        ]
        train_code = subprocess.run(cmd, cwd=_REPO_ROOT, env=env).returncode
    else:
        # 多机训练
        env = {**os.environ, "CLUSTER_TRAIN_EVAL_CONFIG": json.dumps(train_cfg)}
        nnodes = int(os.environ.get("NNODES", (num_gpus + _SINGLE_NODE_MAX_GPUS - 1) // _SINGLE_NODE_MAX_GPUS))
        node_rank = int(os.environ.get("NODE_RANK", 0))
        nproc_per_node = int(os.environ.get("NPROC_PER_NODE", _SINGLE_NODE_MAX_GPUS))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", str(nnodes), "--nproc_per_node", str(nproc_per_node),
            "--node_rank", str(node_rank), "--master_addr", master_addr, "--master_port", master_port,
            script,
        ]
        train_code = subprocess.run(cmd, cwd=_REPO_ROOT, env=env).returncode
    
    if train_code != 0:
        print(f"[cluster_train_eval] Training failed with code {train_code}, skipping evaluation", flush=True)
        sys.exit(train_code)
    
    print(f"[cluster_train_eval] Phase 1 completed. Training successful.", flush=True)

    # Capture training wandb run info (written to file by ptd.py destroy)
    training_wandb = _read_training_wandb_info()
    if training_wandb.get("run_id"):
        print(f"[cluster_train_eval] Captured training wandb run: "
              f"id={training_wandb['run_id']}, project={training_wandb.get('project')}", flush=True)
    
    # ========== 阶段二：评测 ==========
    print(f"[cluster_train_eval] Phase 2: Evaluation with {eval_num_gpus} GPUs...", flush=True)
    
    # 构建评测配置
    eval_cfg = {**cfg, "mode": "eval", "num_gpus": eval_num_gpus}
    if training_wandb.get("run_id"):
        eval_cfg["_training_wandb"] = training_wandb
    
    # 处理评测 checkpoint 路径：
    # 1. 若指定了 eval_checkpoint_path，用作评测的 checkpoint_path
    # 2. 否则根据 generate_type 自动构造：
    #    - lora: checkpoint_path/weight_{train_steps}
    #    - baseline (full finetune): checkpoint_path 直接使用
    eval_ckpt = cfg.get("eval_checkpoint_path") or cfg.get("eval_checkpoint_path")
    generate_type = _resolve_eval_generate_type(cfg)
    eval_cfg["generate_type"] = generate_type
    
    if eval_ckpt:
        eval_cfg["checkpoint_path"] = eval_ckpt
        print(f"[cluster_train_eval] Using eval_checkpoint_path: {eval_ckpt}", flush=True)
    else:
        base_ckpt = (cfg.get("checkpoint_path") or cfg.get("checkpoint_path") or "").rstrip("/")
        if base_ckpt:
            if generate_type == "lora":
                # LoRA: 权重保存在 weight_{train_steps}/ 子目录
                train_steps = cfg.get("train_steps") or COMMON_EXPERIMENT_PARAMS["lamo"].get("train_steps", 1000)
                auto_eval_ckpt = f"{base_ckpt}/weight_{train_steps}"
                print(f"[cluster_train_eval] Auto-constructed eval checkpoint (LoRA): {auto_eval_ckpt}", flush=True)
            else:
                # Full finetune (baseline): 直接使用 checkpoint_path
                auto_eval_ckpt = base_ckpt
                print(f"[cluster_train_eval] Auto-constructed eval checkpoint (full finetune): {auto_eval_ckpt}", flush=True)
            eval_cfg["checkpoint_path"] = auto_eval_ckpt
    
    # 评测使用单机（eval_num_gpus <= 8）
    if eval_num_gpus <= _SINGLE_NODE_MAX_GPUS:
        env = {**os.environ, "CLUSTER_TRAIN_EVAL_CONFIG": json.dumps(eval_cfg)}
        # 使用不同的 MASTER_PORT 避免与训练冲突
        eval_master_port = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
        env["MASTER_PORT"] = eval_master_port
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1", "--nproc_per_node", str(eval_num_gpus),
            script,
        ]
        eval_code = _run_subprocess_streaming(cmd, cwd=_REPO_ROOT, env=env)
    else:
        env = {**os.environ, "CLUSTER_TRAIN_EVAL_CONFIG": json.dumps(eval_cfg)}
        eval_master_port = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
        env["MASTER_PORT"] = eval_master_port
        nnodes = (eval_num_gpus + _SINGLE_NODE_MAX_GPUS - 1) // _SINGLE_NODE_MAX_GPUS
        node_rank = int(os.environ.get("NODE_RANK", 0))
        if node_rank >= nnodes:
            print(f"[cluster_train_eval] Node {node_rank} not needed for evaluation, exiting", flush=True)
            sys.exit(0)
        nproc_per_node = min(_SINGLE_NODE_MAX_GPUS, eval_num_gpus - node_rank * _SINGLE_NODE_MAX_GPUS)
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", str(nnodes), "--nproc_per_node", str(nproc_per_node),
            "--node_rank", str(node_rank), "--master_addr", master_addr, "--master_port", eval_master_port,
            script,
        ]
        eval_code = _run_subprocess_streaming(cmd, cwd=_REPO_ROOT, env=env)
    
    if eval_code != 0:
        print(f"[cluster_train_eval] Evaluation failed with code {eval_code}", flush=True)
        sys.exit(eval_code)
    
    print(f"[cluster_train_eval] Phase 2 completed. Evaluation successful.", flush=True)
    print(f"[cluster_train_eval] train_eval mode completed successfully.", flush=True)
    sys.exit(0)


_captured_wandb_run_info: dict = {}


def _install_wandb_init_hook():
    """Monkey-patch ``wandb.init`` to record the run's id/project/entity.

    This is called *before* training so that we can capture the info before
    ``wandb.finish()`` clears it.  Avoids modifying the finetrainers package.
    Returns the original ``wandb.init`` so the caller can restore it later.
    """
    try:
        import wandb as _wb
        _orig = _wb.init

        def _hooked_init(*args, **kwargs):
            run = _orig(*args, **kwargs)
            if run is not None and "run_id" not in _captured_wandb_run_info:
                _captured_wandb_run_info.update({
                    "run_id": run.id,
                    "project": run.project,
                    "entity": run.entity or "",
                })
            return run

        _wb.init = _hooked_init
        return _orig
    except ImportError:
        return None


def _uninstall_wandb_init_hook(orig):
    """Restore the original ``wandb.init``."""
    if orig is not None:
        try:
            import wandb as _wb
            _wb.init = orig
        except Exception:
            pass


def _read_training_wandb_info() -> dict:
    """Return training wandb run info captured by the init hook, env vars, or file."""
    if _captured_wandb_run_info.get("run_id"):
        return dict(_captured_wandb_run_info)
    run_id = os.environ.get("_FINETRAINERS_WANDB_RUN_ID")
    if run_id:
        return {
            "run_id": run_id,
            "project": os.environ.get("_FINETRAINERS_WANDB_PROJECT", ""),
            "entity": os.environ.get("_FINETRAINERS_WANDB_ENTITY", ""),
        }
    info_path = os.path.join(_REPO_ROOT, ".training_wandb_info.json")
    if os.path.isfile(info_path):
        try:
            with open(info_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main(config=None, experiment_tracker=None):
    """
    统一入口。根据 config["mode"] 决定执行训练、评测或两者：
      - mode: "train"      → 仅训练
      - mode: "eval"       → 仅评测
      - mode: "train_eval" → 先训练后评测
    
    config source:
    - direct function argument;
    - CLUSTER_TRAIN_EVAL_CONFIG environment variable containing JSON.
    """
    if config is None:
        raw = os.environ.get("CLUSTER_TRAIN_EVAL_CONFIG")
        if raw:
            try:
                config = json.loads(raw)
            except Exception:
                config = {}
    cfg = config if isinstance(config, dict) else {}

    # Optionally forward tracker metadata to evaluation.
    if experiment_tracker is not None and "experiment_tracker" not in cfg:
        if isinstance(experiment_tracker, dict):
            cfg["experiment_tracker"] = experiment_tracker
        else:
            try:
                cfg["experiment_tracker"] = {
                    "wandb": {
                        "entity": getattr(experiment_tracker, "entity", None),
                        "project": getattr(experiment_tracker, "project", None),
                    }
                }
            except Exception:
                pass

    mode = cfg.get("mode", "train").lower()
    world_size = os.environ.get("WORLD_SIZE")
    is_worker = _is_distributed_worker_env()
    
    if mode == "train_eval" or mode == "both":
        if is_worker:
            # 集群平台直接以分布式方式调用 main()，WORLD_SIZE > 1
            # 训练后所有 rank 直接参与评测（不再用 subprocess），保持进程存活以确保集群日志收集正常
            print(f"[cluster_train_eval] Mode: train_eval (distributed entry, WORLD_SIZE={world_size})", flush=True)

            # Save Ray's stdout/stderr BEFORE training.  Training frameworks
            # (accelerate / finetrainers) replace sys.stdout (e.g. /dev/null
            # for non-main ranks, or custom wrappers for rank 0).  We must
            # restore Ray's originals so the cluster log collector keeps working.
            _saved_stdout = sys.stdout
            _saved_stderr = sys.stderr

            # Hook wandb.init so we capture the training run's id/project/entity
            # *before* parallel_backend.destroy() calls wandb.finish().
            _orig_wandb_init = _install_wandb_init_hook()

            _run_training(cfg)

            _uninstall_wandb_init_hook(_orig_wandb_init)
            sys.stdout = _saved_stdout
            sys.stderr = _saved_stderr

            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            eval_num_gpus = _resolve_eval_num_gpus(cfg)

            # Synchronize all ranks, then tear down training's distributed group
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
            except Exception:
                pass

            print(f"[cluster_train_eval] RANK {rank}: Training completed, entering evaluation phase (eval_num_gpus={eval_num_gpus})...", flush=True)

            # Capture training wandb run info so eval can resume the same run
            training_wandb = {}
            if rank == 0:
                training_wandb = _read_training_wandb_info()
                if training_wandb.get("run_id"):
                    print(f"[cluster_train_eval] Captured training wandb run: "
                          f"id={training_wandb['run_id']}, project={training_wandb.get('project')}", flush=True)
                else:
                    print("[cluster_train_eval] No training wandb run info found, eval will create new run", flush=True)

            # Ranks on other nodes (>= eval_num_gpus) cannot see the marker
            # file on node 0, so let them exit immediately instead of blocking.
            if rank >= eval_num_gpus:
                print(f"[cluster_train_eval] RANK {rank}: not needed for evaluation, exiting.", flush=True)
                return

            # Free GPU cache so evaluation subprocesses have enough memory
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            benchmark = _resolve_eval_benchmark(cfg)

            eval_cfg = _prepare_eval_config_after_training(cfg, rank=rank)

            # Inject training wandb info so eval can resume the same run
            if training_wandb.get("run_id"):
                eval_cfg["_training_wandb"] = training_wandb

            # Ranks 0..eval_num_gpus-1: participate in evaluation via cluster_eval.main()
            _run_evaluation(eval_cfg)

            if rank == 0:
                summary_path = _eval_summary_path(benchmark, eval_cfg)
                if os.path.isfile(summary_path):
                    with open(summary_path) as f:
                        text = f.read()
                    print("[cluster_train_eval] === Evaluation result_summary ===", flush=True)
                    print(text, flush=True)
                    print("[cluster_train_eval] === End result_summary ===", flush=True)
        else:
            # Launcher：依次启动训练和评测
            _run_train_eval_launcher(cfg)
    elif mode == "eval" or mode == "evaluation":
        print(f"[cluster_train_eval] Mode: evaluation", flush=True)
        _run_evaluation(cfg)
    else:
        print(f"[cluster_train_eval] Mode: training", flush=True)
        _run_training(cfg)


if __name__ == "__main__":
    main()
