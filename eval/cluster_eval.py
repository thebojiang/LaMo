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
VideoPhy evaluation: 生成视频 -> SA/PC 评测 -> result_summary + wandb 打点。

支持两种基座模型（通过 model_type 配置）：
  - lamo: LaMo 模型（默认）
  - cogvideox: CogVideoX baseline

支持两种训练类型（通过 generate_type 配置）：
  - Full finetune: generate_type=baseline。checkpoint_path 为本地 checkpoint 根（含 transformer/、scheduler/）；
    可选 pretrained_model_path 作为本地/public base，再 overlay transformer+scheduler。
  - LoRA: generate_type=lora。需 pretrained_model_path（base，本地目录或公开模型 ID），checkpoint_path 指向本地 weight_2000 等目录
    （含 pytorch_lora_weights.safetensors 或 adapter_model.safetensors + scheduler/），
    并配置 lora_name, lora_rank, lora_alpha。

evaluation_fn_config 主要项：
  - model_type: lamo | cogvideox（默认 lamo）
  - checkpoint_path: 训练产出本地路径（full finetune: cluster 含 transformer/scheduler；lora: weight_2000 目录）
  - result_path: 本地结果输出目录
  - input_csv, entailment_checkpoint（或 videocon_checkpoint）, pretrained_model_path（lora 必填，ft 可选）
  - generate_type: baseline | lora
  - lora_name, lora_rank, lora_alpha（仅 lora）
  - num_frames: 视频帧数（默认 49）
  - fps: 帧率（默认 8）

多进程时：阶段一仅 rank 0 下载所有权重到本地并写 .cluster_eval_download_done；
rank 1-7 等待 download_done。阶段二：rank 0-7 并行运行 videophy_cluster_generate（各 rank 处理 index %% world_size == rank 的 shard），
各 rank 写 .cluster_eval_gen_rank_N_done；rank 0 在自身生成结束后等待其余 rank 的 done marker，合并 manifest，再跑 entailment + 上传 + wandb，
写 .cluster_eval_generation_done 与 .cluster_eval_job_done。其他 rank 等 generation_done、job_done 后退出。
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional, Tuple

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# -----------------------------------------------------------------------------
# Environment variables: same as train_wan_full_finetune.sh (and reference train_stream.py).
# Platform can override via required_environment_variables in model_config.yaml.
# -----------------------------------------------------------------------------
os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("FINETRAINERS_LOG_LEVEL", "DEBUG")

logging.getLogger("urllib3").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Fix: Prevent transformers from using broken Apex FusedLayerNorm on cluster
# The cluster has Apex installed but CUDA extension not compiled, causing:
# "No module named 'fused_layer_norm_cuda'"
# Solution: Block apex.normalization import BEFORE any transformers import
# This forces transformers to use standard PyTorch LayerNorm instead
# -----------------------------------------------------------------------------
sys.modules["apex"] = None
sys.modules["apex.normalization"] = None
logging.info("Blocked apex.normalization to prevent FusedLayerNorm issues")

# -----------------------------------------------------------------------------
# Model path defaults
# -----------------------------------------------------------------------------
_COGVIDEOX_2B_MODEL = os.environ.get("LAMO_COGVIDEOX_2B_MODEL", "THUDM/CogVideoX-2b")
_COGVIDEOX_5B_MODEL = os.environ.get("LAMO_COGVIDEOX_5B_MODEL", "THUDM/CogVideoX-5b")

# 模型默认配置：分辨率由 tests/_generation_common.py 内部根据 model_type 自动决定
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


def _load_config():
    raw = os.environ.get("CLUSTER_EVAL_CONFIG")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


def _resolve_base_model_path(source_path: str, download_target: str) -> str:
    """
    解析 base 模型路径：本地目录直接返回其绝对路径。公开模型 ID 原样返回，
    由 diffusers/transformers 自行处理。
    返回供后续使用的 model_path。
    """
    if not source_path or not str(source_path).strip():
        return download_target
    source_path = source_path.strip().rstrip("/")
    if not os.path.isdir(source_path):
        print(f"[cluster_eval] Using public model id as-is: {source_path}", flush=True)
        return source_path
    print(f"[cluster_eval] Using local base as-is: {source_path}", flush=True)
    return os.path.abspath(source_path)


def _build_eval_model_dir_from_local_base(work_dir: str, local_base: str) -> str:
    """
    用本地 base 建一个评测用权重目录：pretrain 部分（vae、text_encoder、transformer 内原有文件等）软链到 base，
    transformer 与 scheduler 为实体目录，便于后续写入 LoRA 和 checkpoint 的 scheduler。
    返回 work_dir/model 的路径。
    """
    base = os.path.abspath(local_base)
    model_path = os.path.join(work_dir, "model")
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    for name in os.listdir(base):
        src = os.path.join(base, name)
        dst = os.path.join(model_path, name)
        if name == "transformer":
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                s = os.path.join(src, f)
                d = os.path.join(dst, f)
                if os.path.isfile(s):
                    os.symlink(s, d)
                else:
                    os.symlink(s, d)  # 子目录也软链
            continue
        if name == "scheduler":
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                os.symlink(os.path.join(src, f), os.path.join(dst, f))
            continue
        os.symlink(src, dst)
    print(f"[cluster_eval] Built eval model dir with symlinks to local base: {model_path}", flush=True)
    return model_path


def _copy_summary_to_output_dir(local_path: str, output_dir: str) -> None:
    if not output_dir or not os.path.isfile(local_path):
        return
    output_dir = output_dir.rstrip("/")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(local_path, os.path.join(output_dir, "result_summary.txt"))


def _setup_model_full_finetune(work_dir: str, checkpoint_path: str, pretrained_path: str) -> Tuple[str, Optional[str]]:
    """
    Full finetune: 若 pretrained_path 已配置，则使用 base 再 overlay transformer/scheduler；
    否则认为 checkpoint_path 为完整本地模型目录。
    返回 (model_path, physical_module_path)。
    """
    if pretrained_path:
        print(f"[cluster_eval] Full finetune: loading base from {pretrained_path}", flush=True)
        pretrained_path = pretrained_path.strip().rstrip("/")
        if os.path.isdir(pretrained_path):
            model_path = _build_eval_model_dir_from_local_base(work_dir, pretrained_path)
        else:
            model_path = _resolve_base_model_path(pretrained_path, os.path.join(work_dir, "model"))
        for sub in ("transformer", "scheduler"):
            src = os.path.join(checkpoint_path.rstrip("/"), sub)
            dst = os.path.join(model_path, sub)
            # 完全覆盖：先清空评测目录下该子目录，再复制 checkpoint 内容，保证仅含 checkpoint 的 transformer/scheduler
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            print(f"[cluster_eval] Overlaying (replace) {src} -> {dst}", flush=True)
            shutil.copytree(src, dst)
    else:
        model_path = os.path.abspath(checkpoint_path)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Full finetune checkpoint not found: {model_path}")

    physical_module_path = None
    pred_path = os.path.join(checkpoint_path.rstrip("/"), "predictor.safetensors")
    if os.path.isfile(pred_path):
        physical_module_path = os.path.abspath(pred_path)
        print(f"[cluster_eval] Full finetune: using predictor.safetensors from {physical_module_path}", flush=True)
    return model_path, physical_module_path


def _setup_model_lora(
    work_dir: str,
    checkpoint_path: str,
    pretrained_path: str,
    model_type: str = "cogvideox",
    predictor_path: Optional[str] = None,
) -> Tuple[str, str, Optional[str]]:
    """
    LoRA: base 为本地目录则新建 work_dir/model，pretrain 部分软链到 base，仅 LoRA 与 scheduler 为实体文件。
    返回 (model_path, lora_path, physical_module_path)。
    lora_path 为 transformer 目录（含 pytorch_lora_weights.safetensors）；当 model_type 为 "lamo" 时
    会从 checkpoint 或 predictor_path 复制 predictor.safetensors 到 transformer 目录，physical_module_path 为该文件路径。
    """
    if not pretrained_path:
        raise ValueError("LoRA evaluation requires pretrained_model_path (base model).")
    weight_path = checkpoint_path.rstrip("/")
    print(f"[cluster_eval] LoRA: loading base from {pretrained_path}", flush=True)
    pretrained_path = pretrained_path.strip().rstrip("/")
    if not os.path.isdir(pretrained_path):
        raise FileNotFoundError(
            "LoRA evaluation requires a local base model directory. "
            "Set pretrained_model_path to <PATH_TO_PRETRAINED_MODEL>."
        )
    model_path = _build_eval_model_dir_from_local_base(work_dir, pretrained_path)
    checkpoint_dir = os.path.abspath(weight_path)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"LoRA checkpoint dir not found: {checkpoint_dir}")
    transformer_dir = os.path.join(model_path, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    lora_file = None
    lora_dir = None
    lora_candidate_dirs = [checkpoint_dir, os.path.join(checkpoint_dir, "transformer")]
    seen_lora_dirs = set()
    lora_candidate_dirs = [
        d for d in lora_candidate_dirs
        if not (d in seen_lora_dirs or seen_lora_dirs.add(d))
    ]
    for name in ("pytorch_lora_weights.safetensors", "adapter_model.safetensors"):
        for candidate_dir in lora_candidate_dirs:
            src = os.path.join(candidate_dir, name)
            if os.path.isfile(src):
                dst = os.path.join(transformer_dir, name)
                shutil.copy2(src, dst)
                lora_file = dst
                lora_dir = candidate_dir
                print(f"[cluster_eval] LoRA: using weights from {src}", flush=True)
                break
        if lora_file:
            break
    if not lora_file:
        raise FileNotFoundError(
            "LoRA weight dir has no pytorch_lora_weights.safetensors or adapter_model.safetensors: "
            f"{checkpoint_dir} (also checked transformer/)"
        )
    scheduler_path = os.path.join(weight_path, "scheduler")
    scheduler_dir = os.path.join(model_path, "scheduler")
    os.makedirs(scheduler_dir, exist_ok=True)
    if os.path.isdir(scheduler_path):
        if os.path.isdir(scheduler_dir):
            shutil.rmtree(scheduler_dir)
        shutil.copytree(scheduler_path, scheduler_dir)

    physical_module_path = None
    if model_type == "lamo":
        pred_candidates = []
        if predictor_path:
            predictor_path = predictor_path.strip().rstrip("/")
            if os.path.isdir(predictor_path):
                pred_candidates.append(os.path.join(predictor_path, "predictor.safetensors"))
            else:
                pred_candidates.append(predictor_path)
        pred_candidates.extend([
            os.path.join(lora_dir, "predictor.safetensors"),
            os.path.join(checkpoint_dir, "predictor.safetensors"),
            os.path.join(checkpoint_dir, "transformer", "predictor.safetensors"),
        ])
        src_pred = next((p for p in pred_candidates if p and os.path.isfile(p)), None)
        if src_pred:
            dst_pred = os.path.join(transformer_dir, "predictor.safetensors")
            shutil.copy2(src_pred, dst_pred)
            physical_module_path = dst_pred
            print(f"[cluster_eval] copied predictor.safetensors to {dst_pred}", flush=True)
        else:
            print(f"[cluster_eval] no predictor.safetensors found in {pred_candidates}", flush=True)

    lora_path_for_gen = transformer_dir
    return model_path, lora_path_for_gen, physical_module_path


def _verify_lamo_weights(
    model_type: str,
    lora_path: Optional[str],
    physical_module_path: Optional[str],
    cfg: dict,
) -> None:
    """
    Verify physical module and LoRA weight files after download (rank 0 only).
    All output uses print(..., flush=True) so it appears in wandb Logs panel.
    """
    print("=" * 60, flush=True)
    print(f"[cluster_eval] Weight verification for model_type={model_type}", flush=True)
    print("=" * 60, flush=True)

    # --- LoRA weights ---
    if lora_path:
        lora_dir = lora_path if os.path.isdir(lora_path) else os.path.dirname(lora_path)
        lora_file = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")
        if os.path.isfile(lora_file):
            size_mb = os.path.getsize(lora_file) / (1024 * 1024)
            print(f"[cluster_eval] [lora] FOUND pytorch_lora_weights.safetensors ({size_mb:.2f} MB)", flush=True)
            print(f"[cluster_eval] [lora]   path: {lora_file}", flush=True)
            print(f"[cluster_eval] [lora]   config: adapter={cfg.get('lora_name', 'lora_adapter')}, "
                  f"rank={cfg.get('lora_rank', 128)}, alpha={cfg.get('lora_alpha', 64)}, "
                  f"scale={int(cfg.get('lora_alpha', 64)) / max(int(cfg.get('lora_rank', 128)), 1):.4f}", flush=True)
        else:
            print(f"[cluster_eval] [lora] WARNING: pytorch_lora_weights.safetensors NOT FOUND at {lora_dir}", flush=True)
    else:
        print("[cluster_eval] [lora] lora_path is None (not using LoRA)", flush=True)

    # --- lamo predictor weights ---
    if model_type != "lamo":
        print(f"[cluster_eval] [phys] model_type={model_type}, no predictor sidecar expected", flush=True)
        print("=" * 60, flush=True)
        return
    if not physical_module_path or not os.path.isfile(physical_module_path):
        print(f"[cluster_eval] [phys] WARNING: predictor.safetensors NOT FOUND (path={physical_module_path})", flush=True)
        print("=" * 60, flush=True)
        return
    size_mb = os.path.getsize(physical_module_path) / (1024 * 1024)
    print(f"[cluster_eval] [phys] FOUND predictor.safetensors ({size_mb:.2f} MB)", flush=True)
    print(f"[cluster_eval] [phys]   path: {physical_module_path}", flush=True)

    try:
        from safetensors.torch import load_file
        import torch

        state = load_file(physical_module_path)
        print(f"[cluster_eval] [phys] Total keys in safetensors: {len(state)}", flush=True)

        version_prefix = "physical_lamo."
        prefix_label = "physical_lamo.*"
        prefixed = [k for k in state.keys() if k.startswith(version_prefix)]
        proj_keys = [k for k in state.keys() if k.startswith("physical_global_proj.")]
        other_keys = [k for k in state.keys()
                      if not k.startswith(version_prefix) and not k.startswith("physical_global_proj.")]
        print(f"[cluster_eval] [phys]   predictor keys ({prefix_label}): {len(prefixed)}", flush=True)
        print(f"[cluster_eval] [phys]   'physical_global_proj.*' keys: {len(proj_keys)}", flush=True)
        if other_keys:
            print(f"[cluster_eval] [phys]   Other keys: {len(other_keys)} -> {sorted(other_keys)[:5]}", flush=True)

        if prefixed:
            phys = {k[len(version_prefix):]: v for k, v in state.items() if k.startswith(version_prefix)}
        else:
            phys = {k: v for k, v in state.items() if not k.startswith("physical_global_proj.")}
            print(f"[cluster_eval] [phys]   No version prefix found, using {len(phys)} non-proj keys", flush=True)

        # Detect dimensions from weights
        if "quantizer.embedding.weight" in phys:
            emb_w = phys["quantizer.embedding.weight"]
            print(f"[cluster_eval] [phys]   Detected codebook_size={emb_w.shape[0]}, "
                  f"token_dim={emb_w.shape[1]} (from quantizer.embedding.weight)", flush=True)
        if "encoder.head.weight" in phys:
            head_w = phys["encoder.head.weight"]
            print(f"[cluster_eval] [phys]   Detected encoder token_dim={head_w.shape[0]} "
                  f"(from encoder.head.weight)", flush=True)

        # Sample weight stats for sanity check
        sample_key = "encoder.head.weight"
        if sample_key in phys:
            w = phys[sample_key]
            print(f"[cluster_eval] [phys]   Sample {sample_key}: shape={list(w.shape)}, "
                  f"mean={w.float().mean():.6f}, std={w.float().std():.6f}", flush=True)
            is_zero = (w.float().abs().max().item() < 1e-8)
            if is_zero:
                print(f"[cluster_eval] [phys]   WARNING: {sample_key} is all zeros — weights may not have been trained!", flush=True)

        # physical_global_proj verification
        if proj_keys:
            for pk in sorted(proj_keys):
                pv = state[pk]
                print(f"[cluster_eval] [phys]   {pk}: shape={list(pv.shape)}, "
                      f"mean={pv.float().mean():.6f}, std={pv.float().std():.6f}", flush=True)
        else:
            print(f"[cluster_eval] [phys]   INFO: no physical_global_proj keys found; "
                  f"this is expected for predictor-only LaMo exports used by the "
                  f"current inference path.", flush=True)

        # Config consistency check
        cfg_token_dim = cfg.get("physical_token_dim")
        cfg_codebook_size = cfg.get("physical_codebook_size")
        if cfg_token_dim is not None and "quantizer.embedding.weight" in phys:
            actual_dim = phys["quantizer.embedding.weight"].shape[1]
            if int(cfg_token_dim) != actual_dim:
                print(f"[cluster_eval] [phys]   MISMATCH: config physical_token_dim={cfg_token_dim} "
                      f"vs checkpoint token_dim={actual_dim}", flush=True)
            else:
                print(f"[cluster_eval] [phys]   Config token_dim={cfg_token_dim} matches checkpoint ✓", flush=True)
        if cfg_codebook_size is not None and "quantizer.embedding.weight" in phys:
            actual_cb = phys["quantizer.embedding.weight"].shape[0]
            if int(cfg_codebook_size) != actual_cb:
                print(f"[cluster_eval] [phys]   MISMATCH: config physical_codebook_size={cfg_codebook_size} "
                      f"vs checkpoint codebook_size={actual_cb}", flush=True)
            else:
                print(f"[cluster_eval] [phys]   Config codebook_size={cfg_codebook_size} matches checkpoint ✓", flush=True)

        print(f"[cluster_eval] [phys] Weight verification PASSED", flush=True)
    except Exception as e:
        print(f"[cluster_eval] [phys] Weight verification failed with error: {e}", flush=True)

    print("=" * 60, flush=True)


def _resolve_entailment_checkpoint(entailment_checkpoint: str, work_dir: str) -> str:
    """解析 entailment checkpoint：本地路径或公开模型 ID直接返回。"""
    raw = (entailment_checkpoint or "").strip()
    if not raw:
        return raw
    return raw


def _parse_summary_file(summary_path: str) -> dict:
    """Parse result_summary.txt into a dict for wandb logging."""
    out = {"videophy/avg_sa": None, "videophy/avg_pc": None}
    if not os.path.isfile(summary_path):
        return out
    with open(summary_path) as f:
        text = f.read()
    # Overall
    m = re.search(r"Average SA score:\s*([\d.]+)", text)
    if m:
        out["videophy/avg_sa"] = float(m.group(1))
    m = re.search(r"Average PC score:\s*([\d.]+)", text)
    if m:
        out["videophy/avg_pc"] = float(m.group(1))
    # By states_of_matter (e.g. "  solid: avg_SA=0.xx, avg_PC=0.xx")
    for line in text.splitlines():
        if "avg_SA=" in line and "avg_PC=" in line:
            parts = line.strip().split(":")
            if len(parts) >= 2:
                label = parts[0].strip()
                rest = parts[1]
                sa_m = re.search(r"avg_SA=([\d.]+)", rest)
                pc_m = re.search(r"avg_PC=([\d.]+)", rest)
                if sa_m:
                    out[f"videophy/by_som/{label}_avg_sa"] = float(sa_m.group(1))
                if pc_m:
                    out[f"videophy/by_som/{label}_avg_pc"] = float(pc_m.group(1))
    # By complexity
    for line in text.splitlines():
        if "complexity=" in line and "avg_SA=" in line:
            comp_m = re.search(r"complexity=(\d+):\s*avg_SA=([\d.]+).*avg_PC=([\d.]+)", line)
            if comp_m:
                c, sa, pc = comp_m.group(1), comp_m.group(2), comp_m.group(3)
                out[f"videophy/by_complexity/{c}_avg_sa"] = float(sa)
                out[f"videophy/by_complexity/{c}_avg_pc"] = float(pc)
    return out


def _get_wandb_run(config: dict):
    """Init wandb if needed and return the run (None if wandb unavailable or not main rank).

    Priority:
    1. Resume the training wandb run (if ``_training_wandb`` info is present in config).
       This way evaluation metrics appear in the SAME run as training metrics.
    2. Create a new eval run using ``experiment_tracker`` from the platform config.
    3. Fallback to ``WANDB_PROJECT``/``WANDB_ENTITY`` env vars or hardcoded defaults.
    """
    if not _is_main_rank():
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes") or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None

    # Already initialized — reuse
    if wandb.run is not None:
        return wandb.run

    try:
        # Strategy 1: resume the training wandb run (train_eval mode)
        training_wb = config.get("_training_wandb") or {}
        if training_wb.get("run_id"):
            print(f"[cluster_eval] Attempting to resume training wandb run: "
                  f"id={training_wb['run_id']}, project={training_wb.get('project')}", flush=True)
            wandb.init(
                id=training_wb["run_id"],
                project=training_wb.get("project") or "LaMo",
                entity=training_wb.get("entity") or None,
                resume="allow",
                reinit=True,
            )
            if wandb.run is not None:
                print(f"[cluster_eval] Resumed training wandb run: {wandb.run.url}", flush=True)
                return wandb.run

        # Strategy 2: new eval run with platform's experiment_tracker config
        tracker = (config.get("experiment_tracker") or {}).get("wandb") or {}
        entity = tracker.get("entity") or os.environ.get("WANDB_ENTITY")
        project = tracker.get("project") or os.environ.get("WANDB_PROJECT") or "LaMo"
        variant = config.get("variant", config.get("model_type", "eval"))
        run_name = config.get("name") or f"videophy-eval-{variant}"

        print(f"[cluster_eval] Creating new eval wandb run: project={project}, entity={entity}, name={run_name}", flush=True)
        wandb.init(project=project, entity=entity, name=run_name, job_type="eval", reinit=True)
        if wandb.run is not None:
            print(f"[cluster_eval] Created eval wandb run: {wandb.run.url}", flush=True)
        return wandb.run
    except Exception as e:
        print(f"[cluster_eval] wandb init failed: {e}", flush=True)
        return None


def _log_to_wandb(metrics: dict, summary_path: str, config: dict) -> None:
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes") or os.environ.get("WANDB_MODE") == "disabled":
        print("[cluster_eval] wandb disabled, skip logging", flush=True)
        return
    try:
        import wandb
    except ImportError:
        print("[cluster_eval] wandb not installed, skip logging", flush=True)
        return
    # Reuse existing run if already inited (e.g. by Phase 2 progress logging)
    if wandb.run is None:
        _get_wandb_run(config)
    if wandb.run is None:
        print("[cluster_eval] wandb run is None after init attempt, skipping wandb logging", flush=True)
        return
    try:
        # Log numeric metrics
        to_log = {k: v for k, v in metrics.items() if v is not None}
        if to_log:
            wandb.log(to_log)
            print(f"[cluster_eval] Logged {len(to_log)} metrics to wandb: {list(to_log.keys())}", flush=True)
        # Log summary text to wandb summary + print to stdout (shows in wandb Logs panel)
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary_text = f.read()
            wandb.run.summary["videophy/summary_text"] = summary_text
            print("=" * 60, flush=True)
            print("result_summary.txt", flush=True)
            print("=" * 60, flush=True)
            print(summary_text, flush=True)
            print("=" * 60, flush=True)
        wandb.finish()
        print("[cluster_eval] wandb.finish() completed successfully", flush=True)
    except Exception as e:
        print(f"[cluster_eval] wandb logging/finish failed: {e}", flush=True)


def _log_wandb_progress(config: dict, **kwargs) -> None:
    """Log Phase 2 progress to wandb (no-op if wandb unavailable or not main rank)."""
    if not _is_main_rank():
        return
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes") or os.environ.get("WANDB_MODE") == "disabled":
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        _get_wandb_run(config)
    if wandb.run is not None and kwargs:
        try:
            wandb.log(kwargs)
        except Exception as e:
            print(f"[cluster_eval] wandb progress log failed: {e}", flush=True)


def _is_main_rank() -> bool:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size <= 1 or rank == 0


def _get_rank_and_world_size():
    """Return (rank, world_size) from environment."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, max(1, world_size)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _count_generated_videos(gen_dir: str) -> int:
    """Count video_*.mp4 under gen_dir/rank_* (Phase 2 各 rank 输出目录)."""
    n = 0
    for rank_dir in glob.glob(os.path.join(gen_dir, "rank_*")):
        if os.path.isdir(rank_dir):
            n += len(glob.glob(os.path.join(rank_dir, "video_*.mp4")))
    return n


def _run_cluster_generation_this_rank(cfg, work_dir: str) -> None:
    """Run videophy_cluster_generate.py for this rank (used by non-zero ranks; rank 0 runs via _run_eval_phase)."""
    args_path = os.path.join(work_dir, ".cluster_eval_phase2_args.json")
    if not os.path.isfile(args_path):
        print(f"[cluster_eval] Phase2 args not found: {args_path}, waiting...", flush=True)
        for _ in range(120):
            time.sleep(2)
            if os.path.isfile(args_path):
                break
        if not os.path.isfile(args_path):
            raise FileNotFoundError(f"Phase2 args not found: {args_path}")
    with open(args_path) as f:
        phase2 = json.load(f)
    rank, world_size = _get_rank_and_world_size()
    gen_dir = phase2["gen_dir"]
    manifest_csv = os.path.join(gen_dir, f"manifest.rank_{rank}.csv")
    gen_cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "tests", "videophy_cluster_generate.py"),
        "--input_csv", phase2["input_csv"],
        "--output_dir", gen_dir,
        "--manifest_csv", manifest_csv,
        "--work_dir", work_dir,
        "--model_path", phase2["model_path"],
        "--model_type", phase2["model_type"],
        "--generate_type", phase2["generate_type"],
        "--num_frames", str(phase2["num_frames"]),
        "--fps", str(phase2["fps"]),
        "--seed", "42",
        "--rank", str(rank),
        "--world_size", str(world_size),
    ]
    if phase2.get("lora_path") and phase2["generate_type"] == "lora":
        gen_cmd.extend([
            "--lora_path", phase2["lora_path"],
            "--lora_name", str(phase2["lora_name"]),
            "--lora_rank", str(phase2["lora_rank"]),
            "--lora_alpha", str(phase2["lora_alpha"]),
        ])
    if phase2.get("model_type") == "lamo":
        if phase2.get("physical_module_path"):
            gen_cmd.extend(["--physical_module_path", phase2["physical_module_path"]])
        gen_cmd.extend(["--guidance_lambda", str(phase2.get("guidance_lambda", 15.0))])
        gen_cmd.extend(["--guidance_step_ratio", str(phase2.get("guidance_step_ratio", 0.8))])
        gen_cmd.extend(["--predictor_hidden_channels", str(phase2.get("predictor_hidden_channels", 256))])
        gen_cmd.extend(["--predictor_num_res_blocks", str(phase2.get("predictor_num_res_blocks", 8))])
        gen_cmd.extend(["--predictor_use_se", str(phase2.get("predictor_use_se", 1))])
        gen_cmd.extend(["--predictor_use_prev_delta", str(phase2.get("predictor_use_prev_delta", 0))])
        gen_cmd.extend(["--predictor_use_prompt_cond", str(phase2.get("predictor_use_prompt_cond", 1))])
        gen_cmd.extend(["--prompt_text_dim", str(phase2.get("prompt_text_dim", 4096))])
        gen_cmd.extend(["--prompt_cond_dim", str(phase2.get("prompt_cond_dim", 128))])
    if phase2.get("limit") is not None:
        gen_cmd.extend(["--limit", str(phase2["limit"])])
    if _as_bool(phase2.get("skip_existing"), default=True):
        gen_cmd.append("--skip_existing")
    wpr = int(phase2.get("workers_per_rank", 1))
    if wpr > 1:
        gen_cmd.extend(["--workers_per_rank", str(wpr)])
    subprocess.run(gen_cmd, cwd=_REPO_ROOT, env=os.environ, check=True)


def _wait_for_marker(work_dir: str, marker_name: str, timeout_seconds: int = 14400, poll_interval: int = 15) -> bool:
    """Block until marker file exists or timeout. Non-zero ranks use this to avoid exiting before rank-0 finishes."""
    marker = os.path.join(work_dir, marker_name)
    deadline = time.monotonic() + timeout_seconds
    last_log = 0.0
    while time.monotonic() < deadline:
        if os.path.isfile(marker):
            return True
        # 每 60s 打一次心跳，避免被平台误判为“卡住/idle 异常”；非 rank-0 在此等待是预期行为
        now = time.monotonic()
        if now - last_log >= 60:
            print(f"[cluster_eval] (waiting for {marker_name}...) ", flush=True)
            last_log = now
        time.sleep(poll_interval)
    return False


def main(config=None, experiment_tracker=None):
    """Evaluation entry point."""
    cfg = config if isinstance(config, dict) else _load_config()
    work_dir = os.path.join(_REPO_ROOT, "eval_work")
    os.makedirs(work_dir, exist_ok=True)

    # 非 rank 0：等下载完成 -> 参与 phase 2 并行生成 -> 等 generation_done -> job_done 后退出
    if not _is_main_rank():
        generation_timeout_seconds = int(cfg.get("generation_timeout_seconds", 14400))
        generation_sync_timeout_seconds = int(
            cfg.get("generation_sync_timeout_seconds", generation_timeout_seconds)
        )
        entailment_eval_timeout_seconds = int(cfg.get("entailment_eval_timeout_seconds", 7200))
        job_done_timeout_seconds = generation_sync_timeout_seconds + entailment_eval_timeout_seconds
        print("[cluster_eval] Non-zero rank: waiting for download_done...", flush=True)
        if not _wait_for_marker(work_dir, ".cluster_eval_download_done", timeout_seconds=generation_sync_timeout_seconds):
            raise TimeoutError("Timed out waiting for .cluster_eval_download_done")
        print("[cluster_eval] Download done, starting Phase 2 generation (this rank)...", flush=True)
        _run_cluster_generation_this_rank(cfg, work_dir)
        print("[cluster_eval] Generation done (this rank), waiting for generation_done...", flush=True)
        if not _wait_for_marker(work_dir, ".cluster_eval_generation_done", timeout_seconds=generation_sync_timeout_seconds):
            raise TimeoutError("Timed out waiting for .cluster_eval_generation_done")
        print("[cluster_eval] Waiting for job_done...", flush=True)
        if not _wait_for_marker(work_dir, ".cluster_eval_job_done", timeout_seconds=job_done_timeout_seconds):
            raise TimeoutError("Timed out waiting for .cluster_eval_job_done")
        print("[cluster_eval] Job done, exiting.", flush=True)
        return

    # ---------- rank 0：阶段一 先下载所有权重到本地 ----------
    print("[cluster_eval] Phase 1: preparing local weights...", flush=True)
    gen_dir = os.path.join(work_dir, "gen")
    os.makedirs(gen_dir, exist_ok=True)
    manifest_csv = os.path.join(work_dir, "videophy_manifest.csv")
    eval_results_csv = os.path.join(work_dir, "videophy_eval_results.csv")
    summary_path = os.path.join(work_dir, "videophy_eval_results_summary.txt")

    checkpoint_path = (cfg.get("checkpoint_path") or cfg.get("checkpoint_path") or "").strip()
    result_path = (cfg.get("result_path") or cfg.get("result_path") or "").strip()
    input_csv = (cfg.get("input_csv") or "").strip()
    if not input_csv:
        print("[cluster_eval] input_csv is required. Use <PATH_TO_EVAL_PROMPTS_CSV>.", flush=True)
        sys.exit(1)
    entailment_ckpt = (cfg.get("entailment_checkpoint") or cfg.get("videocon_checkpoint") or "").strip()
    if not entailment_ckpt:
        print("[cluster_eval] entailment_checkpoint (or videocon_checkpoint) is required", flush=True)
        sys.exit(1)

    generate_type = (cfg.get("generate_type") or "baseline").strip().lower()
    pretrained_path = (
        cfg.get("pretrained_model_path")
        or cfg.get("pretrained_model_path")
        or ""
    ).strip() or None
    model_type = cfg.get("model_type", "lamo")

    # 1) 按 generate_type 准备生成模型：full finetune 或 LoRA
    lora_path = None
    physical_module_path = None
    if generate_type == "lora":
        if not checkpoint_path:
            print("[cluster_eval] LoRA requires checkpoint_path (e.g. .../weight_2000)", flush=True)
            sys.exit(1)
        try:
            model_path, lora_path, physical_module_path = _setup_model_lora(
                work_dir,
                checkpoint_path,
                pretrained_path,
                model_type=model_type,
                predictor_path=cfg.get("physical_module_path") or cfg.get("predictor_path"),
            )
        except Exception as e:
            print(f"[cluster_eval] LoRA setup failed: {e}", flush=True)
            sys.exit(1)
        _verify_lamo_weights(model_type, lora_path, physical_module_path, cfg)
    else:
        if checkpoint_path:
            model_path, physical_module_path = _setup_model_full_finetune(work_dir, checkpoint_path, pretrained_path)
            _verify_lamo_weights(model_type, lora_path, physical_module_path, cfg)
        else:
            model_path = cfg.get("model_path") or os.path.join(work_dir, "model")
            if not os.path.isdir(model_path):
                print(f"[cluster_eval] Full finetune needs checkpoint_path or existing model_path; not found: {model_path}", flush=True)
                sys.exit(1)

    # 2) input_csv must be local.
    input_csv = os.path.join(_REPO_ROOT, input_csv) if not os.path.isabs(input_csv) else input_csv
    if not os.path.isfile(input_csv):
        print(f"[cluster_eval] input_csv not found: {input_csv}", flush=True)
        sys.exit(1)

    # 3) 解析 entailment checkpoint（local filesystem 则下载到 cache 或 work_dir/entailment_ckpt）
    entailment_local = _resolve_entailment_checkpoint(entailment_ckpt, work_dir)
    if not entailment_local or not os.path.isdir(entailment_local):
        print(f"[cluster_eval] entailment checkpoint not found: {entailment_local}", flush=True)
        sys.exit(1)

    # 阶段一结束：写 phase2 参数供其他 rank 使用，再写 marker
    model_defaults = MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS["lamo"])
    phase2_args = {
        "model_path": model_path,
        "lora_path": lora_path,
        "input_csv": input_csv,
        "gen_dir": gen_dir,
        "work_dir": work_dir,
        "model_type": model_type,
        "generate_type": generate_type,
        "num_frames": int(cfg.get("num_frames", model_defaults["num_frames"])),
        "fps": int(cfg.get("fps", model_defaults["fps"])),
        "limit": cfg.get("limit"),
        "lora_name": cfg.get("lora_name", "lora_adapter"),
        "lora_rank": int(cfg.get("lora_rank", 128)),
        "lora_alpha": int(cfg.get("lora_alpha", 64)),
        "workers_per_rank": int(cfg.get("workers_per_rank", 1)),
        "skip_existing": _as_bool(cfg.get("skip_existing"), default=True),
    }
    if model_type == "lamo":
        phase2_args["physical_module_path"] = physical_module_path
        phase2_args["guidance_lambda"] = float(cfg.get("guidance_lambda", 15.0))
        phase2_args["guidance_step_ratio"] = float(cfg.get("guidance_step_ratio", 0.8))
        phase2_args["predictor_hidden_channels"] = int(cfg.get("predictor_hidden_channels", 256))
        phase2_args["predictor_num_res_blocks"] = int(cfg.get("predictor_num_res_blocks", 8))
        phase2_args["predictor_use_se"] = int(cfg.get("predictor_use_se", 1))
        phase2_args["predictor_use_prev_delta"] = int(cfg.get("predictor_use_prev_delta", 0))
        phase2_args["predictor_use_prompt_cond"] = int(cfg.get("predictor_use_prompt_cond", 1))
        phase2_args["prompt_text_dim"] = int(cfg.get("prompt_text_dim", 4096))
        phase2_args["prompt_cond_dim"] = int(cfg.get("prompt_cond_dim", 128))

    try:
        with open(os.path.join(work_dir, ".cluster_eval_phase2_args.json"), "w") as f:
            json.dump(phase2_args, f, indent=0)
    except Exception:
        pass
    try:
        with open(os.path.join(work_dir, ".cluster_eval_download_done"), "w") as f:
            f.write("ok\n")
    except Exception:
        pass
    print("[cluster_eval] Phase 1 done. Phase 2: multi-rank generation...", flush=True)

    # ---------- 阶段二 rank 0 也跑生成，然后等其余 rank、合并 manifest、评测（写 job_done）----------
    try:
        _run_eval_phase(cfg, work_dir, gen_dir, manifest_csv, eval_results_csv, summary_path,
                        model_path, lora_path, input_csv, entailment_local, generate_type)
    finally:
        try:
            with open(os.path.join(work_dir, ".cluster_eval_job_done"), "w") as f:
                f.write("ok\n")
        except Exception:
            pass


def _run_eval_phase(cfg, work_dir, gen_dir, manifest_csv, eval_results_csv, summary_path,
                    model_path, lora_path, input_csv, entailment_local, generate_type):
    # world_size 必须用环境变量（集群真实 rank 数），不能用 cfg 里旧的 multiprocessing world_size
    _, world_size = _get_rank_and_world_size()
    workers_per_rank = int(cfg.get("workers_per_rank", 1))
    model_type = cfg.get("model_type", "lamo")
    model_defaults = MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS["lamo"])
    num_frames = int(cfg.get("num_frames", model_defaults["num_frames"]))
    fps = int(cfg.get("fps", model_defaults["fps"]))
    limit = cfg.get("limit")
    lora_name = cfg.get("lora_name", "lora_adapter")
    lora_rank = int(cfg.get("lora_rank", 128))
    lora_alpha = int(cfg.get("lora_alpha", 64))
    result_path = (cfg.get("result_path") or "").strip()
    skip_existing = _as_bool(cfg.get("skip_existing"), default=True)
    generation_timeout_seconds = int(cfg.get("generation_timeout_seconds", 3600 * 4))
    generation_sync_timeout_seconds = int(
        cfg.get("generation_sync_timeout_seconds", generation_timeout_seconds)
    )
    entailment_eval_timeout_seconds = int(cfg.get("entailment_eval_timeout_seconds", 3600 * 2))

    # Phase 2 开始时在 wandb 打点，便于在 wandb 上看到生成进度
    try:
        n_total = len(pd.read_csv(input_csv).dropna(subset=["caption"]))
        if limit is not None:
            n_total = min(n_total, limit)
    except Exception:
        n_total = None
    _get_wandb_run(cfg)
    progress_metrics = {"videophy/generation_total": n_total, "videophy/world_size": world_size}
    _log_wandb_progress(cfg, **{k: v for k, v in progress_metrics.items() if v is not None})

    # 4) rank 0 运行 cluster 生成脚本；然后等待其余 rank 的 done marker，合并 manifest
    rank0_manifest = os.path.join(gen_dir, "manifest.rank_0.csv")
    gen_cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "tests", "videophy_cluster_generate.py"),
        "--input_csv", input_csv,
        "--output_dir", gen_dir,
        "--manifest_csv", rank0_manifest,
        "--work_dir", work_dir,
        "--model_path", model_path,
        "--model_type", model_type,
        "--generate_type", generate_type,
        "--num_frames", str(num_frames),
        "--fps", str(fps),
        "--seed", "42",
        "--rank", "0",
        "--world_size", str(world_size),
    ]
    if generate_type == "lora" and lora_path:
        gen_cmd.extend([
            "--lora_path", lora_path,
            "--lora_name", str(lora_name),
            "--lora_rank", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
        ])
    if limit is not None:
        gen_cmd.extend(["--limit", str(limit)])
    if skip_existing:
        gen_cmd.append("--skip_existing")
    if workers_per_rank > 1:
        gen_cmd.extend(["--workers_per_rank", str(workers_per_rank)])
    phase2_path = os.path.join(work_dir, ".cluster_eval_phase2_args.json")
    if model_type == "lamo" and os.path.isfile(phase2_path):
        try:
            with open(phase2_path) as f:
                phase2 = json.load(f)
            if phase2.get("physical_module_path"):
                gen_cmd.extend(["--physical_module_path", phase2["physical_module_path"]])
            gen_cmd.extend(["--guidance_lambda", str(phase2.get("guidance_lambda", 15.0))])
            gen_cmd.extend(["--guidance_step_ratio", str(phase2.get("guidance_step_ratio", 0.8))])
            gen_cmd.extend(["--predictor_hidden_channels", str(phase2.get("predictor_hidden_channels", 256))])
            gen_cmd.extend(["--predictor_num_res_blocks", str(phase2.get("predictor_num_res_blocks", 8))])
            gen_cmd.extend(["--predictor_use_se", str(phase2.get("predictor_use_se", 1))])
            gen_cmd.extend(["--predictor_use_prev_delta", str(phase2.get("predictor_use_prev_delta", 0))])
            gen_cmd.extend(["--predictor_use_prompt_cond", str(phase2.get("predictor_use_prompt_cond", 1))])
            gen_cmd.extend(["--prompt_text_dim", str(phase2.get("prompt_text_dim", 4096))])
            gen_cmd.extend(["--prompt_cond_dim", str(phase2.get("prompt_cond_dim", 128))])
        except Exception as e:
            print(f"[cluster_eval] failed to read phase2 args for rank0 gen_cmd: {e}", flush=True)

    rank0_stdout_path = os.path.join(work_dir, "rank0_generate_stdout.log")
    print(f"[cluster_eval] Generation timeout: {generation_timeout_seconds}s", flush=True)
    if skip_existing:
        print("[cluster_eval] Generation resume enabled: existing non-empty mp4 files will be skipped", flush=True)
    # Stream subprocess output to stdout in real time AND save to log file.
    # A background thread reads from the pipe so the main thread can monitor progress.
    f_log = open(rank0_stdout_path, "w")
    proc = subprocess.Popen(
        gen_cmd, cwd=_REPO_ROOT, env=os.environ,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )

    def _tee_output():
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f_log.write(line)
            f_log.flush()

    tee_thread = threading.Thread(target=_tee_output, daemon=True)
    tee_thread.start()

    ret = None
    deadline = time.monotonic() + generation_timeout_seconds

    def _all_ranks_done():
        for r in range(world_size):
            if not os.path.isfile(os.path.join(work_dir, f".cluster_eval_gen_rank_{r}_done")):
                return False
        return True

    last_progress_log = 0.0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            now = time.monotonic()
            if now - last_progress_log >= 30:
                cur = _count_generated_videos(gen_dir)
                print(f"[cluster_eval] Videos generated: {cur}/{n_total or '?'}", flush=True)
                _log_wandb_progress(cfg, **{"videophy/videos_generated_so_far": cur})
                last_progress_log = now
            time.sleep(5)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
            print(
                f"[cluster_eval] videophy_cluster_generate (rank 0) timed out "
                f"after {generation_timeout_seconds}s",
                flush=True,
            )
            sys.exit(1)
        ret = proc.returncode
    except Exception:
        proc.kill()
        proc.wait()
        raise
    finally:
        tee_thread.join(timeout=10)
        f_log.close()
    if ret != 0:
        print(f"[cluster_eval] videophy_cluster_generate failed with {ret}", flush=True)
        if os.path.isfile(rank0_stdout_path) and os.path.getsize(rank0_stdout_path) > 0:
            with open(rank0_stdout_path) as f:
                content = f.read()
            print(f"[cluster_eval] rank0_generate_stdout.log (tail):", content[-8000:], flush=True)
        sys.exit(ret)

    # 等待 rank 1..N-1 的生成完成（进度条继续更新，直到全部 rank done marker 出现）
    # rank 0 已跑完，但部分慢 rank 可能还在生成中
    all_done = _all_ranks_done()
    if not all_done:
        print("[cluster_eval] Rank 0 done, waiting for remaining ranks...", flush=True)
        deadline = time.monotonic() + generation_sync_timeout_seconds
        while time.monotonic() < deadline:
            if _all_ranks_done():
                break
            cur = _count_generated_videos(gen_dir)
            print(f"[cluster_eval] Videos generated: {cur}/{n_total or '?'}", flush=True)
            time.sleep(10)
        else:
            print("[cluster_eval] Timeout waiting for all ranks to finish generation", flush=True)
            sys.exit(1)
    print("[cluster_eval] All ranks finished generation, merging manifests...", flush=True)

    # 按 rank 顺序合并 manifest
    parts = []
    for r in range(world_size):
        p = os.path.join(gen_dir, f"manifest.rank_{r}.csv")
        if os.path.isfile(p):
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(f"No manifest.rank_*.csv under {gen_dir}")
    merged = pd.concat(parts, ignore_index=True)
    if "global_index" in merged.columns:
        merged = merged.sort_values("global_index").reset_index(drop=True)
        merged.drop(columns=["global_index"], inplace=True, errors="ignore")
    merged.to_csv(manifest_csv, index=False)
    print(f"[cluster_eval] Merged manifest written to {manifest_csv} ({len(merged)} rows)", flush=True)
    _log_wandb_progress(cfg, **{"videophy/videos_generated": len(merged)})

    try:
        with open(os.path.join(work_dir, ".cluster_eval_generation_done"), "w") as f:
            f.write("ok\n")
    except Exception:
        pass

    # 5) SA/PC 评测（仅在全部视频生成完成后执行）
    videophy_repo = (cfg.get("videophy_repo_path") or os.path.join(_REPO_ROOT, "eval", "videophy")).strip()
    if not os.path.isabs(videophy_repo):
        videophy_repo = os.path.join(_REPO_ROOT, videophy_repo)
    eval_script = os.path.join(
        videophy_repo, "videocon", "training", "pipeline_video", "entailment_eval_unified.py"
    )
    if not os.path.isfile(eval_script):
        print(f"[cluster_eval] Eval script not found: {eval_script}", flush=True)
        sys.exit(1)
    videocon_root = os.path.join(videophy_repo, "videocon")
    eval_cmd = [
        sys.executable,
        eval_script,
        "--input_csv", manifest_csv,
        "--output_csv", eval_results_csv,
        "--checkpoint", entailment_local,
        "--batch_size", "16",
    ]
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join([_REPO_ROOT, videocon_root, env.get("PYTHONPATH", "")])
    _log_wandb_progress(cfg, **{"videophy/entailment_started": 1})
    print(f"[cluster_eval] Running entailment eval (timeout={entailment_eval_timeout_seconds}s)...", flush=True)
    ret = subprocess.run(
        eval_cmd, cwd=videocon_root, env=env,
        capture_output=True, text=True, timeout=entailment_eval_timeout_seconds,
    )
    if ret.returncode != 0:
        print(f"[cluster_eval] entailment_eval_unified failed with {ret.returncode}", flush=True)
        if ret.stdout:
            print("[cluster_eval] entailment_eval_unified stdout:", ret.stdout[-8000:], flush=True)
        if ret.stderr:
            print("[cluster_eval] entailment_eval_unified stderr:", ret.stderr[-8000:], flush=True)
        sys.exit(ret.returncode)

    # entailment_eval_unified 写入的 summary 路径是 output_csv 的 base + _summary.txt
    summary_path_actual = eval_results_csv.rsplit(".", 1)[0] + "_summary.txt"
    if not os.path.isfile(summary_path_actual):
        summary_path_actual = summary_path
    if not os.path.isfile(summary_path_actual):
        print("[cluster_eval] Summary file not found after eval", flush=True)
    else:
        # 6) 上传 result_summary 到 local filesystem
        if result_path:
            _copy_summary_to_output_dir(summary_path_actual, result_path)
        # 7) Wandb 打点
        metrics = _parse_summary_file(summary_path_actual)
        _log_to_wandb(metrics, summary_path_actual, cfg)
    print("[cluster_eval] Done.", flush=True)


if __name__ == "__main__":
    main()
