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
VideoPhy-2 evaluation: 生成视频 -> VideoPhy-2 AutoEvaluator (SA + PC in one pass)
-> result_summary + wandb 打点。

与 ``cluster_eval.py`` 的差异：
  * 评测器：使用 VideoPhy-2 AutoEvaluator（MplugOwl 微调版本），通过
    ``eval/videophy2/entailment_eval_unified.py`` 跑 SA + PC 双任务；
  * 指标：SA / PC 的 1-5 整数 rating，输出 mean、ratio>=3、ratio>=4、ratio>=5，
    以及 Joint(SA>=4 AND PC>=4)；
  * 输入 CSV 约定：包含 ``caption`` 列作为原始（短）prompt；若同时含
    ``upsampled_caption`` 列（VideoPhy-2 公开数据集格式），
    则优先用 upsampled_caption 做生成，并把原 caption 存为 short_caption
    用于下游 SA 评分；
  * 目录：``eval_work_videophy2/``；marker 前缀：``.cluster_videophy2_*``。

多进程编排与 cluster_eval.py 完全一致：
  * rank 0 阶段一准备本地权重 + 写 phase2 参数；
  * rank 0..N-1 阶段二并行跑 ``videophy_cluster_generate.py`` 各自的 shard；
  * rank 0 阶段三合并 manifest -> 跑 entailment_eval_unified (VideoPhy-2 版) ->
    上传 summary + wandb 打点 -> 写 job_done；
  * 其他 rank 等 generation_done / job_done 后退出。

evaluation_fn_config 主要项（与 cluster_eval.py 保持一致，额外项见下）：
  * entailment_checkpoint（或 videocon_checkpoint / videophy2_checkpoint）：
    VideoPhy-2 AutoEvaluator 的本地 checkpoint 路径。
  * eval_num_frames: 评测器采样帧数，默认 32（VideoPhy-2 官方默认）。
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

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# -----------------------------------------------------------------------------
# Environment (同 cluster_eval.py / cluster_vbench_eval.py)
# -----------------------------------------------------------------------------
os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("FINETRAINERS_LOG_LEVEL", "DEBUG")

logging.getLogger("urllib3").setLevel(logging.WARNING)

# Prevent transformers from using broken Apex FusedLayerNorm on cluster
sys.modules["apex"] = None
sys.modules["apex.normalization"] = None
logging.info("Blocked apex.normalization to prevent FusedLayerNorm issues")

# Reuse model-setup helpers from cluster_eval.py to avoid duplication.
from eval.cluster_eval import (  # noqa: E402
    MODEL_DEFAULTS,
    _as_bool,
    _resolve_entailment_checkpoint,
    _setup_model_full_finetune,
    _setup_model_lora,
    _verify_lamo_weights,
)


def _copy_file_to_output_dir(local_path: str, output_dir: str, filename: str) -> None:
    """Copy ``local_path`` to ``{output_dir}/{filename}`` for local output callers."""
    if not output_dir or not os.path.isfile(local_path):
        return
    output_dir = output_dir.rstrip("/")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(local_path, os.path.join(output_dir, filename))


# -----------------------------------------------------------------------------
# Rank / marker helpers
# -----------------------------------------------------------------------------
def _is_main_rank() -> bool:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size <= 1 or rank == 0


def _is_local_main_rank() -> bool:
    """True if this process is local rank 0 on its node.

    On multi-process runs, local rank 0 prepares Phase-1 artifacts and other
    local ranks wait for the local marker.
    """
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


def _get_rank_and_world_size():
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, max(1, world_size)


# -----------------------------------------------------------------------------
# Local-only open-source build: cross-node local filesystem synchronization has
# been removed. Multi-node evaluation should use a shared filesystem or a
# project-specific sync plugin.
# -----------------------------------------------------------------------------
def _sanitize_for_local_path(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def _resolve_run_id(cfg: dict) -> str:
    """Return a stable run_id used to namespace this run's local filesystem sync prefix.

    CRITICAL: this MUST return the same value on every rank of the same job,
    otherwise each rank picks a different sync prefix and Phase 4 deadlocks
    waiting for files that were uploaded elsewhere.

    Priority:
      1. cfg["_training_wandb"]["run_id"]  (when train_eval forwards it)
      2. cfg["run_id"]                     (explicit override)
      3. cfg["name"]                       (yaml-level run name, sanitized)
      4. Deterministic hash of cfg         (identical across ranks because
         the cluster orchestration delivers the same cfg dict / serialized JSON
         to every rank). This replaces the earlier ``time.time()_os.getpid()``
         fallback, which was rank-divergent and broke cross-node sync.
    """
    tw = cfg.get("_training_wandb") or {}
    if tw.get("run_id"):
        return _sanitize_for_local_path(tw["run_id"])
    for key in ("run_id", "name"):
        if cfg.get(key):
            return _sanitize_for_local_path(cfg[key])
    import hashlib
    # Strip fields that may legitimately differ between ranks or that are not
    # part of the "logical" eval identity. ``_training_wandb`` is handled above.
    cfg_for_hash = {k: v for k, v in cfg.items()
                    if k not in ("_training_wandb", "run_id", "name")}
    try:
        cfg_str = json.dumps(cfg_for_hash, sort_keys=True, default=str)
    except Exception:
        cfg_str = repr(sorted(cfg_for_hash.items()))
    return "cfg_" + hashlib.md5(cfg_str.encode()).hexdigest()[:16]


def _load_config():
    raw = (os.environ.get("CLUSTER_VIDEOPHY2_EVAL_CONFIG")
           or os.environ.get("CLUSTER_EVAL_CONFIG"))
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


def _wait_for_marker(work_dir: str, marker_name: str, timeout_seconds: int = 14400,
                     poll_interval: int = 15) -> bool:
    marker = os.path.join(work_dir, marker_name)
    deadline = time.monotonic() + timeout_seconds
    last_log = 0.0
    while time.monotonic() < deadline:
        if os.path.isfile(marker):
            return True
        now = time.monotonic()
        if now - last_log >= 60:
            print(f"[cluster_videophy2_eval] (waiting for {marker_name}...) ", flush=True)
            last_log = now
        time.sleep(poll_interval)
    return False


def _count_generated_videos(gen_dir: str) -> int:
    n = 0
    for rank_dir in glob.glob(os.path.join(gen_dir, "rank_*")):
        if os.path.isdir(rank_dir):
            n += len(glob.glob(os.path.join(rank_dir, "video_*.mp4")))
    return n


# -----------------------------------------------------------------------------
# Input CSV preprocessing: VideoPhy-2 public CSV 有 caption + upsampled_caption，
# 生成阶段用 upsampled_caption（更详细的描述更适合 T2V 生成），SA 评分用原 caption
# （VideoPhy-2 的 short-caption SA 评分输入）。
# -----------------------------------------------------------------------------
def _prepare_input_csv_for_videophy2(src_csv: str, work_dir: str) -> str:
    """Normalize VideoPhy-2 input CSV to the schema expected by
    ``videophy_cluster_generate.py``:

      - ``caption`` is the text used for generation (upsampled if available).
      - ``short_caption`` carries the original short caption so the SA step
        can pass it through the manifest to entailment_eval_unified.

    If the input already has ``caption`` only (no upsampled), it is used as-is
    for both roles.  Writes to ``work_dir/videophy2_input_prepared.csv`` and
    returns that path.
    """
    df = pd.read_csv(src_csv)
    if "caption" not in df.columns:
        raise ValueError(f"VideoPhy-2 input csv must have 'caption' column: {src_csv}")

    if "upsampled_caption" in df.columns:
        # 用 upsampled_caption 作为生成 prompt，原 caption 保留为 short_caption 供 SA 评分
        short = df["caption"].astype(str)
        upsampled = df["upsampled_caption"].astype(str)
        df_out = df.copy()
        # 若数据中已有 short_caption（罕见），不覆盖
        if "short_caption" not in df_out.columns:
            df_out["short_caption"] = short
        df_out["caption"] = upsampled
        print(f"[cluster_videophy2_eval] Detected 'upsampled_caption' column: "
              f"using upsampled for generation, original caption -> short_caption for SA.",
              flush=True)
    else:
        df_out = df.copy()
        if "short_caption" not in df_out.columns:
            df_out["short_caption"] = df_out["caption"].astype(str)
        print(f"[cluster_videophy2_eval] No 'upsampled_caption' column; using "
              f"'caption' for both generation and SA evaluation.", flush=True)

    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, "videophy2_input_prepared.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[cluster_videophy2_eval] Prepared input CSV: {out_path} ({len(df_out)} rows)",
          flush=True)
    return out_path


# -----------------------------------------------------------------------------
# Summary parsing -> wandb metrics
# -----------------------------------------------------------------------------
_SA_OVERALL = re.compile(
    r"SA Mean:\s*([\d.]+),\s*>=3:\s*([\d.]+)%\s*,\s*>=4:\s*([\d.]+)%\s*,\s*>=5:\s*([\d.]+)%"
)
_PC_OVERALL = re.compile(
    r"PC Mean:\s*([\d.]+),\s*>=3:\s*([\d.]+)%\s*,\s*>=4:\s*([\d.]+)%\s*,\s*>=5:\s*([\d.]+)%"
)
_JOINT = re.compile(r"Joint\s*\(SA>=4\s*AND\s*PC>=4\):\s*([\d.]+)%")
_SOM_LINE = re.compile(
    r"^\s*(\S.*?)\s*\(n=(\d+)\):\s*SA_mean=([\d.]+)\s*SA>=4=([\d.]+)%\s*"
    r"PC_mean=([\d.]+)\s*PC>=4=([\d.]+)%\s*Joint>=4=([\d.]+)%\s*$"
)
_COMPLEXITY_LINE = re.compile(
    r"^\s*complexity=(\S+)\s*\(n=(\d+)\):\s*SA_mean=([\d.]+)\s*SA>=4=([\d.]+)%\s*"
    r"PC_mean=([\d.]+)\s*PC>=4=([\d.]+)%\s*Joint>=4=([\d.]+)%\s*$"
)


def _parse_summary_file(summary_path: str) -> dict:
    """Parse VideoPhy-2 summary text into wandb metric dict (videophy2/* keys)."""
    out = {
        "videophy2/sa_mean": None, "videophy2/sa_ge3": None,
        "videophy2/sa_ge4": None, "videophy2/sa_ge5": None,
        "videophy2/pc_mean": None, "videophy2/pc_ge3": None,
        "videophy2/pc_ge4": None, "videophy2/pc_ge5": None,
        "videophy2/joint_ge4": None,
    }
    if not os.path.isfile(summary_path):
        return out
    with open(summary_path) as f:
        text = f.read()

    m = _SA_OVERALL.search(text)
    if m:
        out["videophy2/sa_mean"] = float(m.group(1))
        out["videophy2/sa_ge3"] = float(m.group(2)) / 100.0
        out["videophy2/sa_ge4"] = float(m.group(3)) / 100.0
        out["videophy2/sa_ge5"] = float(m.group(4)) / 100.0
    m = _PC_OVERALL.search(text)
    if m:
        out["videophy2/pc_mean"] = float(m.group(1))
        out["videophy2/pc_ge3"] = float(m.group(2)) / 100.0
        out["videophy2/pc_ge4"] = float(m.group(3)) / 100.0
        out["videophy2/pc_ge5"] = float(m.group(4)) / 100.0
    m = _JOINT.search(text)
    if m:
        out["videophy2/joint_ge4"] = float(m.group(1)) / 100.0

    # By states_of_matter
    in_som = False
    in_complexity = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("=== By states_of_matter"):
            in_som, in_complexity = True, False
            continue
        if stripped.startswith("=== By complexity"):
            in_som, in_complexity = False, True
            continue
        if stripped.startswith("==="):
            in_som = in_complexity = False
            continue
        if in_som:
            m = _SOM_LINE.match(line)
            if m:
                label = m.group(1).strip()
                out[f"videophy2/by_som/{label}_sa_mean"] = float(m.group(3))
                out[f"videophy2/by_som/{label}_sa_ge4"] = float(m.group(4)) / 100.0
                out[f"videophy2/by_som/{label}_pc_mean"] = float(m.group(5))
                out[f"videophy2/by_som/{label}_pc_ge4"] = float(m.group(6)) / 100.0
                out[f"videophy2/by_som/{label}_joint_ge4"] = float(m.group(7)) / 100.0
        elif in_complexity:
            m = _COMPLEXITY_LINE.match(line)
            if m:
                c = m.group(1)
                out[f"videophy2/by_complexity/{c}_sa_mean"] = float(m.group(3))
                out[f"videophy2/by_complexity/{c}_sa_ge4"] = float(m.group(4)) / 100.0
                out[f"videophy2/by_complexity/{c}_pc_mean"] = float(m.group(5))
                out[f"videophy2/by_complexity/{c}_pc_ge4"] = float(m.group(6)) / 100.0
                out[f"videophy2/by_complexity/{c}_joint_ge4"] = float(m.group(7)) / 100.0
    return out


# -----------------------------------------------------------------------------
# Wandb helpers (mirrors cluster_eval.py / cluster_vbench_eval.py)
# -----------------------------------------------------------------------------
def _get_wandb_run(config: dict):
    if not _is_main_rank():
        return None
    try:
        import wandb
    except ImportError:
        return None
    if wandb.run is not None:
        return wandb.run
    try:
        training_wb = config.get("_training_wandb") or {}
        if training_wb.get("run_id"):
            print(f"[cluster_videophy2_eval] Resuming training wandb run: "
                  f"id={training_wb['run_id']}, project={training_wb.get('project')}",
                  flush=True)
            wandb.init(
                id=training_wb["run_id"],
                project=training_wb.get("project") or "LaMo",
                entity=training_wb.get("entity") or None,
                resume="allow",
                reinit=True,
            )
            if wandb.run is not None:
                print(f"[cluster_videophy2_eval] Resumed training wandb run: {wandb.run.url}",
                      flush=True)
                return wandb.run

        tracker = (config.get("experiment_tracker") or {}).get("wandb") or {}
        entity = tracker.get("entity") or os.environ.get("WANDB_ENTITY")
        project = tracker.get("project") or os.environ.get("WANDB_PROJECT") or "LaMo"
        variant = config.get("variant", config.get("model_type", "eval"))
        run_name = config.get("name") or f"videophy2-eval-{variant}"
        print(f"[cluster_videophy2_eval] Creating new eval wandb run: project={project}, "
              f"entity={entity}, name={run_name}", flush=True)
        wandb.init(project=project, entity=entity, name=run_name, job_type="videophy2-eval", reinit=True)
        if wandb.run is not None:
            print(f"[cluster_videophy2_eval] Created eval wandb run: {wandb.run.url}", flush=True)
        return wandb.run
    except Exception as e:
        print(f"[cluster_videophy2_eval] wandb init failed: {e}", flush=True)
        return None


def _log_wandb_progress(config: dict, **kwargs) -> None:
    if not _is_main_rank() or not kwargs:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        _get_wandb_run(config)
    if wandb.run is not None:
        try:
            wandb.log(kwargs)
        except Exception as e:
            print(f"[cluster_videophy2_eval] wandb progress log failed: {e}", flush=True)


def _log_to_wandb(metrics: dict, summary_path: str, config: dict) -> None:
    try:
        import wandb
    except ImportError:
        print("[cluster_videophy2_eval] wandb not installed, skip logging", flush=True)
        return
    if wandb.run is None:
        _get_wandb_run(config)
    if wandb.run is None:
        print("[cluster_videophy2_eval] wandb run is None, skipping wandb logging", flush=True)
        return
    try:
        to_log = {k: v for k, v in metrics.items() if v is not None}
        if to_log:
            wandb.log(to_log)
            print(f"[cluster_videophy2_eval] Logged {len(to_log)} metrics to wandb: "
                  f"{sorted(list(to_log.keys()))}", flush=True)
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary_text = f.read()
            wandb.run.summary["videophy2/summary_text"] = summary_text
            print("=" * 60, flush=True)
            print("result_summary.txt (VideoPhy-2)", flush=True)
            print("=" * 60, flush=True)
            print(summary_text, flush=True)
            print("=" * 60, flush=True)
        wandb.finish()
        print("[cluster_videophy2_eval] wandb.finish() completed successfully", flush=True)
    except Exception as e:
        print(f"[cluster_videophy2_eval] wandb logging/finish failed: {e}", flush=True)


# -----------------------------------------------------------------------------
# Generation (shared with cluster_eval: uses videophy_cluster_generate.py)
# -----------------------------------------------------------------------------
def _build_gen_cmd(phase2: dict, rank: int, world_size: int, manifest_csv: str,
                   work_dir: str) -> list:
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "tests", "videophy_cluster_generate.py"),
        "--input_csv", phase2["input_csv"],
        "--output_dir", phase2["gen_dir"],
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
        cmd.extend([
            "--lora_path", phase2["lora_path"],
            "--lora_name", str(phase2["lora_name"]),
            "--lora_rank", str(phase2["lora_rank"]),
            "--lora_alpha", str(phase2["lora_alpha"]),
        ])
    if phase2["model_type"] == "lamo":
        if phase2.get("physical_module_path"):
            cmd.extend(["--physical_module_path", phase2["physical_module_path"]])
        cmd.extend(["--guidance_lambda", str(phase2.get("guidance_lambda", 15.0))])
        cmd.extend(["--guidance_step_ratio", str(phase2.get("guidance_step_ratio", 0.8))])
        cmd.extend(["--predictor_hidden_channels", str(phase2.get("predictor_hidden_channels", 256))])
        cmd.extend(["--predictor_num_res_blocks", str(phase2.get("predictor_num_res_blocks", 8))])
        cmd.extend(["--predictor_use_se", str(phase2.get("predictor_use_se", 1))])
        cmd.extend(["--predictor_use_prev_delta", str(phase2.get("predictor_use_prev_delta", 0))])
        cmd.extend(["--predictor_use_prompt_cond", str(phase2.get("predictor_use_prompt_cond", 1))])
        cmd.extend(["--prompt_text_dim", str(phase2.get("prompt_text_dim", 4096))])
        cmd.extend(["--prompt_cond_dim", str(phase2.get("prompt_cond_dim", 128))])
    if phase2.get("limit") is not None:
        cmd.extend(["--limit", str(phase2["limit"])])
    if _as_bool(phase2.get("skip_existing"), default=True):
        cmd.append("--skip_existing")
    cmd.extend(["--done_marker_prefix", phase2.get("done_marker_prefix", ".cluster_eval_gen_rank_")])
    workers_per_rank = int(phase2.get("workers_per_rank", 1))
    if workers_per_rank > 1:
        cmd.extend(["--workers_per_rank", str(workers_per_rank)])
    return cmd


# -----------------------------------------------------------------------------
# Phase 1: local weight + checkpoint preparation
# -----------------------------------------------------------------------------
def _phase1_download(cfg: dict, work_dir: str) -> dict:
    print("[cluster_videophy2_eval] Phase 1: preparing local weights...", flush=True)
    gen_dir = os.path.join(work_dir, "gen")
    os.makedirs(gen_dir, exist_ok=True)

    checkpoint_path = (cfg.get("checkpoint_path") or cfg.get("checkpoint_path") or "").strip()
    input_csv = (cfg.get("input_csv") or "").strip()
    if not input_csv:
        raise ValueError("input_csv is required. Use <PATH_TO_VIDEOPHY2_PROMPTS_CSV>.")
    entailment_ckpt = (
        cfg.get("entailment_checkpoint")
        or cfg.get("videocon_checkpoint")
        or cfg.get("videophy2_checkpoint")
        or ""
    ).strip()
    if not entailment_ckpt:
        print("[cluster_videophy2_eval] videophy2_checkpoint is required.", flush=True)
        sys.exit(1)

    generate_type = (cfg.get("generate_type") or "baseline").strip().lower()
    pretrained_path = (cfg.get("pretrained_model_path") or cfg.get("pretrained_model_path") or "").strip() or None
    model_type = cfg.get("model_type", "lamo")
    if model_type not in ("lamo", "cogvideox"):
        raise ValueError("Open-source VideoPhy2 evaluation supports model_type='lamo' or 'cogvideox'.")
    if generate_type not in ("baseline", "lora"):
        raise ValueError("VideoPhy2 generate_type must be one of: baseline, lora")

    lora_path = None
    physical_module_path = None
    if generate_type == "lora":
        if not checkpoint_path:
            print("[cluster_videophy2_eval] LoRA evaluation requires checkpoint_path.", flush=True)
            sys.exit(1)
        model_path, lora_path, physical_module_path = _setup_model_lora(
            work_dir, checkpoint_path, pretrained_path, model_type=model_type,
        )
        _verify_lamo_weights(model_type, lora_path, physical_module_path, cfg)
    else:
        if checkpoint_path:
            model_path, physical_module_path = _setup_model_full_finetune(work_dir, checkpoint_path, pretrained_path)
            _verify_lamo_weights(model_type, lora_path, physical_module_path, cfg)
        else:
            model_path = cfg.get("model_path") or os.path.join(work_dir, "model")
            if not os.path.isdir(model_path):
                print(f"[cluster_videophy2_eval] model_path not found: {model_path}", flush=True)
                sys.exit(1)

    input_csv = os.path.join(_REPO_ROOT, input_csv) if not os.path.isabs(input_csv) else input_csv
    if not os.path.isfile(input_csv):
        print(f"[cluster_videophy2_eval] input_csv not found: {input_csv}", flush=True)
        sys.exit(1)
    prepared_csv = _prepare_input_csv_for_videophy2(input_csv, work_dir)

    entailment_local = _resolve_entailment_checkpoint(entailment_ckpt, work_dir)
    if not entailment_local or not os.path.isdir(entailment_local):
        print(f"[cluster_videophy2_eval] evaluator checkpoint not found: {entailment_local}", flush=True)
        sys.exit(1)

    model_defaults = MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS["lamo"])
    phase2_args = {
        "model_path": model_path,
        "lora_path": lora_path,
        "input_csv": prepared_csv,
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
        "done_marker_prefix": ".cluster_videophy2_gen_rank_",
        "entailment_local": entailment_local,
        "eval_num_frames": int(cfg.get("eval_num_frames", 32)),
    }
    if model_type == "lamo":
        phase2_args.update({
            "physical_module_path": physical_module_path,
            "guidance_lambda": float(cfg.get("guidance_lambda", 15.0)),
            "guidance_step_ratio": float(cfg.get("guidance_step_ratio", 0.8)),
            "predictor_hidden_channels": int(cfg.get("predictor_hidden_channels", 256)),
            "predictor_num_res_blocks": int(cfg.get("predictor_num_res_blocks", 8)),
            "predictor_use_se": int(cfg.get("predictor_use_se", 1)),
            "predictor_use_prev_delta": int(cfg.get("predictor_use_prev_delta", 0)),
            "predictor_use_prompt_cond": int(cfg.get("predictor_use_prompt_cond", 1)),
            "prompt_text_dim": int(cfg.get("prompt_text_dim", 4096)),
            "prompt_cond_dim": int(cfg.get("prompt_cond_dim", 128)),
        })

    with open(os.path.join(work_dir, ".cluster_videophy2_phase2_args.json"), "w") as f:
        json.dump(phase2_args, f, indent=0)
    with open(os.path.join(work_dir, ".cluster_videophy2_download_done"), "w") as f:
        f.write("ok\n")
    print("[cluster_videophy2_eval] Phase 1 done.", flush=True)
    return phase2_args

def _read_phase2_args(work_dir: str) -> dict:
    args_path = os.path.join(work_dir, ".cluster_videophy2_phase2_args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Phase2 args not found: {args_path}")
    with open(args_path) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Phase 2: per-rank generation
# -----------------------------------------------------------------------------
def _run_phase2_generate_this_rank(cfg: dict, work_dir: str, rank: int, world_size: int,
                                   is_global_main: bool, n_total: int) -> str:
    """Run videophy_cluster_generate for this rank's shard and return the local manifest path."""
    phase2 = _read_phase2_args(work_dir)
    gen_dir = phase2["gen_dir"]
    manifest_csv = os.path.join(gen_dir, f"manifest.rank_{rank}.csv")
    cmd = _build_gen_cmd(phase2, rank, world_size, manifest_csv, work_dir)
    workers_per_rank = int(phase2.get("workers_per_rank", 1))
    generation_timeout_seconds = int(cfg.get("generation_timeout_seconds", 3600 * 6))
    print(f"[cluster_videophy2_eval] Rank {rank}/{world_size} starting Phase 2 generation "
          f"(workers_per_rank={workers_per_rank})...", flush=True)

    if is_global_main and world_size > 1:
        # On the global main rank, tee output for richer cluster logs and emit
        # periodic per-30s progress so wandb has a running view.
        rank0_log = os.path.join(work_dir, "rank0_generate_stdout.log")
        f_log = open(rank0_log, "w")
        proc = subprocess.Popen(
            cmd, cwd=_REPO_ROOT, env=os.environ,
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

        deadline = time.monotonic() + generation_timeout_seconds
        last_progress_log = 0.0
        try:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                now = time.monotonic()
                if now - last_progress_log >= 30:
                    cur = _count_generated_videos(gen_dir)
                    print(f"[cluster_videophy2_eval] Rank 0 view: videos generated "
                          f"on this node = {cur} (global target {n_total or '?'}, "
                          f"other nodes report via local filesystem)", flush=True)
                    _log_wandb_progress(cfg, **{"videophy2/videos_generated_so_far_node0": cur})
                    last_progress_log = now
                time.sleep(5)
            if proc.poll() is None:
                proc.kill()
                proc.wait()
                print(f"[cluster_videophy2_eval] Rank 0 generate timed out "
                      f"after {generation_timeout_seconds}s", flush=True)
                sys.exit(1)
            ret_code = proc.returncode
        except Exception:
            proc.kill()
            proc.wait()
            raise
        finally:
            tee_thread.join(timeout=10)
            f_log.close()
        if ret_code != 0:
            print(f"[cluster_videophy2_eval] Rank 0 generate failed with {ret_code}", flush=True)
            if os.path.isfile(rank0_log) and os.path.getsize(rank0_log) > 0:
                with open(rank0_log) as fp:
                    content = fp.read()
                print(f"[cluster_videophy2_eval] rank0_generate_stdout.log (tail):",
                      content[-8000:], flush=True)
            sys.exit(ret_code)
    else:
        # Other ranks: simple subprocess.run, output goes through torchrun's
        # per-rank capture.
        ret = subprocess.run(cmd, cwd=_REPO_ROOT, env=os.environ).returncode
        if ret != 0:
            print(f"[cluster_videophy2_eval] Rank {rank} generate failed with {ret}", flush=True)
            sys.exit(ret)

    if not os.path.isfile(manifest_csv):
        print(f"[cluster_videophy2_eval] Rank {rank}: expected manifest not produced: "
              f"{manifest_csv}", flush=True)
        sys.exit(1)

    return manifest_csv


def _wait_for_all_rank_generation(work_dir: str, world_size: int, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_log = 0.0
    while time.monotonic() < deadline:
        missing = [
            r for r in range(world_size)
            if not os.path.isfile(os.path.join(work_dir, f".cluster_videophy2_gen_rank_{r}_done"))
        ]
        if not missing:
            return
        now = time.monotonic()
        if now - last_log >= 30:
            print(f"[cluster_videophy2_eval] Waiting for rank generation markers; "
                  f"missing ranks: {missing}", flush=True)
            last_log = now
        time.sleep(5)
    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for all rank generation markers"
    )


# -----------------------------------------------------------------------------
# Phase 3/4: rank 0 evaluator over locally merged manifests
# -----------------------------------------------------------------------------
def _phase4_single_rank(cfg: dict, work_dir: str, world_size: int,
                               manifest_csv: str, eval_results_csv: str,
                               summary_path: str) -> str:
    """Backward-compatible single-rank Phase 3+4 path: rank 0 merges manifests
    from local FS (only valid when all ranks share local FS — i.e. world_size==1
    or single-node without local filesystem sync) and runs ``entailment_eval_unified.py``
    once on the merged manifest.

    Used when ``world_size == 1`` or when no ``result_path`` is configured.
    """
    phase2 = _read_phase2_args(work_dir)
    gen_dir = phase2["gen_dir"]
    entailment_local = phase2["entailment_local"]
    num_frames_eval = int(phase2.get("eval_num_frames", cfg.get("eval_num_frames", 32)))
    entailment_eval_timeout_seconds = int(cfg.get("entailment_eval_timeout_seconds", 3600 * 6))

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
    print(f"[cluster_videophy2_eval] Legacy Phase 4: merged manifest -> {manifest_csv} "
          f"({len(merged)} rows)", flush=True)
    _log_wandb_progress(cfg, **{"videophy2/videos_generated": len(merged)})

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    eval_cwd = cfg.get("videophy2_repo_path") or os.path.join(_REPO_ROOT, "eval", "videophy2")
    if not os.path.isabs(eval_cwd):
        eval_cwd = os.path.join(_REPO_ROOT, eval_cwd)
    eval_cwd = os.path.abspath(eval_cwd)
    eval_script = os.path.join(eval_cwd, "entailment_eval_unified.py")
    eval_cmd = [
        sys.executable, eval_script,
        "--input_csv", manifest_csv,
        "--output_csv", eval_results_csv,
        "--checkpoint", entailment_local,
        "--num_frames", str(num_frames_eval),
    ]
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join([_REPO_ROOT, eval_cwd, env.get("PYTHONPATH", "")])
    _log_wandb_progress(cfg, **{"videophy2/entailment_started": 1})
    print(f"[cluster_videophy2_eval] Legacy Phase 4: running VideoPhy-2 AutoEvaluator "
          f"(num_frames={num_frames_eval}, timeout={entailment_eval_timeout_seconds}s)...",
          flush=True)
    ret = subprocess.run(
        eval_cmd, cwd=eval_cwd, env=env,
        capture_output=True, text=True, timeout=entailment_eval_timeout_seconds,
    )
    if ret.returncode != 0:
        print(f"[cluster_videophy2_eval] entailment_eval_unified failed with {ret.returncode}",
              flush=True)
        if ret.stdout:
            print("[cluster_videophy2_eval] eval stdout:", ret.stdout[-8000:], flush=True)
        if ret.stderr:
            print("[cluster_videophy2_eval] eval stderr:", ret.stderr[-8000:], flush=True)
        sys.exit(ret.returncode)
    if ret.stdout:
        print(ret.stdout[-4000:], flush=True)

    summary_path_actual = eval_results_csv.rsplit(".", 1)[0] + "_summary.txt"
    if not os.path.isfile(summary_path_actual):
        summary_path_actual = summary_path
    return summary_path_actual


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main(config=None, experiment_tracker=None):
    cfg = config if isinstance(config, dict) else _load_config()
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

    work_dir = os.path.join(_REPO_ROOT, "eval_work_videophy2")
    os.makedirs(work_dir, exist_ok=True)

    rank, world_size = _get_rank_and_world_size()
    is_global_main = (rank == 0)
    is_local_main = _is_local_main_rank()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size

    run_id = _resolve_run_id(cfg)
    generation_timeout_seconds = int(cfg.get("generation_timeout_seconds", 3600 * 6))
    generation_sync_timeout_seconds = int(
        cfg.get("generation_sync_timeout_seconds", generation_timeout_seconds)
    )
    entailment_eval_timeout_seconds = int(cfg.get("entailment_eval_timeout_seconds", 3600 * 6))
    job_done_timeout_seconds = generation_sync_timeout_seconds + entailment_eval_timeout_seconds

    # Multi-node requires a shared filesystem or custom sync plugin in the
    # open-source build.
    if multi_node:
        print(f"[cluster_videophy2_eval] FATAL: multi-node run (world_size={world_size}, "
              f"local_world_size={local_world_size}) requires a shared filesystem "
              f"or custom synchronization plugin.", flush=True)
        sys.exit(1)

    if is_global_main:
        print(f"[cluster_videophy2_eval] run_id={run_id}", flush=True)
        print(f"[cluster_videophy2_eval] world_size={world_size}, "
              f"local_world_size={local_world_size}, multi_node={multi_node}", flush=True)

    # ---------- Phase 1: each node's local rank 0 downloads ----------
    if is_local_main:
        try:
            _phase1_download(cfg, work_dir)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[cluster_videophy2_eval] Rank {rank}: Phase 1 failed: {e}", flush=True)
            sys.exit(1)
    else:
        print(f"[cluster_videophy2_eval] Rank {rank}: waiting for local download_done...",
              flush=True)
        if not _wait_for_marker(
            work_dir,
            ".cluster_videophy2_download_done",
            timeout_seconds=generation_sync_timeout_seconds,
        ):
            print(f"[cluster_videophy2_eval] Rank {rank}: timeout waiting for "
                  f"local download_done marker", flush=True)
            sys.exit(1)

    # Set up wandb on the global main rank (others stay quiet).
    if is_global_main:
        _get_wandb_run(cfg)
        try:
            input_csv_for_count = _read_phase2_args(work_dir)["input_csv"]
            n_total = len(pd.read_csv(input_csv_for_count).dropna(subset=["caption"]))
            limit = cfg.get("limit")
            if limit is not None:
                n_total = min(n_total, int(limit))
        except Exception:
            n_total = None
        _log_wandb_progress(
            cfg,
            **{k: v for k, v in {
                "videophy2/generation_total": n_total,
                "videophy2/world_size": world_size,
            }.items() if v is not None},
        )
    else:
        n_total = None

    # ---------- Phase 2: every rank generates its shard ----------
    _run_phase2_generate_this_rank(
        cfg, work_dir, rank, world_size, is_global_main, n_total,
    )

    if not is_global_main:
        print(f"[cluster_videophy2_eval] Rank {rank}: generation done, waiting for rank 0 summary.", flush=True)
        if not _wait_for_marker(
            work_dir,
            ".cluster_videophy2_job_done",
            timeout_seconds=job_done_timeout_seconds,
        ):
            print(f"[cluster_videophy2_eval] Rank {rank}: timeout waiting for job_done marker",
                  flush=True)
            sys.exit(1)
        return

    # ---------- Phase 3/4 (rank 0 only) ----------
    _wait_for_all_rank_generation(work_dir, world_size, generation_sync_timeout_seconds)
    manifest_csv = os.path.join(work_dir, "videophy2_manifest.csv")
    eval_results_csv = os.path.join(work_dir, "videophy2_eval_results.csv")
    summary_path = os.path.join(work_dir, "videophy2_eval_results_summary.txt")
    result_path = (cfg.get("result_path") or cfg.get("result_path") or "").strip()

    try:
        summary_path_actual = _phase4_single_rank(
            cfg, work_dir, world_size, manifest_csv, eval_results_csv, summary_path,
        )
        try:
            with open(os.path.join(work_dir, ".cluster_videophy2_generation_done"), "w") as f:
                f.write("ok\n")
        except Exception:
            pass

        if not summary_path_actual or not os.path.isfile(summary_path_actual):
            print("[cluster_videophy2_eval] Summary file not found after eval", flush=True)
            return

        if result_path:
            _copy_file_to_output_dir(summary_path_actual, result_path, "result_summary_videophy2.txt")
        metrics = _parse_summary_file(summary_path_actual)
        _log_to_wandb(metrics, summary_path_actual, cfg)
        print("[cluster_videophy2_eval] Done.", flush=True)
    finally:
        try:
            with open(os.path.join(work_dir, ".cluster_videophy2_job_done"), "w") as f:
                f.write("ok\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
