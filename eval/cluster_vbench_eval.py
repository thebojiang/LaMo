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
Cluster VBench evaluation: 生成视频 -> 按维度 evaluate.py -> zip + cal_final_score
-> 保存结果 + wandb 打点。

与 cluster_eval.py (VideoPhy) 的主要区别：
  - 使用 VBench 的 prompt suite (JSON)，按维度分组，而不是单一 CSV。
  - 每个 prompt 生成 ``num_samples`` 条视频（VBench 标准 = 5），文件名为
    ``{prompt}-{sample_idx}.mp4``。
  - 评测阶段，对每个维度分别跑一次
    ``torchrun --standalone --nproc_per_node=N eval/vbench/evaluate.py``。
  - 结束后把所有维度结果打包为 zip，用 ``scripts/cal_final_score.py`` 算 total score，
    并保存 result_summary。

evaluation_fn_config 主要项：
  - model_type, generate_type, pretrained_model_path, checkpoint_path,
    lora_name/rank/alpha: 与 cluster_eval.py 完全一致（复用其权重准备逻辑）
  - result_path: 本地评测 summary/zip 输出目录（previous field name）
  - dimensions: 评测维度列表（默认 16 个 VBench 标准维度）
  - num_samples: 每个 prompt 生成的视频数（默认 5，VBench 标准值）
  - prompt_info_json: VBench_full_info.json 路径（默认指向 eval/benchmark_data/vbench/）
  - vbench_root: VBench 代码根目录（默认 eval/vbench/）
  - vbench_pretrained_path: VBench 第三方评测模型（DINO/CLIP/UMT/AMT/RAFT/GRiT/
    ViCLIP/...）的本地目录（对应 ~/.cache/vbench/ 的目录布局）。默认
    user-provided VBench pretrained-weights directory. Set to
    "none" 或空串可禁用，届时 VBench 会在运行时自行 wget 下载（需能访问外网）。
  - skip_vbench_pretrained_download: true 时强制跳过本地预置模型准备，走 wget fallback。
  - detectron2 相关（仅当 dimensions 包含 GRiT 维度 object_class/multiple_objects/
    color/spatial_relationship 时需要）：
      * detectron2_wheel_path: 预构建 wheel 所在本地目录，优先从该目录安装
        ``detectron2-*.whl`` 安装（秒级），失败则 fallback 到源码编译。
      * detectron2_install_spec: 覆盖 ``pip install`` 的 spec，默认
        ``git+https://github.com/facebookresearch/detectron2.git``。可指定 fork
        或 commit hash，例如 ``git+https://github.com/.../detectron2.git@abcdef``。
      * skip_detectron2_install: true 时完全跳过安装逻辑，依赖镜像内已有的
        detectron2；import 失败则 GRiT 维度会在 evaluate.py 内报错但不阻塞其他维度。
  - num_frames/fps: 生成参数（默认与 cluster_eval 一致）
  - limit: 限制 prompt 数量（调试用，默认全部）

多机多卡编排（无共享文件系统假设）：
  - Phase 1 (每节点 leader: LOCAL_RANK==0，跨节点并行)：下载模型权重 / VBench cache /
    安装 detectron2 / 写本地 phase2_args.json + ``.cluster_vbench_download_done`` marker。
  - Phase 2 (所有 rank 并行)：每个 rank 跑 ``vbench_cluster_generate.py`` 生成
    ``tasks[global_rank::world_size]``，视频写到节点本地 ``gen/videos/``。
    Node leader 等 **本节点** 的所有 local rank 写完 ``.cluster_vbench_gen_rank_{R}_done``。
  - Phase 3 (每节点 leader 跨节点并行)：对每个维度 torchrun ``--standalone --nnodes=1
    --nproc_per_node=8 evaluate.py``，**只评本节点的视频子集**，结果写本地
    ``results/{dim}/*_eval_results.json``。
  - Phase 4/5: 开源版本不内置跨节点对象存储同步；多节点评测需要共享文件系统或项目侧同步插件。
    Global rank 0 按 VBench 自带 combiner（见 ``_VBENCH_DIM_COMBINERS``）跨节点
    重聚合，写最终 summary / zip / 上传 result_path / wandb 打点。
  - 非 leader rank：等本节点的 ``.cluster_vbench_job_done`` 后退出。
  - 单节点情况下（num_nodes==1）staging 步骤会跳过，直接 fallback 到本地 results。
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
import zipfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# -----------------------------------------------------------------------------
# Environment variables (same as cluster_eval.py; platform can override)
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

# Reuse model-setup helpers from cluster_eval.py to avoid duplicating the
# download/LoRA/verification logic.
from eval.cluster_eval import (  # noqa: E402
    _as_bool,
    _setup_model_full_finetune,
    _setup_model_lora,
    _verify_lamo_weights,
    MODEL_DEFAULTS,
)

# -----------------------------------------------------------------------------
# VBench constants
# -----------------------------------------------------------------------------
VBENCH_STANDARD_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]

_DEFAULT_VBENCH_ROOT = os.path.join(_REPO_ROOT, "eval", "vbench")
_DEFAULT_PROMPT_INFO = os.path.join(_REPO_ROOT, "eval", "benchmark_data", "vbench", "VBench_full_info.json")
_DEFAULT_CAL_SCRIPT = os.path.join(_DEFAULT_VBENCH_ROOT, "scripts", "cal_final_score.py")
_DEFAULT_EVAL_SCRIPT = os.path.join(_DEFAULT_VBENCH_ROOT, "evaluate.py")

# Local mirror of ~/.cache/vbench/ (DINO, AMT, RAFT, CLIP, UMT, ViCLIP, GRiT, ...).
# Expected layout under this prefix:
#   clip_model/ViT-B-32.pt
#   clip_model/ViT-L-14.pt
#   umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth
#   amt_model/amt-s.pth
#   raft_model/models/raft-things.pth
#   dino_model/dino_vitbase16_pretrain.pth
#   dino_model/facebookresearch_dino_main/   (git clone of facebookresearch/dino)
#   aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth
#   pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
#   grit_model/grit_b_densecap_objectdet.pth
#   caption_model/tag2text_swin_14m.pth
#   ViCLIP/ViClip-InternVid-10M-FLT.pth
_DEFAULT_VBENCH_PRETRAINED_PATH = os.environ.get("LAMO_VBENCH_PRETRAINED_PATH", "")

# GRiT-based dimensions require detectron2; skip detectron2 install if none selected.
_DETECTRON2_DIMS = {"object_class", "multiple_objects", "color", "spatial_relationship"}
_DEFAULT_DETECTRON2_SPEC = "git+https://github.com/facebookresearch/detectron2.git"

_SINGLE_NODE_MAX_GPUS = 8


# -----------------------------------------------------------------------------
# detectron2 runtime install (needed by VBench GRiT dimensions)
# -----------------------------------------------------------------------------
def _detectron2_needed(dimensions) -> bool:
    if isinstance(dimensions, str):
        dimensions = [d.strip() for d in dimensions.split(",") if d.strip()]
    return bool(_DETECTRON2_DIMS.intersection(set(dimensions or [])))


def _find_cuda_home() -> str:
    """Locate a CUDA toolkit (with nvcc) suitable for building detectron2.

    Priority:
      1. ``$CUDA_HOME`` if it contains ``bin/nvcc``.
      2. ``/usr/local/cuda-{torch.version.cuda}`` (exact match with torch's
         bundled CUDA, which matters for ABI).
      3. ``/usr/local/cuda`` and common versioned fallbacks.
      4. Parent of ``which nvcc``.
    """
    env_home = os.environ.get("CUDA_HOME")
    if env_home and os.path.isfile(os.path.join(env_home, "bin", "nvcc")):
        return env_home

    candidates = []
    try:
        import torch  # local import so main() import path stays lean
        bundled = getattr(torch.version, "cuda", None)
    except Exception:
        bundled = None
    if bundled:
        candidates.append(f"/usr/local/cuda-{bundled}")
        major = bundled.split(".")[0]
        candidates.append(f"/usr/local/cuda-{major}")
    candidates += [
        "/usr/local/cuda",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-11.8",
        "/opt/cuda",
    ]
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(os.path.join(c, "bin", "nvcc")):
            return c
    try:
        out = subprocess.check_output(["which", "nvcc"], text=True).strip()
        if out:
            return os.path.dirname(os.path.dirname(out))
    except Exception:
        pass
    return ""


def _build_env_for_detectron2() -> dict:
    """Construct a build environment for ``pip install detectron2``.

    - Set ``CUDA_HOME``/``PATH``/``LD_LIBRARY_PATH`` to the detected toolkit.
    - Force CUDA build even if torch.cuda.is_available() is False (cluster login
      nodes may lack GPUs at build time, but runtime workers have them).
    - Limit compile parallelism so low-memory workers don't OOM on nvcc.
    - Restrict ``TORCH_CUDA_ARCH_LIST`` to the typical cluster architectures.
    """
    env = {**os.environ}

    cuda_home = _find_cuda_home()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:" + env.get("LD_LIBRARY_PATH", "")
    else:
        print("[cluster_vbench_eval] WARNING: CUDA_HOME not found; detectron2 build "
              "will likely fail. Set CUDA_HOME in the cluster image.", flush=True)

    env.setdefault("FORCE_CUDA", "1")
    # Cover A100 / L40 / 4090 / H100. Override via cfg if needed.
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")
    env.setdefault("MAX_JOBS", str(min(os.cpu_count() or 4, 8)))
    # pyproject-based builds isolate a fresh env which lacks the installed torch;
    # that breaks detectron2's setup.py (it does `import torch`). PIP_NO_BUILD_
    # ISOLATION is a belt-and-suspenders safeguard alongside --no-build-isolation.
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    return env


def _try_import_detectron2() -> str:
    """Return installed detectron2 version, or empty string if not importable."""
    try:
        import importlib
        if "detectron2" in sys.modules:
            # Force reload in case we just pip-installed it in the same process
            del sys.modules["detectron2"]
            # Also drop submodules so stale state doesn't shadow the new install
            for name in list(sys.modules):
                if name.startswith("detectron2."):
                    del sys.modules[name]
        mod = importlib.import_module("detectron2")
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return ""
    except Exception as e:
        print(f"[cluster_vbench_eval] detectron2 import failed with non-ImportError: "
              f"{type(e).__name__}: {e}", flush=True)
        return ""


def _ensure_detectron2(cfg: dict, dimensions, work_dir: str) -> bool:
    """Install detectron2 from source on rank 0 if any GRiT dimension is requested.

    cfg options:
      - ``skip_detectron2_install: true``  -> do nothing, caller warned if missing.
      - ``detectron2_install_spec: "..."``  -> override the pip spec (default is
        the upstream git URL). Useful to pin a fork / commit hash.
      - ``detectron2_wheel_path: "local filesystem URI..."``  -> pre-built wheel directory on
        local path. If set and resolves to a valid
        wheel, skips the source build entirely (much faster on restricted nodes).

    Returns True iff detectron2 is importable after this call.
    """
    if not _detectron2_needed(dimensions):
        print("[cluster_vbench_eval] detectron2 not required "
              "(no GRiT dimensions selected)", flush=True)
        return False

    if cfg.get("skip_detectron2_install"):
        ver = _try_import_detectron2()
        if ver:
            print(f"[cluster_vbench_eval] skip_detectron2_install=True; using "
                  f"preinstalled detectron2 {ver}", flush=True)
            return True
        print("[cluster_vbench_eval] skip_detectron2_install=True but detectron2 "
              "not importable; GRiT dimensions will fail at eval time.", flush=True)
        return False

    ver = _try_import_detectron2()
    if ver:
        print(f"[cluster_vbench_eval] detectron2 already installed: {ver}", flush=True)
        return True

    env = _build_env_for_detectron2()
    print(f"[cluster_vbench_eval] detectron2 build env: "
          f"CUDA_HOME={env.get('CUDA_HOME','')}, "
          f"TORCH_CUDA_ARCH_LIST={env.get('TORCH_CUDA_ARCH_LIST')}, "
          f"MAX_JOBS={env.get('MAX_JOBS')}", flush=True)

    # Make sure build helpers are present (ninja speeds up compile 3-5x).
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "-q", "ninja", "wheel", "setuptools"],
            env=env, check=False,
        )
    except Exception:
        pass

    # Option A: prebuilt local wheel.
    wheel_path = (cfg.get("detectron2_wheel_path") or "").strip()
    local_wheel_path = None
    if wheel_path:
        if os.path.isdir(wheel_path):
            whls = sorted(
                (os.path.join(wheel_path, f) for f in os.listdir(wheel_path)
                 if f.endswith(".whl") and "detectron2" in f.lower()),
                key=os.path.getmtime, reverse=True,
            )
            local_wheel_path = whls[0] if whls else None
        elif os.path.isfile(wheel_path):
            local_wheel_path = wheel_path

    if local_wheel_path:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               local_wheel_path]
        print(f"[cluster_vbench_eval] Installing detectron2 from wheel...", flush=True)
    else:
        spec = (cfg.get("detectron2_install_spec") or _DEFAULT_DETECTRON2_SPEC).strip()
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "--no-build-isolation", spec]
        print(f"[cluster_vbench_eval] Source-building detectron2 from {spec} "
              f"(this usually takes 5-15 min)...", flush=True)

    t0 = time.time()
    try:
        # Stream output so the cluster log collector can see build progress live.
        log_path = os.path.join(work_dir, "detectron2_install.log")
        code = _run_and_stream(cmd, cwd=work_dir, env=env, log_path=log_path,
                               timeout=3600)
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)
    except Exception as e:
        dt = time.time() - t0
        print(f"[cluster_vbench_eval] detectron2 install FAILED after {dt:.1f}s: {e}",
              flush=True)
        print("[cluster_vbench_eval] HINT: GRiT dimensions "
              f"({sorted(_DETECTRON2_DIMS)}) will fail. Fixes:", flush=True)
        print("  1) Remove GRiT dims from `dimensions` in the YAML.", flush=True)
        print("  2) Pre-build a wheel and set `detectron2_wheel_path`.", flush=True)
        print("  3) Ensure the cluster image has CUDA toolkit + gcc>=5.4.",
              flush=True)
        return False

    dt = time.time() - t0
    print(f"[cluster_vbench_eval] detectron2 install completed in {dt:.1f}s", flush=True)

    ver = _try_import_detectron2()
    if ver:
        print(f"[cluster_vbench_eval] detectron2 version: {ver}", flush=True)
        return True
    print("[cluster_vbench_eval] WARNING: detectron2 pip install succeeded but "
          "module still not importable. Aborting GRiT-dim support.", flush=True)
    return False


def _resolve_vbench_cache_dir(work_dir: str) -> str:
    """Return the local path to use as VBENCH_CACHE_DIR.

    Prefer an existing ``$VBENCH_CACHE_DIR`` / ``~/.cache/vbench`` that the user
    may have pre-populated; otherwise place it inside ``work_dir`` so the
    cluster filesystem is writeable.
    """
    env_cache = os.environ.get("VBENCH_CACHE_DIR")
    if env_cache:
        return env_cache
    return os.path.join(work_dir, "vbench_cache")


def _prepare_vbench_pretrained(cfg: dict, work_dir: str) -> str:
    """Return a local VBench pretrained-weights cache directory."""
    cache_dir = _resolve_vbench_cache_dir(work_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if cfg.get("skip_vbench_pretrained_download"):
        print(f"[cluster_vbench_eval] skip_vbench_pretrained_download=True; "
              f"relying on runtime wget in {cache_dir}", flush=True)
        return cache_dir

    raw = cfg.get("vbench_pretrained_path")
    if raw is None:
        local_path = _DEFAULT_VBENCH_PRETRAINED_PATH
    else:
        local_path = str(raw).strip()

    if not local_path or local_path.lower() == "none":
        print(f"[cluster_vbench_eval] vbench_pretrained_path disabled; VBench "
              f"will wget missing models into {cache_dir} at runtime", flush=True)
        return cache_dir

    if os.path.isdir(local_path):
        print(f"[cluster_vbench_eval] Using local VBench pretrained dir: {local_path}",
              flush=True)
        return local_path
    print(f"[cluster_vbench_eval] vbench_pretrained_path is not a local dir: {local_path}; ignoring",
          flush=True)
    return cache_dir


# -----------------------------------------------------------------------------
# Rank helpers
# -----------------------------------------------------------------------------
def _is_main_rank() -> bool:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size <= 1 or rank == 0


def _get_rank_and_world_size():
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, max(1, world_size)


def _get_node_info():
    """Detect (node_rank, num_nodes, local_rank, local_world_size).

    Resolution order for each field:
      - node_rank:        ``NODE_RANK`` -> ``GROUP_RANK`` (torchrun) -> ``RANK // local_world_size``
      - local_world_size: ``LOCAL_WORLD_SIZE`` (torchrun) -> ``NPROC_PER_NODE`` ->
                          ``torch.cuda.device_count()`` -> 1
      - num_nodes:        ``NNODES`` -> ``WORLD_SIZE // local_world_size``
      - local_rank:       ``LOCAL_RANK`` (torchrun) -> ``RANK % local_world_size``
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))

    raw_lws = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE")
    if raw_lws:
        try:
            local_world_size = max(1, int(raw_lws))
        except ValueError:
            local_world_size = 1
    else:
        try:
            import torch
            local_world_size = max(1, torch.cuda.device_count())
        except Exception:
            local_world_size = 1
    # Cap to world_size in case env over-reports (e.g. single-node single-GPU testing)
    local_world_size = min(local_world_size, world_size)

    raw_nr = os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK")
    if raw_nr is not None and raw_nr != "":
        try:
            node_rank = int(raw_nr)
        except ValueError:
            node_rank = rank // local_world_size
    else:
        node_rank = rank // local_world_size

    raw_nnodes = os.environ.get("NNODES")
    if raw_nnodes:
        try:
            num_nodes = max(1, int(raw_nnodes))
        except ValueError:
            num_nodes = max(1, (world_size + local_world_size - 1) // local_world_size)
    else:
        num_nodes = max(1, (world_size + local_world_size - 1) // local_world_size)

    raw_lr = os.environ.get("LOCAL_RANK")
    if raw_lr is not None and raw_lr != "":
        try:
            local_rank = int(raw_lr)
        except ValueError:
            local_rank = rank % local_world_size
    else:
        local_rank = rank % local_world_size

    return node_rank, num_nodes, local_rank, local_world_size


def _is_node_leader() -> bool:
    _, _, local_rank, _ = _get_node_info()
    return local_rank == 0


def _resolve_staging_path(cfg: dict) -> str:
    """local filesystem staging has been removed from the open-source build."""
    return ""


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
            print(f"[cluster_vbench_eval] (waiting for {marker_name}...) ", flush=True)
            last_log = now
        time.sleep(poll_interval)
    return False


def _load_config():
    raw = os.environ.get("CLUSTER_VBENCH_EVAL_CONFIG") or os.environ.get("CLUSTER_EVAL_CONFIG")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


def _count_generated_videos(videos_path: str) -> int:
    if not os.path.isdir(videos_path):
        return 0
    return len(glob.glob(os.path.join(videos_path, "*.mp4")))


def _count_generated_videos_with_inflight(gen_dir: str) -> int:
    """Count finished videos under ``gen_dir/videos/`` plus in-flight ones under
    ``gen_dir/rank_{R}_tmp/``.

    During Phase 2, each rank writes to its per-rank ``rank_{R}_tmp/video_{i}.mp4``
    and only moves them to the shared ``videos/`` dir after ALL of its tasks
    finish (see ``tests/vbench_cluster_generate.py``). Counting only ``videos/``
    therefore stays at 0 for the entire run on this rank until a single big
    jump at the end. To get a meaningful live progress, also count ``video_*.mp4``
    under each ``rank_*_tmp`` directory.
    """
    if not os.path.isdir(gen_dir):
        return 0
    finished = len(glob.glob(os.path.join(gen_dir, "videos", "*.mp4")))
    inflight = len(glob.glob(os.path.join(gen_dir, "rank_*_tmp", "video_*.mp4")))
    return finished + inflight


def _count_videos_for_dimension(prompt_info_json: str, videos_path: str,
                                dimension: str, num_samples_expected: int = 5) -> int:
    """Count how many on-disk videos belong to ``dimension``.

    VBench's ``build_full_info_json`` (mode=vbench_standard) probes filenames
    of the form ``{prompt}-{i}.{ext}`` for ``i in range(5)``. We mirror that
    exact logic so the returned count matches the non-empty ``video_list``
    VBench will construct for this dimension.

    Used to cap ``nproc_per_node`` per-dimension: if a rank would get 0 videos
    after ``distribute_list_to_rank`` (``videos[rank::world]``), several
    VBench dimensions crash with ``ZeroDivisionError`` (sim/cnt with cnt=0).
    """
    try:
        with open(prompt_info_json) as f:
            entries = json.load(f)
    except Exception as e:
        print(f"[cluster_vbench_eval] Failed to read prompt info for dim count: {e}",
              flush=True)
        return 0
    if not os.path.isdir(videos_path):
        return 0
    try:
        video_names = set(os.listdir(videos_path))
    except OSError:
        return 0
    count = 0
    probe_range = max(5, int(num_samples_expected) or 5)
    for entry in entries:
        if dimension not in entry.get("dimension", []):
            continue
        prompt = entry.get("prompt_en", "")
        if not prompt:
            continue
        for i in range(probe_range):
            matched = False
            for ext in (".mp4", ".gif"):
                if f"{prompt}-{i}{ext}" in video_names:
                    count += 1
                    matched = True
                    break
            if not matched:
                continue
    return count


# Mirrors ``tests/vbench_cluster_generate._sanitize_prompt_for_filename`` so the
# eval-side prompt-to-filename mapping matches the names actually written to disk
# by the generate phase. MUST be kept in sync with that function -- otherwise
# trailing-dot/space prompts (e.g. "An astronaut flying in space.") get written
# as ``...space-0.mp4`` but VBench probes ``...space.-0.mp4`` and silently
# drops ~half the records for imaging_quality / aesthetic_quality /
# overall_consistency.
_VBENCH_FILENAME_ILLEGAL = re.compile(r"[/\\\x00]")


def _sanitize_prompt_for_filename(prompt: str) -> str:
    cleaned = _VBENCH_FILENAME_ILLEGAL.sub("_", prompt.strip())
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        cleaned = "empty_prompt"
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    return cleaned


def _build_per_node_prompt_info_json(prompt_info_json: str, videos_path: str,
                                     dimension: str, num_samples: int,
                                     dst_path: str) -> int:
    """Write a pruned VBench prompt-info JSON containing only entries whose
    videos exist on this node.

    Why: VBench's ``build_full_info_json`` (vbench_standard mode) probes for
    ``range(5)`` samples per prompt over the FULL prompt suite. With
    ``num_samples != 5`` or with multi-node sharding, this prints noisy
    "WARNING!!! This required video is not found!" lines for every
    (prompt, sample_idx) tuple that doesn't exist locally — which is most of
    them in a 16-rank/2-node run. Worse, if the node has zero videos for a
    given prompt the entry is still added with an empty ``video_list`` and
    later ``load_dimension_info`` quietly returns an empty list, which makes
    several VBench evaluators (``human_action``, ``aesthetic_quality``,
    ``imaging_quality``, ``overall_consistency``, ...) trip on
    ``acc = cor_num / cnt`` with ``cnt == 0``, killing the whole torchrun
    for that dimension and dropping the dim from the final score.

    Filtering the JSON to only entries with at least one local video matches
    exactly what evaluate.py would see anyway (after build_full_info_json),
    plus eliminates the warnings and the empty-shard crashes.

    Returns the number of (entry, sample) videos found on disk, for caller
    to use as the ``n_videos_dim`` cap on ``nproc_per_node``.
    """
    try:
        with open(prompt_info_json) as f:
            entries = json.load(f)
    except Exception as e:
        print(f"[cluster_vbench_eval] Failed to read prompt info for "
              f"per-node filter: {e}", flush=True)
        return 0
    if not os.path.isdir(videos_path):
        with open(dst_path, "w") as f:
            json.dump([], f)
        return 0
    try:
        video_names = set(os.listdir(videos_path))
    except OSError:
        with open(dst_path, "w") as f:
            json.dump([], f)
        return 0

    probe_range = max(5, int(num_samples) or 5)
    pruned = []
    n_videos = 0
    for entry in entries:
        if dimension not in entry.get("dimension", []):
            continue
        raw_prompt = entry.get("prompt_en", "")
        if not raw_prompt:
            continue
        # Generate phase writes ``{sanitize(prompt)}-{i}.mp4`` to disk (strips
        # trailing dot/space and replaces path separators). VBench's downstream
        # ``build_full_info_json`` probes ``{prompt_en}-{i}.{ext}`` against
        # ``os.listdir(videos_path)`` -- so we must (a) probe with the
        # sanitized name to see what's actually on disk, and (b) write the
        # sanitized name as ``prompt_en`` in the pruned JSON, otherwise
        # VBench will probe with the trailing-dot raw prompt and silently
        # drop those entries (~50% loss for imaging_quality/aesthetic_quality/
        # overall_consistency).
        file_prompt = _sanitize_prompt_for_filename(raw_prompt)
        present = []
        for i in range(probe_range):
            for ext in (".mp4", ".gif"):
                if f"{file_prompt}-{i}{ext}" in video_names:
                    present.append(i)
                    break
        if not present:
            continue
        n_videos += len(present)
        # Keep only the dim we're evaluating so VBench's per-dim filter is
        # a no-op (avoids accidentally pulling in other dims via the
        # entry's ``dimension`` list).
        pruned_entry = {
            "prompt_en": file_prompt,
            "dimension": [dimension],
        }
        if "auxiliary_info" in entry:
            aux = entry["auxiliary_info"]
            if isinstance(aux, dict) and dimension in aux:
                pruned_entry["auxiliary_info"] = {dimension: aux[dimension]}
            else:
                pruned_entry["auxiliary_info"] = aux
        pruned.append(pruned_entry)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(pruned, f)
    return n_videos


# -----------------------------------------------------------------------------
# Wandb helpers (mirrors cluster_eval.py)
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
            print(f"[cluster_vbench_eval] Attempting to resume training wandb run: "
                  f"id={training_wb['run_id']}, project={training_wb.get('project')}", flush=True)
            wandb.init(
                id=training_wb["run_id"],
                project=training_wb.get("project") or "LaMo",
                entity=training_wb.get("entity") or None,
                resume="allow",
                reinit=True,
            )
            if wandb.run is not None:
                print(f"[cluster_vbench_eval] Resumed training wandb run: {wandb.run.url}", flush=True)
                return wandb.run

        tracker = (config.get("experiment_tracker") or {}).get("wandb") or {}
        entity = tracker.get("entity") or os.environ.get("WANDB_ENTITY")
        project = tracker.get("project") or os.environ.get("WANDB_PROJECT") or "LaMo"
        variant = config.get("variant", config.get("model_type", "eval"))
        run_name = config.get("name") or f"vbench-eval-{variant}"
        print(f"[cluster_vbench_eval] Creating new eval wandb run: project={project}, "
              f"entity={entity}, name={run_name}", flush=True)
        wandb.init(project=project, entity=entity, name=run_name, job_type="vbench-eval", reinit=True)
        if wandb.run is not None:
            print(f"[cluster_vbench_eval] Created eval wandb run: {wandb.run.url}", flush=True)
        return wandb.run
    except Exception as e:
        print(f"[cluster_vbench_eval] wandb init failed: {e}", flush=True)
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
            print(f"[cluster_vbench_eval] wandb progress log failed: {e}", flush=True)


def _log_to_wandb(metrics: dict, summary_text: str, config: dict) -> None:
    """Log per-dim metrics + summary text to wandb (best-effort).

    NOTE: This function is wandb-only. Printing the summary to stdout (so the
    cluster log shows it) is handled by ``_print_summary_to_log``,
    which runs unconditionally even when wandb fails or isn't reachable. This
    decoupling avoids the previous bug where a failed ``_get_wandb_run`` made
    the function early-return and the summary never appeared in cluster logs.
    """
    try:
        import wandb
    except ImportError:
        print("[cluster_vbench_eval] wandb not installed, skip logging", flush=True)
        return
    if wandb.run is None:
        _get_wandb_run(config)
    if wandb.run is None:
        print("[cluster_vbench_eval] wandb run is None, skipping wandb logging", flush=True)
        return
    try:
        to_log = {k: v for k, v in metrics.items() if v is not None}
        if to_log:
            wandb.log(to_log)
            print(f"[cluster_vbench_eval] Logged {len(to_log)} metrics to wandb: "
                  f"{list(to_log.keys())}", flush=True)
        if summary_text:
            wandb.run.summary["vbench/summary_text"] = summary_text
        wandb.finish()
        print("[cluster_vbench_eval] wandb.finish() completed successfully", flush=True)
    except Exception as e:
        print(f"[cluster_vbench_eval] wandb logging/finish failed: {e}", flush=True)


def _print_summary_to_log(summary_path: str) -> None:
    """Read ``summary_path`` and print it with delimiters to stdout.

    Mirrors the pattern in ``cluster_train_eval._eval_summary_path`` flow, so
    the cluster log shows the final result block clearly,
    independent of whether wandb logging succeeded.
    """
    if not summary_path or not os.path.isfile(summary_path):
        print(f"[cluster_vbench_eval] summary file not found: {summary_path}", flush=True)
        return
    try:
        with open(summary_path) as f:
            text = f.read()
    except Exception as e:
        print(f"[cluster_vbench_eval] failed to read summary {summary_path}: {e}",
              flush=True)
        return
    print("[cluster_vbench_eval] === Evaluation result_summary (vbench) ===", flush=True)
    print(text, flush=True)
    print("[cluster_vbench_eval] === End result_summary (vbench) ===", flush=True)


# -----------------------------------------------------------------------------
# VBench post-processing helpers
# -----------------------------------------------------------------------------
def _collect_dimension_scores(results_root: str) -> dict:
    """Parse every ``*_eval_results.json`` under ``results_root`` produced by
    VBench's ``evaluate.py``.  Each file maps dimension -> [score, details...].
    """
    out = {}
    for path in glob.glob(os.path.join(results_root, "**", "*_eval_results.json"), recursive=True):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[cluster_vbench_eval] Failed to parse {path}: {e}", flush=True)
            continue
        if not isinstance(data, dict):
            continue
        for dim, payload in data.items():
            score = None
            if isinstance(payload, list) and payload:
                score = payload[0]
            elif isinstance(payload, (int, float)):
                score = payload
            elif isinstance(payload, dict):
                score = payload.get("avg") or payload.get("score")
            if score is None:
                continue
            try:
                score = float(score)
            except Exception:
                continue
            # Keep first occurrence per dimension (evaluate.py writes one file per run)
            out.setdefault(dim, score)
    return out


def _compute_vbench_total(dim_scores: dict, vbench_root: str) -> dict:
    """Use cal_final_score's normalization tables to compute quality/semantic/total."""
    scripts_dir = os.path.join(vbench_root, "scripts")
    orig_path = list(sys.path)
    try:
        sys.path.insert(0, scripts_dir)
        import importlib
        constant = importlib.import_module("constant")
        importlib.reload(constant)
    except Exception as e:
        print(f"[cluster_vbench_eval] Could not import VBench constants: {e}", flush=True)
        return {}
    finally:
        sys.path[:] = orig_path

    task_info = getattr(constant, "TASK_INFO", [])
    dim_weight = getattr(constant, "DIM_WEIGHT", {})
    normalize = getattr(constant, "NORMALIZE_DIC", {})
    quality_list = getattr(constant, "QUALITY_LIST", [])
    semantic_list = getattr(constant, "SEMANTIC_LIST", [])
    quality_w = getattr(constant, "QUALITY_WEIGHT", 4)
    semantic_w = getattr(constant, "SEMANTIC_WEIGHT", 1)

    # VBench score tables use dimension names with spaces, evaluate.py's JSON uses
    # underscore names.  Convert before lookup.
    name_map = {k: k.replace(" ", "_") for k in task_info}
    aligned = {}
    for pretty_key in task_info:
        under_key = name_map[pretty_key]
        if under_key in dim_scores:
            aligned[pretty_key] = dim_scores[under_key]

    if not aligned:
        return {"vbench/dimension_scores": dim_scores}

    normalized = {}
    for k in task_info:
        if k not in aligned:
            continue
        rng = normalize.get(k)
        if not rng:
            continue
        mn, mx = rng["Min"], rng["Max"]
        if mx == mn:
            continue
        normalized[k] = (aligned[k] - mn) / (mx - mn) * dim_weight.get(k, 1.0)

    def _avg(keys):
        vals = [normalized[k] for k in keys if k in normalized]
        total_w = sum(dim_weight.get(k, 1.0) for k in keys if k in normalized)
        if not vals or total_w == 0:
            return None
        return sum(vals) / total_w

    quality = _avg(quality_list)
    semantic = _avg(semantic_list)
    total = None
    if quality is not None and semantic is not None:
        total = (quality * quality_w + semantic * semantic_w) / (quality_w + semantic_w)
    elif quality is not None:
        total = quality
    elif semantic is not None:
        total = semantic

    metrics = {}
    for dim_under, v in dim_scores.items():
        metrics[f"vbench/dim/{dim_under}"] = v
    if quality is not None:
        metrics["vbench/quality_score"] = quality
    if semantic is not None:
        metrics["vbench/semantic_score"] = semantic
    if total is not None:
        metrics["vbench/total_score"] = total
    return metrics


def _write_summary_file(summary_path: str, dim_scores: dict, metrics: dict, config: dict) -> str:
    lines = ["VBench Evaluation Summary", "=" * 60]
    variant = config.get("variant", config.get("model_type", "?"))
    lines.append(f"Model: {variant}")
    lines.append(f"Num prompts per dim sampled: {config.get('num_samples', 5)}")
    lines.append("")
    lines.append("Per-dimension scores:")
    for dim in sorted(dim_scores.keys()):
        lines.append(f"  {dim}: {dim_scores[dim]:.4f}")
    lines.append("")
    if "vbench/quality_score" in metrics:
        lines.append(f"Quality Score:  {metrics['vbench/quality_score']:.4f}")
    if "vbench/semantic_score" in metrics:
        lines.append(f"Semantic Score: {metrics['vbench/semantic_score']:.4f}")
    if "vbench/total_score" in metrics:
        lines.append(f"Total Score:    {metrics['vbench/total_score']:.4f}")
    text = "\n".join(lines) + "\n"
    with open(summary_path, "w") as f:
        f.write(text)
    return text


def _zip_results(results_root: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(results_root):
            for fn in files:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, results_root)
                zf.write(fp, rel)
    print(f"[cluster_vbench_eval] Zipped results to {zip_path}", flush=True)


# -----------------------------------------------------------------------------
# Subprocess streaming helper
# -----------------------------------------------------------------------------
def _run_and_stream(cmd, cwd, env, log_path=None, timeout=None):
    """Run subprocess while piping stdout through the parent so cluster log
    collectors see the output.  Optional ``log_path`` tees output to a file."""
    log_f = open(log_path, "w") if log_path else None
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )

    def _tee():
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_f:
                log_f.write(line)
                log_f.flush()

    t = threading.Thread(target=_tee, daemon=True)
    t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        if log_f:
            log_f.close()
        raise
    t.join(timeout=10)
    if log_f:
        log_f.close()
    return proc.returncode


# -----------------------------------------------------------------------------
# Generation phase
# -----------------------------------------------------------------------------
def _build_gen_cmd(cfg: dict, phase2: dict, rank: int, world_size: int,
                   manifest_csv: str, work_dir: str) -> list:
    gen_script = os.path.join(_REPO_ROOT, "tests", "vbench_cluster_generate.py")
    model_type = phase2["model_type"]
    cmd = [
        sys.executable, gen_script,
        "--prompt_info_json", phase2["prompt_info_json"],
        "--dimensions", ",".join(phase2["dimensions"]),
        "--num_samples", str(phase2["num_samples"]),
        "--output_dir", phase2["gen_dir"],
        "--manifest_csv", manifest_csv,
        "--work_dir", work_dir,
        "--model_path", phase2["model_path"],
        "--model_type", model_type,
        "--generate_type", phase2["generate_type"],
        "--num_frames", str(phase2["num_frames"]),
        "--num_inference_steps", str(phase2["num_inference_steps"]),
        "--guidance_scale", str(phase2["guidance_scale"]),
        "--fps", str(phase2["fps"]),
        "--seed", str(phase2.get("seed", 42)),
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
    if phase2.get("limit") is not None:
        cmd.extend(["--limit", str(phase2["limit"])])
    if _as_bool(phase2.get("skip_existing"), default=True):
        cmd.append("--skip_existing")
    if int(phase2.get("workers_per_rank", 1)) > 1:
        cmd.extend(["--workers_per_rank", str(phase2["workers_per_rank"])])
    if model_type == "lamo":
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
    return cmd


def _phase1_setup_node(cfg: dict, work_dir: str, node_rank: int, num_nodes: int):
    os.makedirs(work_dir, exist_ok=True)
    gen_dir = os.path.abspath(cfg.get("gen_dir") or os.path.join(work_dir, "gen"))
    videos_path = os.path.join(gen_dir, "videos")
    results_root = os.path.abspath(cfg.get("results_root") or os.path.join(work_dir, "results"))
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(videos_path, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    model_type = (cfg.get("model_type") or "lamo").strip()
    if model_type not in ("lamo", "cogvideox"):
        raise ValueError("Open-source VBench evaluation supports model_type='lamo' or 'cogvideox'.")
    requested_generate_type = (cfg.get("generate_type") or "baseline").strip().lower()
    if requested_generate_type not in ("baseline", "lora", "full-finetune"):
        raise ValueError("generate_type must be one of baseline, lora, full-finetune")

    checkpoint_path = (cfg.get("checkpoint_path") or "").strip()
    pretrained_path = (cfg.get("pretrained_model_path") or "").strip() or None
    model_path = cfg.get("model_path") or pretrained_path
    lora_path = None
    physical_module_path = None

    if requested_generate_type == "lora":
        if not checkpoint_path:
            raise ValueError("LoRA VBench evaluation requires checkpoint_path.")
        model_path, lora_path, physical_module_path = _setup_model_lora(
            work_dir, checkpoint_path, pretrained_path, model_type=model_type
        )
        _verify_lamo_weights(model_type, lora_path, physical_module_path, cfg)
        generate_type_for_inference = "lora"
    elif requested_generate_type == "full-finetune":
        if not checkpoint_path:
            raise ValueError("full-finetune VBench evaluation requires checkpoint_path.")
        model_path, physical_module_path = _setup_model_full_finetune(
            work_dir, checkpoint_path, pretrained_path
        )
        _verify_lamo_weights(model_type, lora_path, physical_module_path, cfg)
        generate_type_for_inference = "baseline"
    else:
        if checkpoint_path:
            model_path, physical_module_path = _setup_model_full_finetune(
                work_dir, checkpoint_path, pretrained_path
            )
            _verify_lamo_weights(model_type, lora_path, physical_module_path, cfg)
        elif not model_path:
            default_path = MODEL_DEFAULTS.get(model_type, {}).get("pretrained_model_name_or_path")
            model_path = default_path or "<PATH_TO_PRETRAINED_MODEL>"
        generate_type_for_inference = "baseline"

    vbench_root = cfg.get("vbench_root") or _DEFAULT_VBENCH_ROOT
    if not os.path.isabs(vbench_root):
        vbench_root = os.path.join(_REPO_ROOT, vbench_root)
    vbench_root = os.path.abspath(vbench_root)
    prompt_info_json = cfg.get("prompt_info_json") or _DEFAULT_PROMPT_INFO
    if not os.path.isabs(prompt_info_json):
        prompt_info_json = os.path.join(_REPO_ROOT, prompt_info_json)
    if not os.path.isfile(prompt_info_json):
        raise FileNotFoundError(f"VBench prompt_info_json not found: {prompt_info_json}")

    dims_raw = cfg.get("dimensions") or VBENCH_STANDARD_DIMENSIONS
    if isinstance(dims_raw, str):
        dimensions = [d.strip() for d in dims_raw.split(",") if d.strip()]
    else:
        dimensions = list(dims_raw)
    if not dimensions:
        dimensions = list(VBENCH_STANDARD_DIMENSIONS)

    cache_dir = _prepare_vbench_pretrained(cfg, work_dir)
    os.environ.setdefault("VBENCH_CACHE_DIR", cache_dir)
    _ensure_detectron2(cfg, dimensions, work_dir)

    model_defaults = MODEL_DEFAULTS.get(model_type, MODEL_DEFAULTS["lamo"])
    phase2_args = {
        "model_path": model_path,
        "lora_path": lora_path,
        "prompt_info_json": prompt_info_json,
        "dimensions": dimensions,
        "num_samples": int(cfg.get("num_samples", 5)),
        "gen_dir": gen_dir,
        "videos_path": videos_path,
        "results_root": results_root,
        "model_type": model_type,
        "generate_type": generate_type_for_inference,
        "num_frames": int(cfg.get("num_frames", model_defaults["num_frames"])),
        "num_inference_steps": int(cfg.get("num_inference_steps", model_defaults.get("num_inference_steps", 50))),
        "guidance_scale": float(cfg.get("guidance_scale", model_defaults.get("guidance_scale", 6.0))),
        "fps": int(cfg.get("fps", model_defaults["fps"])),
        "seed": int(cfg.get("seed", 42)),
        "limit": cfg.get("limit"),
        "lora_name": cfg.get("lora_name", "lora_adapter"),
        "lora_rank": int(cfg.get("lora_rank", 128)),
        "lora_alpha": int(cfg.get("lora_alpha", 64)),
        "workers_per_rank": int(cfg.get("workers_per_rank", 1)),
        "skip_existing": _as_bool(cfg.get("skip_existing"), default=True),
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

    with open(os.path.join(work_dir, ".cluster_vbench_phase2_args.json"), "w") as f:
        json.dump(phase2_args, f, indent=2)
    with open(os.path.join(work_dir, ".cluster_vbench_download_done"), "w") as f:
        f.write("ok\n")
    print(f"[cluster_vbench_eval] node {node_rank}/{num_nodes} Phase 1 done. "
          f"Dimensions={dimensions}, num_samples={phase2_args['num_samples']}.", flush=True)

    return (model_path, lora_path, physical_module_path, vbench_root, prompt_info_json,
            dimensions, generate_type_for_inference, model_type, gen_dir, videos_path, results_root)


def _read_phase2_args(work_dir: str) -> dict:
    with open(os.path.join(work_dir, ".cluster_vbench_phase2_args.json")) as f:
        return json.load(f)


def _run_generation_this_rank(cfg: dict, work_dir: str) -> None:
    rank, world_size = _get_rank_and_world_size()
    phase2 = _read_phase2_args(work_dir)
    manifest_csv = os.path.join(phase2["gen_dir"], f"manifest.rank_{rank}.csv")
    cmd = _build_gen_cmd(cfg, phase2, rank, world_size, manifest_csv, work_dir)
    log_path = os.path.join(work_dir, f"vbench_generate_rank_{rank}.log")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([_REPO_ROOT, env.get("PYTHONPATH", "")])
    ret = _run_and_stream(
        cmd,
        cwd=_REPO_ROOT,
        env=env,
        log_path=log_path,
        timeout=int(cfg.get("generation_timeout_seconds", 3600 * 6)),
    )
    if ret != 0:
        raise RuntimeError(f"vbench_cluster_generate rank {rank} failed with exit code {ret}")
    with open(os.path.join(work_dir, f".cluster_vbench_gen_rank_{rank}_done"), "w") as f:
        f.write("ok\n")


def _phase2_node_leader_run_and_wait(cfg: dict, work_dir: str, gen_dir: str, videos_path: str,
                                     node_rank: int, local_world_size: int, world_size: int,
                                     n_total_videos_global_estimate=None) -> None:
    rank, _ = _get_rank_and_world_size()
    _run_generation_this_rank(cfg, work_dir)

    generation_timeout_seconds = int(cfg.get("generation_timeout_seconds", 3600 * 6))
    generation_sync_timeout_seconds = int(
        cfg.get("generation_sync_timeout_seconds", generation_timeout_seconds)
    )
    deadline = time.monotonic() + generation_sync_timeout_seconds
    last_log = 0.0
    while time.monotonic() < deadline:
        done = all(
            os.path.isfile(os.path.join(work_dir, f".cluster_vbench_gen_rank_{r}_done"))
            for r in range(world_size)
        )
        if done:
            return
        now = time.monotonic()
        if now - last_log >= 30:
            cur = _count_generated_videos_with_inflight(gen_dir)
            print(f"[cluster_vbench_eval] Videos generated: {cur}/{n_total_videos_global_estimate or '?'}", flush=True)
            _log_wandb_progress(cfg, **{"vbench/videos_generated_so_far": cur})
            last_log = now
        time.sleep(5)
    raise TimeoutError("Timed out waiting for VBench generation ranks to finish")


def _phase3_local_evaluate(cfg: dict, work_dir: str, gen_dir: str, videos_path: str,
                           results_root: str, vbench_root: str, prompt_info_json: str,
                           dimensions, node_rank: int, num_nodes: int) -> None:
    os.makedirs(results_root, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("VBENCH_CACHE_DIR", _resolve_vbench_cache_dir(work_dir))
    env["PYTHONPATH"] = os.pathsep.join([_REPO_ROOT, vbench_root, env.get("PYTHONPATH", "")])
    if cfg.get("vbench_dist_backend"):
        env["VBENCH_DIST_BACKEND"] = str(cfg["vbench_dist_backend"])
    max_gpus = int(cfg.get("vbench_eval_gpus", min(_SINGLE_NODE_MAX_GPUS, max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1"))))))
    eval_script = cfg.get("vbench_eval_script") or os.path.join(vbench_root, "evaluate.py")
    if not os.path.isfile(eval_script):
        eval_script = _DEFAULT_EVAL_SCRIPT
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"VBench evaluate.py not found: {eval_script}")

    for dim in dimensions:
        dim_prompt_json = os.path.join(work_dir, f"vbench_prompts_{dim}.json")
        n_dim_videos = _build_per_node_prompt_info_json(
            prompt_info_json, videos_path, dim, int(cfg.get("num_samples", 5)), dim_prompt_json
        )
        if n_dim_videos <= 0:
            print(f"[cluster_vbench_eval] skip dim={dim}: no generated videos found", flush=True)
            continue
        nproc = max(1, min(max_gpus, n_dim_videos))
        dim_out = os.path.join(results_root, dim)
        os.makedirs(dim_out, exist_ok=True)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node", str(nproc),
            eval_script,
            "--output_path", dim_out,
            "--full_json_dir", dim_prompt_json,
            "--videos_path", videos_path,
            "--dimension", dim,
            "--load_ckpt_from_local", "True",
            "--mode", "vbench_standard",
        ]
        print(f"[cluster_vbench_eval] evaluating VBench dim={dim} with nproc={nproc}", flush=True)
        ret = _run_and_stream(cmd, cwd=vbench_root, env=env,
                              log_path=os.path.join(work_dir, f"vbench_eval_{dim}.log"),
                              timeout=int(cfg.get("vbench_eval_timeout_seconds", 3600 * 4)))
        if ret != 0:
            raise RuntimeError(f"VBench evaluate.py failed for {dim} with exit code {ret}")


def _phase4_upload_to_staging(cfg: dict, results_root: str, work_dir: str,
                              node_rank: int, staging_prefix: str) -> None:
    with open(os.path.join(work_dir, f".cluster_vbench_node_{node_rank}_done"), "w") as f:
        f.write("ok\n")


def _phase5_master_aggregate(cfg: dict, work_dir: str, num_nodes: int, staging_prefix: str,
                             dimensions, vbench_root: str, local_results_root: str) -> None:
    dim_scores = _collect_dimension_scores(local_results_root)
    metrics = _compute_vbench_total(dim_scores, vbench_root)
    result_dir = cfg.get("result_path") or cfg.get("result_path") or os.path.join(work_dir, "summary")
    result_dir = os.path.abspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "result_summary.txt")
    summary_text = _write_summary_file(summary_path, dim_scores, metrics, cfg)
    _zip_results(local_results_root, os.path.join(result_dir, "vbench_results.zip"))
    _print_summary_to_log(summary_path)
    _log_to_wandb(metrics, summary_text, cfg)


def main(config=None, experiment_tracker=None):
    """Cluster entry point for VBench evaluation (multi-node aware).

    Architecture (no shared filesystem assumption):
      Phase 1  per-node setup:        each node's leader (LOCAL_RANK==0) downloads
                                      model + vbench cache + installs detectron2 +
                                      writes phase2_args.json + .download_done marker
                                      [parallel across all nodes].
      Phase 2  generation:            every rank generates ``tasks[global_rank::world_size]``
                                      into the LOCAL ``gen/videos/``; node leader
                                      Popen+tees one shard, waits for its node's local
                                      ranks to finish (.gen_rank_{R}_done markers).
      Phase 3  per-node evaluation:   each node leader runs torchrun evaluate.py
                                      (--standalone --nnodes=1 --nproc_per_node=8)
                                      for every dimension on its LOCAL video subset
                                      [parallel across all nodes].
      Phase 4  staging upload:        each node leader uploads its results to
                                      ``{result_path}/staging/node_{N}/`` and writes
                                      a ``node_{N}.done`` marker.
      Phase 5  master aggregation:    global rank 0 waits for all node markers, reads
                                      every node's per-dim JSONs, applies the VBench
                                      per-dim combiners (see ``_VBENCH_DIM_COMBINERS``)
                                      to the union of per-video records, then writes
                                      the global summary, zip, and wandb logs.
    """
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

    rank, world_size = _get_rank_and_world_size()
    node_rank, num_nodes, local_rank, local_world_size = _get_node_info()
    is_node_leader = (local_rank == 0)
    is_global_master = (rank == 0)
    generation_timeout_seconds = int(cfg.get("generation_timeout_seconds", 3600 * 6))
    generation_sync_timeout_seconds = int(
        cfg.get("generation_sync_timeout_seconds", generation_timeout_seconds)
    )
    vbench_eval_timeout_seconds = int(cfg.get("vbench_eval_timeout_seconds", 3600 * 4))
    dims_raw_for_timeout = cfg.get("dimensions") or VBENCH_STANDARD_DIMENSIONS
    if isinstance(dims_raw_for_timeout, str):
        n_dims_for_timeout = len([d.strip() for d in dims_raw_for_timeout.split(",") if d.strip()])
    else:
        try:
            n_dims_for_timeout = len(list(dims_raw_for_timeout))
        except TypeError:
            n_dims_for_timeout = len(VBENCH_STANDARD_DIMENSIONS)
    n_dims_for_timeout = max(1, n_dims_for_timeout or len(VBENCH_STANDARD_DIMENSIONS))
    job_done_timeout_seconds = (
        generation_sync_timeout_seconds
        + n_dims_for_timeout * vbench_eval_timeout_seconds
        + 3600
    )
    print(f"[cluster_vbench_eval] starting: rank={rank}/{world_size}, "
          f"node_rank={node_rank}/{num_nodes}, local_rank={local_rank}/{local_world_size}, "
          f"node_leader={is_node_leader}, global_master={is_global_master}", flush=True)

    work_dir = os.path.join(_REPO_ROOT, "eval_work_vbench")
    os.makedirs(work_dir, exist_ok=True)
    staging_prefix = _resolve_staging_path(cfg)
    if num_nodes > 1:
        print("[cluster_vbench_eval] FATAL: multi-node VBench evaluation requires "
              "a shared filesystem or custom synchronization plugin in the "
              "open-source build.".format(num_nodes), flush=True)
        sys.exit(1)
    if staging_prefix and is_global_master:
        print(f"[cluster_vbench_eval] master: staging_path={staging_prefix}", flush=True)

    # ===================== Non-leader (worker) path =====================
    if not is_node_leader:
        print(f"[cluster_vbench_eval] rank {rank}: worker path, "
              f"waiting for local download_done marker...", flush=True)
        if not _wait_for_marker(
            work_dir,
            ".cluster_vbench_download_done",
            timeout_seconds=generation_sync_timeout_seconds,
        ):
            print(f"[cluster_vbench_eval] rank {rank}: download_done timeout; aborting",
                  flush=True)
            sys.exit(1)
        print(f"[cluster_vbench_eval] rank {rank}: starting Phase 2 generation...", flush=True)
        _run_generation_this_rank(cfg, work_dir)
        print(f"[cluster_vbench_eval] rank {rank}: generation done; waiting for "
              f"local job_done marker (set by this node's leader after Phase 4 / 5)...",
              flush=True)
        if not _wait_for_marker(
            work_dir,
            ".cluster_vbench_job_done",
            timeout_seconds=job_done_timeout_seconds,
        ):
            print(f"[cluster_vbench_eval] rank {rank}: job_done timeout; aborting",
                  flush=True)
            sys.exit(1)
        print(f"[cluster_vbench_eval] rank {rank}: job done, exiting.", flush=True)
        return

    # ===================== Node-leader path (parallel across nodes) =====================
    try:
        # ---------- Phase 1 ----------
        (model_path, lora_path, physical_module_path, vbench_root, prompt_info_json,
         dimensions, generate_type, model_type, gen_dir, videos_path,
         results_root) = _phase1_setup_node(cfg, work_dir, node_rank, num_nodes)

        # Prompt-suite size estimate (only used for progress logging, not correctness)
        try:
            with open(prompt_info_json) as f:
                prompt_entries = json.load(f)
        except Exception:
            prompt_entries = []
        wanted = set(dimensions)
        unique_prompts = {
            e.get("prompt_en")
            for e in prompt_entries
            if e.get("prompt_en") and wanted & set(e.get("dimension", []))
        }
        n_unique_prompts = len(unique_prompts)
        limit = cfg.get("limit")
        if limit is not None:
            n_unique_prompts = min(n_unique_prompts, int(limit))
        n_total_videos = n_unique_prompts * int(cfg.get("num_samples", 5))

        if is_global_master:
            _get_wandb_run(cfg)
            _log_wandb_progress(cfg, **{
                "vbench/generation_total": n_total_videos,
                "vbench/unique_prompts": n_unique_prompts,
                "vbench/num_samples_per_prompt": int(cfg.get("num_samples", 5)),
                "vbench/world_size": world_size,
                "vbench/num_nodes": num_nodes,
                "vbench/num_dimensions": len(dimensions),
            })

        # ---------- Phase 2 ----------
        _phase2_node_leader_run_and_wait(
            cfg, work_dir, gen_dir, videos_path, node_rank, local_world_size, world_size,
            n_total_videos_global_estimate=n_total_videos,
        )
        with open(os.path.join(work_dir, ".cluster_vbench_generation_done"), "w") as f:
            f.write("ok\n")

        # ---------- Phase 3 ----------
        _phase3_local_evaluate(
            cfg, work_dir, gen_dir, videos_path, results_root, vbench_root,
            prompt_info_json, dimensions, node_rank, num_nodes,
        )

        # ---------- Phase 4 ----------
        _phase4_upload_to_staging(
            cfg, results_root, work_dir, node_rank, staging_prefix,
        )

        # ---------- Phase 5 (global master only) ----------
        if is_global_master:
            _phase5_master_aggregate(
                cfg, work_dir, num_nodes, staging_prefix, dimensions,
                vbench_root, local_results_root=results_root,
            )
    finally:
        # Always release the local non-leader workers, even on failure paths.
        try:
            with open(os.path.join(work_dir, ".cluster_vbench_job_done"), "w") as f:
                f.write("ok\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
