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
Cluster video generation for VBench evaluation: one process per rank, each rank
processes a shard of (prompt, sample_idx) tasks.

Unlike VideoPhy (which reads a CSV), VBench uses a JSON prompt suite (VBench_full_info.json):
  [{"prompt_en": "...", "dimension": ["subject_consistency", ...]}, ...]

For each prompt belonging to any of the requested dimensions, we generate
``num_samples`` videos (VBench standard expects 5 samples per prompt), named
``{prompt}-{sample_idx}.mp4`` under ``output_dir/videos/``. This is the layout
VBench's evaluate.py expects when running with ``--mode=vbench_standard``.

Sharding: tasks are assigned round-robin across ranks
(``global_task_idx % world_size == rank``).

Implementation detail: ``_generation_common.generate_video`` writes
``video_{i}.mp4`` files, so each rank first generates into
``output_dir/rank_{rank}_tmp/video_{i}.mp4`` then moves the files to the shared
``output_dir/videos/{prompt}-{sample_idx}.mp4``.

Reads RANK and WORLD_SIZE from environment if --rank/--world_size not given.
Writes ``work_dir/.cluster_vbench_gen_rank_{rank}_done`` when finished.
"""

import sys
sys.modules["apex"] = None
sys.modules["apex.normalization"] = None
print("[vbench_cluster_generate] Blocked apex.normalization to prevent FusedLayerNorm issues")

import argparse
import csv
import json
import multiprocessing
import os
import re
import shutil
from pathlib import Path

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from _generation_common import generate_video


# VBench's ``get_prompt_from_filename`` uses ``Path(path).stem`` and strips a
# trailing ``-\d+`` suffix.  So the on-disk filename must contain the prompt
# text verbatim, minus only characters that would be illegal in a filename.
# We only replace path separators and NUL; everything else goes through.
_ILLEGAL_CHARS = re.compile(r"[/\\\x00]")


def _sanitize_prompt_for_filename(prompt: str) -> str:
    """Sanitize prompt text for use in a filename while keeping VBench-compatible."""
    cleaned = _ILLEGAL_CHARS.sub("_", prompt.strip())
    # Avoid trailing dots/spaces which confuse some filesystems
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        cleaned = "empty_prompt"
    # Protect against very long names (most fs limit = 255 bytes)
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    return cleaned


def _load_vbench_tasks(prompt_info_json: str, dimensions, num_samples: int, limit=None):
    """Return (tasks, total_prompts) where tasks is a list of (prompt_en, sample_idx)."""
    with open(prompt_info_json, "r") as f:
        data = json.load(f)
    wanted = set(dimensions)
    prompts = []
    seen = set()
    for entry in data:
        prompt = entry.get("prompt_en")
        dims = set(entry.get("dimension", []))
        if not prompt or not (wanted & dims):
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    if limit is not None:
        prompts = prompts[: int(limit)]
    tasks = [(p, s) for p in prompts for s in range(num_samples)]
    return tasks, len(prompts)


def _generate_worker(device_id, task_slice, tmp_output_dir, kwargs):
    """Worker for single-GPU multi-process (same-GPU parallelism via Pool).

    Derives a distinct seed per (prompt, sample_idx) task so VBench's
    ``num_samples > 1`` protocol produces different videos per sample.
    Using ``base_seed + sample_idx`` keeps results reproducible while ensuring
    sample 0..4 of the same prompt have different initial noise.
    """
    torch.cuda.set_device(device_id)
    dtype = torch.float16 if kwargs.get("dtype_str") == "float16" else torch.bfloat16
    prompts = [p for (p, _s) in task_slice]
    base_seed = int(kwargs["seed"])
    per_prompt_seeds = [base_seed + int(s) for (_p, s) in task_slice]
    gen_kw = dict(
        model_path=kwargs["model_path"],
        prompts=prompts,
        lora_path=kwargs.get("lora_path"),
        lora_name=kwargs["lora_name"],
        lora_rank=kwargs["lora_rank"],
        lora_alpha=kwargs["lora_alpha"],
        num_frames=kwargs["num_frames"],
        output_file=tmp_output_dir,
        num_inference_steps=kwargs["num_inference_steps"],
        guidance_scale=kwargs["guidance_scale"],
        phys_guidance_scale=kwargs["phys_guidance_scale"],
        generate_type=kwargs["generate_type"],
        model_type=kwargs["model_type"],
        fps=kwargs["fps"],
        seed=base_seed,
        per_prompt_seeds=per_prompt_seeds,
        dtype=dtype,
        start_index=kwargs["start_index"],
        device=device_id,
    )
    if kwargs.get("model_type") == "lamo":
        gen_kw["physical_module_path"] = kwargs.get("physical_module_path")
        gen_kw["guidance_lambda"] = kwargs.get("guidance_lambda", 15.0)
        gen_kw["guidance_step_ratio"] = kwargs.get("guidance_step_ratio", 0.8)
        gen_kw["predictor_hidden_channels"] = kwargs.get("predictor_hidden_channels", 256)
        gen_kw["predictor_num_res_blocks"] = kwargs.get("predictor_num_res_blocks", 8)
        gen_kw["predictor_use_se"] = bool(kwargs.get("predictor_use_se", True))
        gen_kw["predictor_use_prev_delta"] = bool(kwargs.get("predictor_use_prev_delta", False))
        gen_kw["predictor_use_prompt_cond"] = bool(kwargs.get("predictor_use_prompt_cond", True))
        gen_kw["prompt_text_dim"] = kwargs.get("prompt_text_dim", 4096)
        gen_kw["prompt_cond_dim"] = kwargs.get("prompt_cond_dim", 128)
    generate_video(**gen_kw)


def _task_output_path(videos_path: str, prompt: str, sample_idx: int) -> str:
    safe = _sanitize_prompt_for_filename(prompt)
    return os.path.join(videos_path, f"{safe}-{sample_idx}.mp4")


def main():
    parser = argparse.ArgumentParser(description="Cluster-sharded VBench video generation")
    parser.add_argument("--prompt_info_json", type=str, required=True,
                        help="Path to VBench_full_info.json")
    parser.add_argument("--dimensions", type=str, required=True,
                        help="Comma-separated VBench dimensions to cover "
                             "(e.g. subject_consistency,object_class,...)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of video samples per prompt (VBench standard=5)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root; final videos go under output_dir/videos/ "
                             "and per-rank tmp dirs under output_dir/rank_{R}_tmp/")
    parser.add_argument("--manifest_csv", type=str, required=True,
                        help="Per-rank manifest CSV listing generated (prompt, path, sample_idx)")
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Where to write the .cluster_vbench_gen_rank_N_done marker")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="lamo")
    parser.add_argument("--generate_type", type=str, default="baseline")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_name", type=str, default="lora_adapter")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--phys_guidance_scale", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--physical_module_path", type=str, default=None)
    parser.add_argument("--guidance_lambda", type=float, default=15.0)
    parser.add_argument("--guidance_step_ratio", type=float, default=0.8)
    parser.add_argument("--predictor_hidden_channels", type=int, default=256)
    parser.add_argument("--predictor_num_res_blocks", type=int, default=8)
    parser.add_argument("--predictor_use_se", type=int, default=1)
    parser.add_argument("--predictor_use_prev_delta", type=int, default=0)
    parser.add_argument("--predictor_use_prompt_cond", type=int, default=1,
                        help="Enable prompt FiLM conditioning (for lamo)")
    parser.add_argument("--prompt_text_dim", type=int, default=4096,
                        help="Pooled prompt vector dim (for lamo)")
    parser.add_argument("--prompt_cond_dim", type=int, default=128,
                        help="FiLM bottleneck dim D_cond (for lamo)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on number of unique prompts to generate (for debug)")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--workers_per_rank", type=int, default=1,
                        help="Parallel workers per rank sharing the same GPU")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip existing non-empty final videos instead of regenerating them")
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else int(
        os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    )
    world_size = args.world_size if args.world_size is not None else int(
        os.environ.get("WORLD_SIZE", "1")
    )
    world_size = max(1, world_size)

    dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
    all_tasks, n_prompts = _load_vbench_tasks(
        args.prompt_info_json, dimensions, args.num_samples, args.limit
    )
    my_task_indices = list(range(rank, len(all_tasks), world_size))
    my_tasks = [all_tasks[i] for i in my_task_indices]

    videos_path = os.path.join(args.output_dir, "videos")
    os.makedirs(videos_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_csv) or ".", exist_ok=True)

    generate_tasks = []
    skipped_existing = 0
    if args.skip_existing:
        for task in my_tasks:
            prompt, sample_idx = task
            dst = _task_output_path(videos_path, prompt, sample_idx)
            if os.path.isfile(dst) and os.path.getsize(dst) > 0:
                skipped_existing += 1
            else:
                generate_tasks.append(task)
    else:
        generate_tasks = list(my_tasks)

    if len(my_tasks) == 0:
        print(f"[vbench_cluster_generate] rank {rank}/{world_size}: no tasks "
              f"(total_prompts={n_prompts}, num_samples={args.num_samples}), "
              f"writing empty manifest and done marker.", flush=True)
        with open(args.manifest_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "videopath", "sample_idx"])
        done = os.path.join(args.work_dir, f".cluster_vbench_gen_rank_{rank}_done")
        with open(done, "w") as f:
            f.write("ok\n")
        return

    tmp_dir = os.path.join(args.output_dir, f"rank_{rank}_tmp")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    n_workers = min(args.workers_per_rank, max(1, len(generate_tasks)))
    if len(generate_tasks) > 0:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs visible.")
        device_id = int(os.environ.get("LOCAL_RANK", rank % num_gpus))
        device_id = device_id % num_gpus
        device_desc = f"GPU {device_id}"
    else:
        device_id = -1
        device_desc = "no GPU needed"

    print(f"[vbench_cluster_generate] rank {rank}/{world_size}: {n_prompts} prompts x "
          f"{args.num_samples} samples = {len(all_tasks)} tasks total; "
          f"this rank owns {len(my_tasks)} videos, generates {len(generate_tasks)} "
          f"and skips {skipped_existing} existing videos with {n_workers} worker(s) "
          f"on {device_desc} -> {tmp_dir}", flush=True)

    base_kwargs = {
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "lora_name": args.lora_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "phys_guidance_scale": args.phys_guidance_scale,
        "generate_type": args.generate_type,
        "model_type": args.model_type,
        "fps": args.fps,
        "seed": args.seed,
        "dtype_str": args.dtype,
        "physical_module_path": args.physical_module_path,
        "guidance_lambda": args.guidance_lambda,
        "guidance_step_ratio": args.guidance_step_ratio,
        "predictor_hidden_channels": args.predictor_hidden_channels,
        "predictor_num_res_blocks": args.predictor_num_res_blocks,
        "predictor_use_se": bool(args.predictor_use_se),
        "predictor_use_prev_delta": bool(args.predictor_use_prev_delta),
        "predictor_use_prompt_cond": bool(args.predictor_use_prompt_cond),
        "prompt_text_dim": args.prompt_text_dim,
        "prompt_cond_dim": args.prompt_cond_dim,
    }

    if len(generate_tasks) == 0:
        print(f"[vbench_cluster_generate] rank {rank}: all owned videos already exist; "
              f"writing manifest only.", flush=True)
    elif n_workers <= 1:
        kw = dict(base_kwargs)
        kw["start_index"] = 0
        _generate_worker(device_id, generate_tasks, tmp_dir, kw)
    else:
        n = len(generate_tasks)
        chunk_size = (n + n_workers - 1) // n_workers
        chunks = []
        for w in range(n_workers):
            start = w * chunk_size
            end = min(start + chunk_size, n)
            if start >= end:
                continue
            kw_w = dict(base_kwargs)
            kw_w["start_index"] = start
            chunks.append((device_id, generate_tasks[start:end], tmp_dir, kw_w))
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            pool.starmap(_generate_worker, chunks)

    # Move each generated video_{i}.mp4 to the shared videos_path with the
    # VBench-expected filename ``{prompt}-{sample_idx}.mp4``.
    generated_index_by_task = {task: i for i, task in enumerate(generate_tasks)}
    missing = 0
    for prompt, sample_idx in generate_tasks:
        src_i = generated_index_by_task[(prompt, sample_idx)]
        src = os.path.join(tmp_dir, f"video_{src_i}.mp4")
        if not os.path.isfile(src):
            print(f"[vbench_cluster_generate] WARNING: missing {src} (prompt={prompt!r}, "
                  f"sample_idx={sample_idx})", flush=True)
            missing += 1
            continue
        dst = _task_output_path(videos_path, prompt, sample_idx)
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.move(src, dst)

    manifest_rows = []
    for prompt, sample_idx in my_tasks:
        dst = _task_output_path(videos_path, prompt, sample_idx)
        if not os.path.isfile(dst):
            continue
        manifest_rows.append((prompt, dst, sample_idx))

    with open(args.manifest_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "videopath", "sample_idx"])
        writer.writerows(manifest_rows)
    n_manifest = len(manifest_rows)
    print(f"[vbench_cluster_generate] rank {rank} manifest written to {args.manifest_csv} "
          f"({n_manifest}/{len(my_tasks)} videos, missing_generated={missing})", flush=True)

    # Clean up empty tmp dir
    try:
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except OSError:
        pass

    done_marker = os.path.join(args.work_dir, f".cluster_vbench_gen_rank_{rank}_done")
    with open(done_marker, "w") as f:
        f.write("ok\n")
    print(f"[vbench_cluster_generate] rank {rank} done.", flush=True)


if __name__ == "__main__":
    main()
