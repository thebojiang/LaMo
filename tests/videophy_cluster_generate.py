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
Cluster video generation for VideoPhy evaluation: one process per rank, each rank
processes rows with index % world_size == rank.

Supports --workers_per_rank N to run multiple workers on the same GPU within a single
rank (multiprocessing.Pool, same as videophy_batch_generate.py's per-rank parallelism).
Total parallelism = num_gpus(ranks) * workers_per_rank.

Reads RANK and WORLD_SIZE from environment if --rank/--world_size not given.
Writes work_dir/{done_marker_prefix}{rank}_done when finished.
Each rank writes videos to output_dir/rank_{rank}/ to avoid filename collision.
"""

# -----------------------------------------------------------------------------
# Fix: Prevent transformers from using broken Apex FusedLayerNorm on cluster
# The cluster has Apex installed but CUDA extension not compiled, causing:
# "No module named 'fused_layer_norm_cuda'"
# Solution 1: Block apex.normalization import BEFORE transformers loads
# Solution 2: Monkey-patch T5LayerNorm after import
# MUST run before any transformers import
# -----------------------------------------------------------------------------
import sys

# Block apex.normalization from being imported - this prevents transformers
# from detecting apex's FusedRMSNorm and falling back to standard PyTorch
sys.modules["apex"] = None
sys.modules["apex.normalization"] = None
print("[videophy_cluster_generate] Blocked apex.normalization to prevent FusedLayerNorm issues")

import multiprocessing
import os
import argparse
import pandas as pd
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from _generation_common import generate_video


def _generate_worker(device_id, chunk_df, start_index, output_dir, manifest_columns, kwargs):
    """Worker for single-GPU multi-process: each worker loads model independently."""
    torch.cuda.set_device(device_id)
    dtype = torch.float16 if kwargs.get("dtype_str") == "float16" else torch.bfloat16
    prompts = chunk_df["caption"].astype(str).tolist()
    gen_kw = dict(
        model_path=kwargs["model_path"],
        prompts=prompts,
        lora_path=kwargs.get("lora_path"),
        lora_name=kwargs["lora_name"],
        lora_rank=kwargs["lora_rank"],
        lora_alpha=kwargs["lora_alpha"],
        num_frames=kwargs["num_frames"],
        output_file=output_dir,
        num_inference_steps=kwargs["num_inference_steps"],
        guidance_scale=kwargs["guidance_scale"],
        phys_guidance_scale=kwargs["phys_guidance_scale"],
        generate_type=kwargs["generate_type"],
        model_type=kwargs["model_type"],
        fps=kwargs["fps"],
        seed=kwargs["seed"],
        dtype=dtype,
        start_index=start_index,
        device=device_id,
        skip_existing=kwargs.get("skip_existing", False),
    )
    if kwargs.get("model_type") == "lamo":
        gen_kw["physical_module_path"] = kwargs.get("physical_module_path")
        gen_kw["guidance_lambda"] = kwargs.get("guidance_lambda", 15.0)
        gen_kw["guidance_step_ratio"] = kwargs.get("guidance_step_ratio", 0.8)
        gen_kw["predictor_hidden_channels"] = kwargs.get("predictor_hidden_channels", 256)
        gen_kw["predictor_num_res_blocks"] = kwargs.get("predictor_num_res_blocks", 8)
        gen_kw["predictor_use_se"] = kwargs.get("predictor_use_se", True)
        gen_kw["predictor_use_prev_delta"] = kwargs.get("predictor_use_prev_delta", False)
        gen_kw["predictor_use_prompt_cond"] = kwargs.get("predictor_use_prompt_cond", True)
        gen_kw["prompt_text_dim"] = kwargs.get("prompt_text_dim", 4096)
        gen_kw["prompt_cond_dim"] = kwargs.get("prompt_cond_dim", 128)
    generate_video(**gen_kw)
    manifest_rows = []
    for i in range(len(chunk_df)):
        r = chunk_df.iloc[i]
        videopath = os.path.join(output_dir, f"video_{start_index + i}.mp4")
        row = {"videopath": videopath, "caption": r["caption"]}
        for col in manifest_columns:
            if col in chunk_df.columns:
                row[col] = r[col]
        manifest_rows.append(row)
    return manifest_rows


def _write_done_marker(work_dir: str, prefix: str, rank: int) -> None:
    done_marker = os.path.join(work_dir, f"{prefix}{rank}_done")
    with open(done_marker, "w") as f:
        f.write("ok\n")


def main():
    parser = argparse.ArgumentParser(description="Cluster-sharded video generation (one rank per process)")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output dir; this rank writes to output_dir/rank_N/")
    parser.add_argument("--manifest_csv", type=str, required=True,
                        help="Output CSV for this rank (e.g. manifest.rank_N.csv)")
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Work dir for writing rank completion marker")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="lamo")
    parser.add_argument("--generate_type", type=str, default="baseline")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_name", type=str, default="lora_adapter")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--phys_guidance_scale", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--physical_module_path", type=str, default=None,
                        help="Path to predictor.safetensors (for lamo)")
    parser.add_argument("--guidance_lambda", type=float, default=15.0,
                        help="Inference guidance strength (for lamo)")
    parser.add_argument("--guidance_step_ratio", type=float, default=0.8,
                        help="Fraction of denoising steps with guidance (for lamo)")
    parser.add_argument("--predictor_hidden_channels", type=int, default=256,
                        help="Predictor hidden channels (for lamo)")
    parser.add_argument("--predictor_num_res_blocks", type=int, default=8,
                        help="Predictor number of ResBlocks (for lamo)")
    parser.add_argument("--predictor_use_se", type=int, default=1,
                        help="Enable SE blocks in predictor (for lamo)")
    parser.add_argument("--predictor_use_prev_delta", type=int, default=0,
                        help="Concat previous latent delta as extra input (for lamo)")
    parser.add_argument("--predictor_use_prompt_cond", type=int, default=1,
                        help="Enable prompt FiLM conditioning in the predictor (for lamo)")
    parser.add_argument("--prompt_text_dim", type=int, default=4096,
                        help="Pooled prompt vector dim (for lamo; 4096 = T5-XXL / CogVideoX-5b)")
    parser.add_argument("--prompt_cond_dim", type=int, default=128,
                        help="FiLM bottleneck dimension D_cond (for lamo)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None,
                        help="Cluster rank (default: env RANK)")
    parser.add_argument("--world_size", type=int, default=None,
                        help="Cluster world size (default: env WORLD_SIZE)")
    parser.add_argument("--workers_per_rank", type=int, default=1,
                        help="Number of parallel workers within this rank (share the same GPU)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip existing non-empty video_*.mp4 files instead of regenerating them")
    parser.add_argument("--done_marker_prefix", type=str, default=".cluster_eval_gen_rank_",
                        help="Rank done marker prefix under work_dir")
    args = parser.parse_args()

    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    world_size = max(1, world_size)

    df = pd.read_csv(args.input_csv)
    if "caption" not in df.columns:
        raise ValueError("input_csv must have 'caption' column")
    df = df.dropna(subset=["caption"])
    if args.limit is not None:
        df = df.head(args.limit)

    # Shard: this rank only processes rows with index % world_size == rank
    global_indices = list(range(rank, len(df), world_size))
    df_shard = df.iloc[global_indices].reset_index(drop=True)
    if len(df_shard) == 0:
        print(f"[videophy_cluster_generate] rank {rank} has no rows, writing empty manifest and done marker.", flush=True)
        os.makedirs(os.path.dirname(args.manifest_csv) or ".", exist_ok=True)
        pd.DataFrame(columns=["videopath", "caption"]).to_csv(args.manifest_csv, index=False)
        _write_done_marker(args.work_dir, args.done_marker_prefix, rank)
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs visible.")
    device_id = int(os.environ.get("LOCAL_RANK", rank % num_gpus))
    device_id = device_id % num_gpus

    rank_output_dir = os.path.join(args.output_dir, f"rank_{rank}")
    os.makedirs(rank_output_dir, exist_ok=True)
    manifest_columns = ["short_caption", "states_of_matter", "complexity", "source", "majority_sa", "majority_pc"]
    n_workers = min(args.workers_per_rank, len(df_shard))

    if n_workers <= 1:
        # Single worker: call generate_video directly
        print(f"[videophy_cluster_generate] rank {rank}/{world_size} processing {len(df_shard)} rows "
              f"(1 worker) on GPU {device_id} -> {rank_output_dir}", flush=True)
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        prompts = df_shard["caption"].astype(str).tolist()
        gen_kw = dict(
            model_path=args.model_path,
            prompts=prompts,
            lora_path=args.lora_path,
            lora_name=args.lora_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            num_frames=args.num_frames,
            output_file=rank_output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            phys_guidance_scale=args.phys_guidance_scale,
            generate_type=args.generate_type,
            model_type=args.model_type,
            fps=args.fps,
            seed=args.seed,
            dtype=dtype,
            start_index=0,
            device=device_id,
            skip_existing=args.skip_existing,
        )
        if args.model_type == "lamo":
            gen_kw["physical_module_path"] = args.physical_module_path
            gen_kw["guidance_lambda"] = args.guidance_lambda
            gen_kw["guidance_step_ratio"] = args.guidance_step_ratio
            gen_kw["predictor_hidden_channels"] = args.predictor_hidden_channels
            gen_kw["predictor_num_res_blocks"] = args.predictor_num_res_blocks
            gen_kw["predictor_use_se"] = bool(args.predictor_use_se)
            gen_kw["predictor_use_prev_delta"] = bool(args.predictor_use_prev_delta)
            gen_kw["predictor_use_prompt_cond"] = bool(args.predictor_use_prompt_cond)
            gen_kw["prompt_text_dim"] = args.prompt_text_dim
            gen_kw["prompt_cond_dim"] = args.prompt_cond_dim
        generate_video(**gen_kw)
        manifest_rows = []
        for i in range(len(df_shard)):
            r = df_shard.iloc[i]
            videopath = os.path.join(rank_output_dir, f"video_{i}.mp4")
            row = {"global_index": global_indices[i], "videopath": videopath, "caption": r["caption"]}
            for col in manifest_columns:
                if col in df_shard.columns:
                    row[col] = r[col]
            manifest_rows.append(row)
    else:
        # Multiple workers on the same GPU (same as videophy_batch_generate.py Pool logic)
        print(f"[videophy_cluster_generate] rank {rank}/{world_size} processing {len(df_shard)} rows "
              f"({n_workers} workers) on GPU {device_id} -> {rank_output_dir}", flush=True)
        n = len(df_shard)
        chunk_size = (n + n_workers - 1) // n_workers
        chunks = []
        for w in range(n_workers):
            start = w * chunk_size
            end = min(start + chunk_size, n)
            if start >= end:
                continue
            chunks.append((device_id, df_shard.iloc[start:end], start))
        kwargs = {
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
            "skip_existing": args.skip_existing,
        }
        if args.model_type == "lamo":
            kwargs["physical_module_path"] = args.physical_module_path
            kwargs["guidance_lambda"] = args.guidance_lambda
            kwargs["guidance_step_ratio"] = args.guidance_step_ratio
            kwargs["predictor_hidden_channels"] = args.predictor_hidden_channels
            kwargs["predictor_num_res_blocks"] = args.predictor_num_res_blocks
            kwargs["predictor_use_se"] = bool(args.predictor_use_se)
            kwargs["predictor_use_prev_delta"] = bool(args.predictor_use_prev_delta)
            kwargs["predictor_use_prompt_cond"] = bool(args.predictor_use_prompt_cond)
            kwargs["prompt_text_dim"] = args.prompt_text_dim
            kwargs["prompt_cond_dim"] = args.prompt_cond_dim
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            results = pool.starmap(
                _generate_worker,
                [
                    (dev, chunk_df, start_idx, rank_output_dir, manifest_columns, kwargs)
                    for dev, chunk_df, start_idx in chunks
                ],
            )
        manifest_rows = []
        flat = [row for res in results for row in res]
        for local_i, row in enumerate(flat):
            row["global_index"] = global_indices[local_i] if local_i < len(global_indices) else -1
            manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    os.makedirs(os.path.dirname(args.manifest_csv) or ".", exist_ok=True)
    manifest_df.to_csv(args.manifest_csv, index=False)
    print(f"[videophy_cluster_generate] rank {rank} manifest written to {args.manifest_csv} ({len(manifest_df)} rows)", flush=True)

    _write_done_marker(args.work_dir, args.done_marker_prefix, rank)
    print(f"[videophy_cluster_generate] rank {rank} done.", flush=True)


if __name__ == "__main__":
    main()
