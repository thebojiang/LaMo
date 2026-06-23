# Getting Started

This document contains the full setup, training, inference, and benchmark
evaluation notes. The main README keeps only the short-start commands.

## Installation

```bash
git clone <LAMO_REPO_URL>
cd <PATH_TO_LAMO_REPO>

conda create -n lamo python=3.10 -y
conda activate lamo

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

VBench uses additional metric dependencies. Install them only when running
VBench:

```bash
pip install -r eval/configs/requirements_vbench.txt
```

## Assets

Benchmark prompt metadata and evaluator code are included in this repository:

- `eval/benchmark_data/videophy/`: VideoPhy CSV metadata.
- `eval/benchmark_data/videophy2/`: VideoPhy2 CSV metadata.
- `eval/benchmark_data/vbench/`: VBench prompt-info JSON files.
- `eval/videophy/`: VideoPhy evaluator code.
- `eval/videophy2/`: VideoPhy2 evaluator code.
- `eval/vbench/`: VBench evaluator code, prompts, and third-party metric code.

Model and evaluator weights are not included. Download them into
`pretrain_models/` or set the corresponding launcher environment variable to
your local path.

Required paths:

- `pretrained_model_name_or_path`: CogVideoX base model path or public model id
  used for training.
- `pretrained_model_path`: local CogVideoX base model directory used for
  evaluation and inference.
- `dataset_config`: training dataset config JSON.
- `checkpoint_path`: LaMo output/checkpoint directory.
- `input_csv`: evaluation prompt CSV. Defaults point to
  `eval/benchmark_data/...`.
- `videocon_checkpoint`: VideoPhy evaluator checkpoint, default
  `pretrain_models/videocon_physics`.
- `videophy2_checkpoint`: VideoPhy2 evaluator checkpoint, default
  `pretrain_models/videophy_2_auto`.
- `vbench_pretrained_path`: VBench metric model cache directory, default
  `pretrain_models/vbench_pretrained`.

Default evaluation weight layout:

```text
pretrain_models/
  CogVideoX-2b-Diffusers/
  LaMo-CogVideoX-2b/
    transformer/diffusion_pytorch_model.safetensors
    scheduler/
    predictor.safetensors
  videocon_physics/
  videophy_2_auto/
  vbench_pretrained/
```

Download evaluator assets:

```bash
bash eval/scripts/download_videocon_physics.sh
bash eval/scripts/download_videophy2_auto.sh
bash eval/scripts/download_vbench_pretrained.sh
```

## Training

Edit `train/configs/model_config_train_eval.yaml`, then launch local training:

```bash
MODEL_CONFIG_YAML=train/configs/model_config_train_eval.yaml \
bash train/scripts/run_cluster_train_local_lamo.sh
```

Useful fields:

- `mode`: `train`, `eval`, or `train_eval`.
- `num_gpus`: number of GPUs for training.
- `eval_num_gpus`: number of GPUs for evaluation after training.
- `train_steps`, `batch_size`, `lr`: core optimization settings.
- `predictor_*`, `prompt_*`, `lambda_bmv_rel_l2`, `bmv_tau`: LaMo
  motion-prior settings.

The launcher uses `torch.distributed.run`; no cluster scheduler is required.
The template trains LoRA weights by default (`training_type: lora`). Evaluation
checkpoints are written under `<output_dir>/weight_<train_steps>` and should
contain `pytorch_lora_weights.safetensors` plus `predictor.safetensors`.

## Inference

Generate with LaMo:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/generate_lamo.py \
  --model_path <PATH_TO_COGVIDEOX_BASE_MODEL> \
  --lora_path <PATH_TO_LORA_WEIGHT_DIR> \
  --physical_module_path <PATH_TO_PREDICTOR_SAFETENSORS> \
  --prompt "A bowl of clear water slowly freezes into a transparent block of ice." \
  --output_file outputs/lamo \
  --guidance_lambda 15.0 \
  --guidance_step_ratio 0.8
```

If `predictor.safetensors` is placed next to the LoRA weights,
`--physical_module_path` can be omitted.

Generate with the CogVideoX baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/generate_cogvideox.py \
  --model_path <PATH_TO_COGVIDEOX_BASE_MODEL> \
  --prompt "A bowl of clear water slowly freezes into a transparent block of ice." \
  --output_file outputs/cogvideox
```

Both scripts also accept `--prompt_path assets/example.json`.

## Evaluation

The public evaluation launchers run on a single machine. They infer `NUM_GPUS`
from `CUDA_VISIBLE_DEVICES`, generate videos in parallel across those GPUs, then
run the benchmark metric after all videos are generated.

By default, the public launchers use `MODEL_TYPE=lamo` and
`GENERATE_TYPE=baseline`. In this repository, `baseline` means a materialized
checkpoint directory such as `pretrain_models/LaMo-CogVideoX-2b/` containing
`transformer/diffusion_pytorch_model.safetensors` and `predictor.safetensors`.

Use path overrides when your model weights live elsewhere:

```bash
BASE_MODEL=<PATH_TO_COGVIDEOX_BASE_MODEL> \
CHECKPOINT_PATH=<PATH_TO_LAMO_CHECKPOINT_DIR> \
CUDA_VISIBLE_DEVICES=0,1 bash eval/scripts/run_videophy_eval.sh
```

To evaluate a LoRA checkpoint from training, set `GENERATE_TYPE=lora`:

```bash
GENERATE_TYPE=lora \
BASE_MODEL=<PATH_TO_COGVIDEOX_BASE_MODEL> \
CHECKPOINT_PATH=<PATH_TO_OUTPUT_DIR>/weight_<train_steps> \
CUDA_VISIBLE_DEVICES=0,1 bash eval/scripts/run_videophy_eval.sh
```

Each run writes logs and summaries under `outputs/eval/`. Advanced environment
overrides are documented at the top of each launcher script.

### VideoPhy

VideoPhy uses `eval/benchmark_data/videophy/videophy_expanded_updated.csv` by
default. Video generation uses the `caption` column; the VideoPhy evaluator uses
`short_caption` for SA when that column is present and falls back to `caption`
otherwise.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash eval/scripts/run_videophy_eval.sh
```

### VideoPhy2

VideoPhy2 uses `eval/benchmark_data/videophy2/videophy2.csv` by default. If the
CSV contains `upsampled_caption`, generation uses `upsampled_caption`, while
the original `caption` is preserved as `short_caption` for SA evaluation.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash eval/scripts/run_videophy2_eval.sh
```

### VBench

VBench uses `eval/benchmark_data/vbench/VBench_full_info.json` by default and
generates `NUM_SAMPLES=5` videos per prompt for the standard dimensions. Video
generation uses the GPUs in `CUDA_VISIBLE_DEVICES`; VBench metric evaluation
uses `VBENCH_EVAL_GPUS=1` by default for local stability.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash eval/scripts/run_vbench_eval.sh
```

Small VBench smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_LIMIT=1 NUM_SAMPLES=1 \
DIMENSIONS=subject_consistency bash eval/scripts/run_vbench_eval.sh
```

## Interpretability Visualization

```bash
CUDA_VISIBLE_DEVICES=0 python tests/interpret_lamo.py \
  --model_path <PATH_TO_COGVIDEOX_BASE_MODEL> \
  --lora_path <PATH_TO_LORA_WEIGHT_DIR> \
  --predictor_ckpt <PATH_TO_PREDICTOR_SAFETENSORS> \
  --prompt "A coin spins rapidly on a wooden table." \
  --num_samples 1 \
  --work_dir outputs/interpret_lamo
```

The script writes per-sample frames, BMV heatmaps, predictor-response heatmaps,
a combined grid, and raw arrays under `--work_dir`.
