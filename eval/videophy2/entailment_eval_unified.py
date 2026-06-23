"""
Unified VideoPhy-2 evaluation script: 一次性跑完 SA (Semantic Adherence) + PC
(Physical Commonsense) 两个指标，并输出 per-sample CSV + overall / per-subcategory
summary。

与 VideoPhy-1 (`entailment_eval_unified.py`) 的区别：
  - 使用 VideoPhy-2 AutoEvaluator（MplugOwl 微调版本，checkpoint: videophy_2_auto），
    通过 `model.generate(...)` 做自回归评分（输出 1-5 的整数）。
  - prompt 模板来自 `template.py`（PROMPT_SA / PROMPT_PHYSICS）。
  - 指标与 VideoPhy-2 论文一致：
      * SA / PC 的 mean、ratio>=3、ratio>=4、ratio>=5（accuracy）
      * Joint：SA>=4 AND PC>=4 的比例
  - 输入 CSV 格式（至少含 videopath、caption）：
      videopath, caption, [short_caption], [states_of_matter], [complexity],
      [majority_sa], [majority_pc]
    其中：
      - 若存在 `short_caption`，SA 评测使用 short_caption（与 VideoPhy-1 的 VideoREPA
        对齐协议一致：用简短的原始 prompt 进行 SA 判断）；否则 fallback 到 caption。
      - states_of_matter / complexity / majority_* 仅用于 summary 分组。

Reference: VideoREPA/evaluation/VIDEOPHY2/inference.py + calculate_mean.py，
在此合并成一次性脚本。
"""

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from template import PROMPT_SA, PROMPT_PHYSICS


GENERATE_KWARGS = {
    "do_sample": False,
    "top_k": 1,
    "temperature": 0.001,
    "max_length": 256,
}

# 英文数字 + 阿拉伯数字 -> 数值
NUM_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def _parse_score(output_text: str) -> int:
    """Parse MplugOwl free-form output into an integer 1-5 (0 on failure)."""
    lower = output_text.lower().strip()
    for key, val in NUM_MAP.items():
        if key in lower:
            return val
    digits = "".join([c for c in lower if c.isdigit()])
    if digits and int(digits) in NUM_MAP.values():
        return int(digits)
    print(f"[videophy2_eval] Warning: could not parse '{output_text}', defaulting to 0.", flush=True)
    return 0


def _run_one(model, processor, tokenizer, videopath: str, prompt: str, num_frames: int) -> int:
    inputs = processor(text=[prompt], videos=[videopath], num_frames=num_frames,
                       return_tensors="pt")
    inputs = {k: (v.bfloat16() if torch.is_tensor(v) and v.dtype == torch.float else v)
              for k, v in inputs.items()}
    inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **GENERATE_KWARGS)
    output_text = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return _parse_score(output_text)


def _expand_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Each input row -> 2 output rows (SA + PC) for downstream CSV."""
    if "caption" not in df.columns:
        raise ValueError("Input CSV must have 'caption' column.")
    if "videopath" not in df.columns:
        raise ValueError("Input CSV must have 'videopath' column.")
    df = df.dropna(subset=["caption", "videopath"]).reset_index(drop=True)

    has_short = "short_caption" in df.columns
    if has_short:
        print("[videophy2_eval] Using 'short_caption' column for SA evaluation.", flush=True)
    else:
        print("[videophy2_eval] No 'short_caption' column; using 'caption' for SA evaluation.", flush=True)

    for col in ("states_of_matter", "complexity", "majority_sa", "majority_pc"):
        if col not in df.columns:
            df[col] = None

    rows = []
    for _, r in df.iterrows():
        caption_full = r["caption"]
        sa_caption = (str(r["short_caption"]).strip()
                      if has_short and pd.notna(r.get("short_caption")) else caption_full)
        rows.append({
            "videopath": r["videopath"],
            "caption": caption_full,
            "sa_caption": sa_caption,
            "states_of_matter": r["states_of_matter"],
            "complexity": r["complexity"],
            "majority_sa": r["majority_sa"],
            "majority_pc": r["majority_pc"],
        })
    return pd.DataFrame(rows)


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _summarize(out_df: pd.DataFrame) -> list:
    """Compute overall + per-subcategory metrics. Returns list of text lines."""
    lines = []
    sa = out_df["sa_score"]
    pc = out_df["pc_score"]
    n = len(out_df)

    def _metrics(series):
        return {
            "mean": float(series.mean()) if len(series) else 0.0,
            "r3": float((series >= 3).mean()) if len(series) else 0.0,
            "r4": float((series >= 4).mean()) if len(series) else 0.0,
            "r5": float((series >= 5).mean()) if len(series) else 0.0,
        }

    sa_m = _metrics(sa)
    pc_m = _metrics(pc)
    joint_r4 = float(((sa >= 4) & (pc >= 4)).mean()) if n else 0.0

    lines.append("=== Overall (n={}) ===".format(n))
    lines.append(
        f"SA Mean: {sa_m['mean']:.3f}, >=3: {_format_pct(sa_m['r3'])}, "
        f">=4: {_format_pct(sa_m['r4'])}, >=5: {_format_pct(sa_m['r5'])}"
    )
    lines.append(
        f"PC Mean: {pc_m['mean']:.3f}, >=3: {_format_pct(pc_m['r3'])}, "
        f">=4: {_format_pct(pc_m['r4'])}, >=5: {_format_pct(pc_m['r5'])}"
    )
    lines.append(f"Joint (SA>=4 AND PC>=4): {_format_pct(joint_r4)}")

    if "states_of_matter" in out_df.columns and out_df["states_of_matter"].notna().any():
        lines.append("")
        lines.append("=== By states_of_matter ===")
        for som in sorted(out_df["states_of_matter"].dropna().unique(), key=str):
            sub = out_df[out_df["states_of_matter"] == som]
            if not len(sub):
                continue
            sm = _metrics(sub["sa_score"])
            pm = _metrics(sub["pc_score"])
            jr4 = float(((sub["sa_score"] >= 4) & (sub["pc_score"] >= 4)).mean())
            lines.append(
                f"  {som} (n={len(sub)}): SA_mean={sm['mean']:.3f} SA>=4={_format_pct(sm['r4'])} "
                f"PC_mean={pm['mean']:.3f} PC>=4={_format_pct(pm['r4'])} Joint>=4={_format_pct(jr4)}"
            )

    if "complexity" in out_df.columns and out_df["complexity"].notna().any():
        lines.append("")
        lines.append("=== By complexity ===")
        for c in sorted(out_df["complexity"].dropna().unique(), key=str):
            sub = out_df[out_df["complexity"] == c]
            if not len(sub):
                continue
            sm = _metrics(sub["sa_score"])
            pm = _metrics(sub["pc_score"])
            jr4 = float(((sub["sa_score"] >= 4) & (sub["pc_score"] >= 4)).mean())
            lines.append(
                f"  complexity={c} (n={len(sub)}): SA_mean={sm['mean']:.3f} "
                f"SA>=4={_format_pct(sm['r4'])} PC_mean={pm['mean']:.3f} "
                f"PC>=4={_format_pct(pm['r4'])} Joint>=4={_format_pct(jr4)}"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description="Unified VideoPhy-2 SA+PC evaluation")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input manifest CSV: videopath, caption (+ optional short_caption, "
                             "states_of_matter, complexity, majority_sa, majority_pc).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV with per-sample sa_score/pc_score; also writes "
                             "_summary.txt next to it.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="VideoPhy-2 AutoEvaluator checkpoint dir (videophy_2_auto).")
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Frames sampled from each video for the evaluator (VideoPhy-2 uses 32).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Kept for CLI parity; VideoPhy-2 autoregressive eval runs one sample at a time.")
    args = parser.parse_args()

    in_df = pd.read_csv(args.input_csv)
    eval_df = _expand_rows(in_df)
    print(f"[videophy2_eval] Evaluating {len(eval_df)} videos (SA+PC each = {2 * len(eval_df)} calls)",
          flush=True)

    print(f"[videophy2_eval] Loading checkpoint: {args.checkpoint}", flush=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(args.checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model = MplugOwlForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    print("[videophy2_eval] Model loaded.", flush=True)

    sa_scores = []
    pc_scores = []
    for i in tqdm(range(len(eval_df)), desc="VideoPhy-2 SA+PC"):
        r = eval_df.iloc[i]
        videopath = r["videopath"]
        sa_prompt = PROMPT_SA.format(caption=r["sa_caption"])
        pc_prompt = PROMPT_PHYSICS
        try:
            sa_scores.append(_run_one(model, processor, tokenizer, videopath, sa_prompt, args.num_frames))
        except Exception as e:
            print(f"[videophy2_eval] SA inference failed for {videopath}: {e}", flush=True)
            sa_scores.append(0)
        try:
            pc_scores.append(_run_one(model, processor, tokenizer, videopath, pc_prompt, args.num_frames))
        except Exception as e:
            print(f"[videophy2_eval] PC inference failed for {videopath}: {e}", flush=True)
            pc_scores.append(0)

    eval_df["sa_score"] = sa_scores
    eval_df["pc_score"] = pc_scores

    out_cols = [c for c in
                ("videopath", "caption", "sa_caption", "sa_score", "pc_score",
                 "states_of_matter", "complexity", "majority_sa", "majority_pc")
                if c in eval_df.columns]
    eval_df[out_cols].to_csv(args.output_csv, index=False)
    print(f"[videophy2_eval] Results written to {args.output_csv}", flush=True)

    summary_lines = _summarize(eval_df)
    summary_path = args.output_csv.rsplit(".", 1)[0] + "_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    for line in summary_lines:
        print(line, flush=True)
    print(f"\n[videophy2_eval] Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
