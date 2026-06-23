"""
Unified VideoPhy evaluation script: runs both SA (Semantic Alignment) and PC (Physical Consistency)
in a single pass, and outputs results CSV plus overall and per-subcategory (states_of_matter, complexity) averages.

Supported input CSV formats:
  - Step 1 manifest CSV: videopath, caption, states_of_matter, complexity, (source, majority_sa, majority_pc).
  - Original public CSV: video_url, caption, ... (use --video_root if video_url is URL and videos are under a local dir).
"""
import os
import csv
import tempfile
import argparse
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from data_utils.xgpt3_dataset import MultiModalDataset
from utils import batchify, set_args

# Prompt templates matching sa_testing.csv and physics_testing.csv
CONVERSATION_PREFIX = (
    "The following is a conversation between a curious human and AI assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    "Human: <|video|>\nHuman: "
)
SA_PROMPT_TEMPLATE = CONVERSATION_PREFIX + 'Does this video entail the description: "{}"?\nAI: '
PC_PROMPT_TEMPLATE = CONVERSATION_PREFIX + "Does this video follow the physical laws?\nAI: "

parser = argparse.ArgumentParser(description="Unified VideoPhy SA+PC evaluation")
parser.add_argument("--input_csv", type=str, required=True,
                    help="Input CSV: Step 1 manifest (videopath, caption, states_of_matter, complexity) or public CSV (video_url, caption). Must have caption and videopath or video_url.")
parser.add_argument("--output_csv", type=str, required=True,
                    help="Output CSV: caption, task_type, score, states_of_matter, complexity, majority_sa, majority_pc, videopath/video_url")
parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--video_root", type=str, default=None,
                    help="Local directory containing videos. If input CSV has video_url as URL, "
                         "videopath = os.path.join(video_root, basename(url)). If URLs are used directly, leave unset only when paths are already local.")

args = parser.parse_args()
softmax = nn.Softmax(dim=2)


def _videopath_from_row(url_or_path, video_root):
    """Resolve videopath: use video_root + basename if video_root set and path looks like URL."""
    s = str(url_or_path).strip()
    if not s:
        return s
    if video_root and (s.startswith("http://") or s.startswith("https://")):
        return os.path.join(video_root, os.path.basename(s.split("?")[0]))
    return s


def build_expanded_dataset(input_csv_path, video_root):
    """
    Read input CSV and build expanded rows: each original row becomes two rows (one SA, one PC).
    Input must have caption and either videopath or video_url. Optional: states_of_matter, complexity, majority_sa, majority_pc.

    SA evaluation uses 'short_caption' column if present (the original VideoPhy short prompt),
    matching the VideoREPA evaluation protocol. If 'short_caption' is absent, falls back to 'caption'.

    Returns: pd.DataFrame with columns videopath, caption, task_type, caption_text, majority_sa, majority_pc,
             states_of_matter, complexity, video_url (or videopath as reference).
    """
    df = pd.read_csv(input_csv_path)
    if "caption" not in df.columns:
        raise ValueError("Input CSV must have 'caption' column.")
    df = df.dropna(subset=["caption"])

    has_short_caption = "short_caption" in df.columns
    if has_short_caption:
        print("[entailment_eval] Using 'short_caption' column for SA evaluation (VideoREPA-aligned).")
    else:
        print("[entailment_eval] No 'short_caption' column found; using 'caption' for SA evaluation.")

    # Path: prefer videopath; else video_url + video_root
    if "videopath" in df.columns:
        path_col = "videopath"
    elif "video_url" in df.columns:
        path_col = "video_url"
    else:
        raise ValueError("Input CSV must have 'videopath' or 'video_url' column.")
    if "majority_sa" not in df.columns:
        df["majority_sa"] = None
    if "majority_pc" not in df.columns:
        df["majority_pc"] = None
    if "states_of_matter" not in df.columns:
        df["states_of_matter"] = None
    if "complexity" not in df.columns:
        df["complexity"] = None

    rows = []
    for _, r in df.iterrows():
        caption_text = r["caption"]
        sa_caption_text = str(r["short_caption"]).strip() if has_short_caption and pd.notna(r.get("short_caption")) else caption_text
        raw_path = r[path_col]
        videopath = raw_path if path_col == "videopath" else _videopath_from_row(raw_path, video_root)
        majority_sa = r.get("majority_sa")
        majority_pc = r.get("majority_pc")
        states_of_matter = r.get("states_of_matter")
        complexity = r.get("complexity")
        video_ref = r.get("video_url") if "video_url" in df.columns else r.get("videopath")

        caption_sa = SA_PROMPT_TEMPLATE.format(sa_caption_text)
        caption_pc = PC_PROMPT_TEMPLATE

        for task_type, cap in [("SA", caption_sa), ("PC", caption_pc)]:
            rows.append({
                "videopath": videopath,
                "caption": cap,
                "task_type": task_type,
                "caption_text": caption_text,
                "majority_sa": majority_sa,
                "majority_pc": majority_pc,
                "states_of_matter": states_of_matter,
                "complexity": complexity,
                "video_url": video_ref,
            })

    return pd.DataFrame(rows)


def get_entail(logits, input_ids, tokenizer):
    logits = softmax(logits)
    token_id_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_id_no = tokenizer.encode("No", add_special_tokens=False)[0]
    entailment = []
    for j in range(len(logits)):
        i = 0
        for i in range(len(input_ids[j])):
            if input_ids[j][i] == tokenizer.pad_token_id:
                i = i - 1
                break
            elif i == len(input_ids[j]) - 1:
                break
        score = logits[j][i][token_id_yes] / (
            logits[j][i][token_id_yes] + logits[j][i][token_id_no]
        )
        entailment.append(score)
    return torch.stack(entailment)


def run_inference(model, tokenizer, dataloader):
    """Run model on dataloader and return list of (videopath, caption, score) in order."""
    results = []
    with torch.no_grad():
        for index, inputs in tqdm(enumerate(dataloader), desc="Inference"):
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if v.dtype == torch.float:
                        inputs[k] = v.bfloat16()
                    inputs[k] = inputs[k].to(model.device)
            outputs = model(
                pixel_values=inputs["pixel_values"],
                video_pixel_values=inputs["video_pixel_values"],
                labels=None,
                num_images=inputs["num_images"],
                num_videos=inputs["num_videos"],
                input_ids=inputs["input_ids"],
                non_padding_mask=inputs["non_padding_mask"],
                non_media_mask=inputs["non_media_mask"],
                prompt_mask=inputs["prompt_mask"],
            )
            logits = outputs["logits"]
            entail_scores = get_entail(logits, inputs["input_ids"], tokenizer)
            for m in range(len(entail_scores)):
                results.append({
                    "videopath": inputs["videopaths"][m],
                    "caption": inputs["captions"][m],
                    "score": entail_scores[m].item(),
                })
    return results


def main():
    input_csv = args.input_csv
    output_csv = args.output_csv
    checkpoint = args.checkpoint
    video_root = args.video_root

    # Build expanded dataset (SA + PC per video)
    expanded_df = build_expanded_dataset(input_csv, video_root)
    print(f"Expanded dataset: {len(expanded_df)} rows ({len(expanded_df)//2} videos x 2 tasks)")

    # Write temp CSV with only videopath, caption for MultiModalDataset (same order as expanded_df)
    temp_fd, temp_csv_path = tempfile.mkstemp(suffix=".csv")
    try:
        os.close(temp_fd)
        expanded_df[["videopath", "caption"]].to_csv(temp_csv_path, index=False)

        # Minimal args for dataset (get_args may be used inside MultiModalDataset)
        class EvalArgs:
            pass
        eval_args = EvalArgs()
        set_args(eval_args)

        tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
        image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
        processor = MplugOwlProcessor(image_processor, tokenizer)

        valid_data = MultiModalDataset(
            temp_csv_path, tokenizer, processor,
            max_length=256, loss_objective="sequential",
        )
        dataloader = DataLoader(
            valid_data, batch_size=args.batch_size, pin_memory=True, collate_fn=batchify,
        )

        model = MplugOwlForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        print("Model loaded.")
        model.eval()

        inference_results = run_inference(model, tokenizer, dataloader)
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

    # Assign scores by order (same as expanded_df)
    if len(inference_results) != len(expanded_df):
        raise RuntimeError(
            f"Result count mismatch: {len(inference_results)} vs expanded {len(expanded_df)}"
        )
    expanded_df["score"] = [r["score"] for r in inference_results]

    # Output CSV: include states_of_matter, complexity
    out_cols = ["video_url", "caption_text", "task_type", "score", "states_of_matter", "complexity", "majority_sa", "majority_pc"]
    out_cols = [c for c in out_cols if c in expanded_df.columns]
    out_df = expanded_df[out_cols].copy()
    out_df = out_df.rename(columns={"caption_text": "caption"})
    out_df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")

    # Overall and subcategory average SA / PC (only these are printed; per-sample details are in CSV only)
    # Use thresholding (>= 0.5) to align with VideoREPA evaluation method (accuracy instead of avg probability)
    summary_path = output_csv.rsplit(".", 1)[0] + "_summary.txt"
    summary_lines = []

    sa_all = expanded_df.loc[expanded_df["task_type"] == "SA", "score"]
    pc_all = expanded_df.loc[expanded_df["task_type"] == "PC", "score"]
    # Threshold at 0.5: convert to binary (0 or 1) then compute accuracy
    sa_acc = float((sa_all >= 0.5).astype(int).mean())
    pc_acc = float((pc_all >= 0.5).astype(int).mean())
    summary_lines.append("=== Overall ===")
    summary_lines.append(f"SA Accuracy: {sa_acc:.4f}")
    summary_lines.append(f"PC Accuracy: {pc_acc:.4f}")
    print("=== Overall ===")
    print(f"SA Accuracy: {sa_acc:.4f}")
    print(f"PC Accuracy: {pc_acc:.4f}")

    # By states_of_matter
    if "states_of_matter" in expanded_df.columns and expanded_df["states_of_matter"].notna().any():
        summary_lines.append("")
        summary_lines.append("=== By states_of_matter ===")
        print("\n=== By states_of_matter ===")
        for som in sorted(expanded_df["states_of_matter"].dropna().unique(), key=str):
            mask = expanded_df["states_of_matter"] == som
            sa_m = expanded_df.loc[mask & (expanded_df["task_type"] == "SA"), "score"]
            pc_m = expanded_df.loc[mask & (expanded_df["task_type"] == "PC"), "score"]
            if len(sa_m) and len(pc_m):
                sa_acc = float((sa_m >= 0.5).astype(int).mean())
                pc_acc = float((pc_m >= 0.5).astype(int).mean())
                summary_lines.append(f"  {som}: SA_acc={sa_acc:.4f}, PC_acc={pc_acc:.4f} (n={len(sa_m)})")
                print(f"  {som}: SA_acc={sa_acc:.4f}, PC_acc={pc_acc:.4f} (n={len(sa_m)})")

    # By complexity (use original dtype for masking; CSV may have int/float complexity)
    if "complexity" in expanded_df.columns and expanded_df["complexity"].notna().any():
        summary_lines.append("")
        summary_lines.append("=== By complexity ===")
        print("\n=== By complexity ===")
        for comp in sorted(expanded_df["complexity"].dropna().unique(), key=str):
            mask = expanded_df["complexity"] == comp
            sa_m = expanded_df.loc[mask & (expanded_df["task_type"] == "SA"), "score"]
            pc_m = expanded_df.loc[mask & (expanded_df["task_type"] == "PC"), "score"]
            if len(sa_m) and len(pc_m):
                sa_acc = float((sa_m >= 0.5).astype(int).mean())
                pc_acc = float((pc_m >= 0.5).astype(int).mean())
                summary_lines.append(f"  complexity={comp}: SA_acc={sa_acc:.4f}, PC_acc={pc_acc:.4f} (n={len(sa_m)})")
                print(f"  complexity={comp}: SA_acc={sa_acc:.4f}, PC_acc={pc_acc:.4f} (n={len(sa_m)})")

    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
