#!/usr/bin/env bash

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

# =============================================================================
# Download all third-party pretrained weights required by VBench's evaluate.py
# into a local mirror directory. The resulting layout matches ~/.cache/vbench/.
#
#   bash eval/scripts/download_vbench_pretrained.sh --dest=<PATH_TO_VBENCH_PRETRAINED_ASSETS>
#
# The script is idempotent: files that already exist (with non-zero size) are
# skipped. Pass --force to re-download everything.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEST_ROOT="${VBENCH_PRETRAINED_DIR:-${REPO_ROOT}/pretrain_models/vbench_pretrained}"
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        --dest=*)   DEST_ROOT="${arg#--dest=}" ;;
        -h|--help)
            sed -n '2,13p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$DEST_ROOT"
echo "[vbench] target dir: $DEST_ROOT"
echo "[vbench] force=$FORCE"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

if ! have_cmd wget; then
    echo "[vbench] ERROR: wget not found; please install it." >&2
    exit 1
fi
if ! have_cmd unzip; then
    echo "[vbench] WARNING: unzip not found; RAFT models.zip will fail to extract." >&2
fi
if ! have_cmd git; then
    echo "[vbench] WARNING: git not found; DINO repo clone will be skipped." >&2
fi

# -----------------------------------------------------------------------------
# download <url> <abs_out_path>  [sha_expected_optional]
# -----------------------------------------------------------------------------
download() {
    local url="$1"
    local out="$2"
    local dir
    dir="$(dirname "$out")"
    mkdir -p "$dir"

    if [[ $FORCE -eq 0 && -s "$out" ]]; then
        printf "[vbench] OK   %s (already exists, %s bytes)\n" \
            "${out#"$DEST_ROOT/"}" "$(stat -c%s "$out")"
        return 0
    fi

    echo "[vbench] GET  ${out#"$DEST_ROOT/"}"
    echo "           <- $url"
    # -c resumes partial downloads; --tries retries transient failures.
    if ! wget --tries=5 --timeout=60 --continue --progress=bar:force:noscroll \
              -O "$out.part" "$url"; then
        echo "[vbench] FAIL $url" >&2
        rm -f "$out.part"
        return 1
    fi
    mv -f "$out.part" "$out"
}

# -----------------------------------------------------------------------------
# clone_repo <url> <abs_out_dir>
# -----------------------------------------------------------------------------
clone_repo() {
    local url="$1"
    local out="$2"
    if [[ $FORCE -eq 0 && -d "$out/.git" ]]; then
        echo "[vbench] OK   ${out#"$DEST_ROOT/"} (git repo already present)"
        return 0
    fi
    if ! have_cmd git; then
        echo "[vbench] SKIP clone $url (git missing)" >&2
        return 0
    fi
    rm -rf "$out"
    mkdir -p "$(dirname "$out")"
    echo "[vbench] CLONE ${out#"$DEST_ROOT/"} <- $url"
    git clone --depth=1 "$url" "$out"
}

FAILED=()

# --- CLIP ViT-B-32 ------------------------------------------------------------
# Used by: background_consistency, appearance_style
download \
    "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
    "$DEST_ROOT/clip_model/ViT-B-32.pt" \
    || FAILED+=("clip/ViT-B-32.pt")

# --- CLIP ViT-L-14 ------------------------------------------------------------
# Used by: aesthetic_quality
download \
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
    "$DEST_ROOT/clip_model/ViT-L-14.pt" \
    || FAILED+=("clip/ViT-L-14.pt")

# --- UMT ----------------------------------------------------------------------
# Used by: human_action
download \
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
    "$DEST_ROOT/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
    || FAILED+=("umt_model")

# --- AMT ----------------------------------------------------------------------
# Used by: motion_smoothness
download \
    "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth" \
    "$DEST_ROOT/amt_model/amt-s.pth" \
    || FAILED+=("amt_model")

# --- RAFT (zipped) ------------------------------------------------------------
# Used by: dynamic_degree
# VBench expects raft_model/models/raft-things.pth
RAFT_DIR="$DEST_ROOT/raft_model"
RAFT_TARGET="$RAFT_DIR/models/raft-things.pth"
if [[ $FORCE -eq 1 || ! -s "$RAFT_TARGET" ]]; then
    mkdir -p "$RAFT_DIR"
    RAFT_ZIP="$RAFT_DIR/models.zip"
    if download \
        "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" \
        "$RAFT_ZIP"; then
        if have_cmd unzip; then
            echo "[vbench] UNZIP raft_model/models.zip"
            unzip -o -q "$RAFT_ZIP" -d "$RAFT_DIR"
            rm -f "$RAFT_ZIP"
            if [[ ! -s "$RAFT_TARGET" ]]; then
                echo "[vbench] ERROR: $RAFT_TARGET not found after unzip" >&2
                FAILED+=("raft_model/raft-things.pth")
            fi
        else
            FAILED+=("raft_model (unzip missing)")
        fi
    else
        FAILED+=("raft_model")
    fi
else
    echo "[vbench] OK   raft_model/models/raft-things.pth (already exists)"
fi

# --- DINO ---------------------------------------------------------------------
# Used by: subject_consistency
download \
    "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" \
    "$DEST_ROOT/dino_model/dino_vitbase16_pretrain.pth" \
    || FAILED+=("dino_model/weights")

clone_repo \
    "https://github.com/facebookresearch/dino" \
    "$DEST_ROOT/dino_model/facebookresearch_dino_main" \
    || FAILED+=("dino_model/repo")

# --- LAION aesthetic predictor ------------------------------------------------
# Used by: aesthetic_quality
# Note: VBench reads this from ~/.cache/aesthetic_model/emb_reader/, and from
# ~/.cache/vbench/aesthetic_model/emb_reader/. We mirror it under the VBench
# cache so VBENCH_CACHE_DIR is self-contained.
download \
    "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth" \
    "$DEST_ROOT/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth" \
    || FAILED+=("aesthetic_model")

# --- pyIQA MUSIQ-SPAQ ---------------------------------------------------------
# Used by: imaging_quality
download \
    "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
    "$DEST_ROOT/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth" \
    || FAILED+=("pyiqa_model")

# --- GRiT ---------------------------------------------------------------------
# Used by: object_class, multiple_objects, color, spatial_relationship
download \
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth" \
    "$DEST_ROOT/grit_model/grit_b_densecap_objectdet.pth" \
    || FAILED+=("grit_model")

# --- Tag2Text -----------------------------------------------------------------
# Used by: scene
download \
    "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth" \
    "$DEST_ROOT/caption_model/tag2text_swin_14m.pth" \
    || FAILED+=("caption_model")

# --- ViCLIP -------------------------------------------------------------------
# Used by: temporal_style, overall_consistency
download \
    "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth" \
    "$DEST_ROOT/ViCLIP/ViClip-InternVid-10M-FLT.pth" \
    || FAILED+=("ViCLIP")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "[vbench] =============================================================="
echo "[vbench] Directory layout under $DEST_ROOT:"
if have_cmd tree; then
    tree -L 3 "$DEST_ROOT" | head -n 80
else
    find "$DEST_ROOT" -maxdepth 3 -printf '%p (%s bytes)\n' 2>/dev/null | sort
fi
echo "[vbench] =============================================================="

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo
    echo "[vbench] WARNING: the following downloads failed:" >&2
    for f in "${FAILED[@]}"; do echo "  - $f" >&2; done
    echo "[vbench] Re-run the script to retry failed items (wget --continue)." >&2
    exit 1
fi

echo
echo "[vbench] All weights ready at: $DEST_ROOT"
echo "[vbench] Set vbench_pretrained_path to this directory in your eval config."
