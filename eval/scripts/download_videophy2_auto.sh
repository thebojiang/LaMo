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
# Download the VideoPhy-2 AutoEvaluator checkpoint (MplugOwl-7B finetune used
# by `eval/videophy2/entailment_eval_unified.py` and
# `eval/cluster_videophy2_eval.py`) to a local mirror directory.
#
# Source HF repo (canonical, MIT license, public, ~14.3 GB):
#   https://huggingface.co/videophysics/videophy_2_auto
#
# Usage:
#   bash eval/scripts/download_videophy2_auto.sh                # default target + repo
#   bash eval/scripts/download_videophy2_auto.sh --force        # force re-download
#   bash eval/scripts/download_videophy2_auto.sh --dest=/path   # override target dir
#   bash eval/scripts/download_videophy2_auto.sh --repo=hbXNov/videophy_2_auto   # override repo
#
# The script tries, in order:
#   1) `huggingface-cli download` (preferred; resumes, validates LFS hashes)
#   2) Python `huggingface_hub.snapshot_download`
#   3) Direct `wget`/`curl` of every file via `https://huggingface.co/.../resolve/main/...`
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEST_ROOT="${VIDEOPHY2_AUTO_DIR:-${REPO_ROOT}/pretrain_models/videophy_2_auto}"
HF_REPO="${VIDEOPHY2_AUTO_REPO:-videophysics/videophy_2_auto}"
HF_REVISION="${VIDEOPHY2_AUTO_REVISION:-main}"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force|-f)        FORCE=1 ;;
        --dest=*)          DEST_ROOT="${arg#--dest=}" ;;
        --repo=*)          HF_REPO="${arg#--repo=}" ;;
        --revision=*)      HF_REVISION="${arg#--revision=}" ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) echo "[videophy2_auto] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$DEST_ROOT"
echo "[videophy2_auto] repo:   $HF_REPO@$HF_REVISION"
echo "[videophy2_auto] target: $DEST_ROOT"
echo "[videophy2_auto] force:  $FORCE"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Files we expect (path -> approx size in bytes; size used only for sanity).
# This list mirrors `huggingface.co/api/models/videophysics/videophy_2_auto/tree/main`.
# Keep in sync if upstream layout changes.
EXPECTED_FILES=(
    ".gitattributes"
    "README.md"
    "config.json"
    "generation_config.json"
    "preprocessor_config.json"
    "pytorch_model-00001-of-00002.bin"
    "pytorch_model-00002-of-00002.bin"
    "pytorch_model.bin.index.json"
    "tokenizer.model"
    "tokenizer_config.json"
)
EXPECTED_SIZES=(
    1519
    268
    7036
    179
    163
    9995752315
    4309856976
    92527
    499723
    176
)

is_complete() {
    # Returns 0 if all expected files exist and are non-empty (and, for LFS
    # blobs, match expected byte size within a small tolerance).
    local i path size actual
    for i in "${!EXPECTED_FILES[@]}"; do
        path="$DEST_ROOT/${EXPECTED_FILES[$i]}"
        size="${EXPECTED_SIZES[$i]}"
        if [[ ! -s "$path" ]]; then
            return 1
        fi
        if [[ "$size" -gt 1048576 ]]; then
            actual=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0)
            if (( actual + 1024 < size )); then
                echo "[videophy2_auto] size mismatch for $path: have=$actual expect=$size" >&2
                return 1
            fi
        fi
    done
    return 0
}

if [[ "$FORCE" -eq 0 ]] && is_complete; then
    echo "[videophy2_auto] All expected files already present in $DEST_ROOT; nothing to do."
    echo "[videophy2_auto] (Use --force to re-download.)"
    exit 0
fi

# Force re-download: clear the dir first.
if [[ "$FORCE" -eq 1 ]]; then
    echo "[videophy2_auto] --force given: clearing $DEST_ROOT"
    rm -rf "${DEST_ROOT:?}"/*
fi

# -----------------------------------------------------------------------------
# 1) huggingface-cli download (preferred)
# -----------------------------------------------------------------------------
try_hf_cli() {
    if ! have_cmd huggingface-cli; then
        echo "[videophy2_auto] huggingface-cli not found; skipping CLI path."
        return 1
    fi
    echo "[videophy2_auto] using huggingface-cli download ..."
    # --resume-download is the default in recent versions; flag is harmless.
    huggingface-cli download \
        "$HF_REPO" \
        --revision "$HF_REVISION" \
        --local-dir "$DEST_ROOT" \
        --local-dir-use-symlinks False
}

# -----------------------------------------------------------------------------
# 2) Python snapshot_download fallback
# -----------------------------------------------------------------------------
try_python_snapshot() {
    if ! have_cmd python && ! have_cmd python3; then
        echo "[videophy2_auto] python not found; skipping snapshot_download path."
        return 1
    fi
    local PY
    PY=$(command -v python3 || command -v python)
    echo "[videophy2_auto] using $PY -m huggingface_hub snapshot_download ..."
    "$PY" - <<PYEOF
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"[videophy2_auto] huggingface_hub not installed: {e}", file=sys.stderr)
    sys.exit(3)
snapshot_download(
    repo_id="$HF_REPO",
    revision="$HF_REVISION",
    local_dir="$DEST_ROOT",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PYEOF
}

# -----------------------------------------------------------------------------
# 3) Raw HTTP fallback (per-file wget/curl) — last resort if neither of the
#    huggingface_hub-based paths is available.
# -----------------------------------------------------------------------------
try_raw_http() {
    local downloader=""
    if have_cmd wget; then downloader="wget"
    elif have_cmd curl; then downloader="curl"
    else
        echo "[videophy2_auto] neither wget nor curl available; cannot fall back."
        return 1
    fi
    echo "[videophy2_auto] falling back to direct $downloader downloads ..."
    local base="https://huggingface.co/$HF_REPO/resolve/$HF_REVISION"
    for f in "${EXPECTED_FILES[@]}"; do
        local out="$DEST_ROOT/$f"
        mkdir -p "$(dirname "$out")"
        if [[ -s "$out" && "$FORCE" -eq 0 ]]; then
            echo "[videophy2_auto] skip existing: $f"
            continue
        fi
        echo "[videophy2_auto] downloading $f ..."
        if [[ "$downloader" == "wget" ]]; then
            wget -c -O "$out" "$base/$f?download=true"
        else
            curl -L --fail --retry 5 --retry-delay 10 -C - -o "$out" "$base/$f?download=true"
        fi
    done
}

if try_hf_cli; then
    echo "[videophy2_auto] huggingface-cli download finished."
elif try_python_snapshot; then
    echo "[videophy2_auto] huggingface_hub.snapshot_download finished."
elif try_raw_http; then
    echo "[videophy2_auto] raw HTTP download finished."
else
    echo "[videophy2_auto] ERROR: all download backends failed." >&2
    exit 1
fi

echo
echo "[videophy2_auto] Verifying downloaded files ..."
if ! is_complete; then
    echo "[videophy2_auto] WARNING: some expected files are still missing or look truncated." >&2
    ls -lh "$DEST_ROOT" || true
    exit 1
fi

echo "[videophy2_auto] Done. Local checkpoint:"
du -sh "$DEST_ROOT"
ls -lh "$DEST_ROOT"

cat <<'NEXT'

================================================================================
Next step: set this local path in your eval YAML:

  videophy2_checkpoint: <PATH_TO_VIDEOPHY2_EVALUATOR>
================================================================================
NEXT
