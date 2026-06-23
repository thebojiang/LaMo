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
# Download the VideoPhy-1 VideoCon-Physics evaluator checkpoint.
#
# Source HF repo (public, ~14 GB):
#   https://huggingface.co/videophysics/videocon_physics
#
# Usage:
#   bash eval/scripts/download_videocon_physics.sh
#   bash eval/scripts/download_videocon_physics.sh --force
#   bash eval/scripts/download_videocon_physics.sh --dest=/path/to/videocon_physics
#
# The default target is ./pretrain_models/videocon_physics, which is ignored by
# git because it contains model weights.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEST_ROOT="${VIDEOCON_PHYSICS_DIR:-${REPO_ROOT}/pretrain_models/videocon_physics}"
HF_REPO="${VIDEOCON_PHYSICS_REPO:-videophysics/videocon_physics}"
HF_REVISION="${VIDEOCON_PHYSICS_REVISION:-main}"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force|-f)        FORCE=1 ;;
        --dest=*)          DEST_ROOT="${arg#--dest=}" ;;
        --repo=*)          HF_REPO="${arg#--repo=}" ;;
        --revision=*)      HF_REVISION="${arg#--revision=}" ;;
        -h|--help)
            sed -n '2,24p' "$0"
            exit 0
            ;;
        *) echo "[videocon_physics] unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$DEST_ROOT"
echo "[videocon_physics] repo:   $HF_REPO@$HF_REVISION"
echo "[videocon_physics] target: $DEST_ROOT"
echo "[videocon_physics] force:  $FORCE"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

EXPECTED_FILES=(
    "README.md"
    "config.json"
    "generation_config.json"
    "preprocessor_config.json"
    "pytorch_model.bin"
    "tokenizer.model"
    "tokenizer_config.json"
)
EXPECTED_SIZES=(
    119
    9161
    202
    163
    14305657064
    499723
    176
)

is_complete() {
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
                echo "[videocon_physics] size mismatch for $path: have=$actual expect=$size" >&2
                return 1
            fi
        fi
    done
    return 0
}

if [[ "$FORCE" -eq 0 ]] && is_complete; then
    echo "[videocon_physics] All expected files already present in $DEST_ROOT; nothing to do."
    exit 0
fi

if [[ "$FORCE" -eq 1 ]]; then
    echo "[videocon_physics] --force given: clearing $DEST_ROOT"
    rm -rf "${DEST_ROOT:?}"/*
fi

try_hf_cli() {
    if ! have_cmd huggingface-cli; then
        echo "[videocon_physics] huggingface-cli not found; skipping CLI path."
        return 1
    fi
    echo "[videocon_physics] using huggingface-cli download ..."
    huggingface-cli download \
        "$HF_REPO" \
        --revision "$HF_REVISION" \
        --local-dir "$DEST_ROOT" \
        --local-dir-use-symlinks False
}

try_python_snapshot() {
    if ! have_cmd python && ! have_cmd python3; then
        echo "[videocon_physics] python not found; skipping snapshot_download path."
        return 1
    fi
    local PY
    PY=$(command -v python3 || command -v python)
    echo "[videocon_physics] using $PY -m huggingface_hub snapshot_download ..."
    "$PY" - <<PYEOF
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"[videocon_physics] huggingface_hub not installed: {e}", file=sys.stderr)
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

try_raw_http() {
    local downloader=""
    if have_cmd wget; then downloader="wget"
    elif have_cmd curl; then downloader="curl"
    else
        echo "[videocon_physics] neither wget nor curl available; cannot fall back."
        return 1
    fi
    echo "[videocon_physics] falling back to direct $downloader downloads ..."
    local base="https://huggingface.co/$HF_REPO/resolve/$HF_REVISION"
    for f in "${EXPECTED_FILES[@]}"; do
        local out="$DEST_ROOT/$f"
        mkdir -p "$(dirname "$out")"
        if [[ -s "$out" && "$FORCE" -eq 0 ]]; then
            echo "[videocon_physics] skip existing: $f"
            continue
        fi
        echo "[videocon_physics] downloading $f ..."
        if [[ "$downloader" == "wget" ]]; then
            wget -c -O "$out" "$base/$f?download=true"
        else
            curl -L --fail --retry 5 --retry-delay 10 -C - -o "$out" "$base/$f?download=true"
        fi
    done
}

if try_hf_cli; then
    echo "[videocon_physics] huggingface-cli download finished."
elif try_python_snapshot; then
    echo "[videocon_physics] huggingface_hub.snapshot_download finished."
elif try_raw_http; then
    echo "[videocon_physics] raw HTTP download finished."
else
    echo "[videocon_physics] ERROR: all download backends failed." >&2
    exit 1
fi

echo
echo "[videocon_physics] Verifying downloaded files ..."
if ! is_complete; then
    echo "[videocon_physics] WARNING: some expected files are still missing or look truncated." >&2
    ls -lh "$DEST_ROOT" || true
    exit 1
fi

echo "[videocon_physics] Done. Local checkpoint:"
du -sh "$DEST_ROOT"
ls -lh "$DEST_ROOT"
