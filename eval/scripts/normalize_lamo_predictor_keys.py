#!/usr/bin/env python3

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

"""Normalize LaMo predictor safetensors key prefixes.

The open-source codebase uses ``physical_lamo.``.  This utility can rewrite a
user-provided source prefix in ``predictor.safetensors`` without changing tensor
values.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LAMO_PREFIX = "physical_lamo."


def _resolve_predictor_path(path: Path) -> Path:
    if path.is_dir():
        return path / "predictor.safetensors"
    return path


def _next_backup_path(path: Path, suffix: str) -> Path:
    candidate = Path(str(path) + suffix)
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = Path(f"{path}{suffix}.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def normalize_predictor_keys(
    path: Path,
    *,
    from_prefix: str,
    dry_run: bool = False,
    backup: bool = True,
    backup_suffix: str = ".source_keys.bak",
) -> int:
    path = _resolve_predictor_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"predictor safetensors not found: {path}")

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        keys = list(handle.keys())
        source_keys = [key for key in keys if key.startswith(from_prefix)]
        modern_keys = [key for key in keys if key.startswith(LAMO_PREFIX)]

        print(f"[normalize_lamo] path: {path}")
        print(f"[normalize_lamo] total keys: {len(keys)}")
        print(f"[normalize_lamo] {LAMO_PREFIX} keys: {len(modern_keys)}")
        print(f"[normalize_lamo] source prefix ({from_prefix}) keys: {len(source_keys)}")

        if not source_keys:
            print("[normalize_lamo] already normalized; no changes needed")
            return 0

        renamed = {}
        for key in keys:
            new_key = key
            if key.startswith(from_prefix):
                new_key = LAMO_PREFIX + key[len(from_prefix):]
            if new_key in renamed:
                raise ValueError(
                    f"key collision while normalizing {path}: {key!r} -> {new_key!r}"
                )
            renamed[new_key] = handle.get_tensor(key)

    sample_old = source_keys[0]
    sample_new = LAMO_PREFIX + sample_old[len(from_prefix):]
    print(f"[normalize_lamo] sample rename: {sample_old} -> {sample_new}")

    if dry_run:
        print("[normalize_lamo] dry run; file was not modified")
        return len(source_keys)

    if backup:
        backup_path = _next_backup_path(path, backup_suffix)
        shutil.copy2(path, backup_path)
        print(f"[normalize_lamo] backup: {backup_path}")

    tmp_path = Path(str(path) + ".tmp")
    try:
        save_file(renamed, tmp_path, metadata=metadata)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"[normalize_lamo] rewrote {len(source_keys)} keys to {LAMO_PREFIX}")
    return len(source_keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite LaMo predictor keys from a source prefix to physical_lamo.*."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to predictor.safetensors, or a checkpoint directory containing it.",
    )
    parser.add_argument(
        "--from-prefix",
        required=True,
        help="Existing predictor key prefix to rewrite, for example '<SOURCE_PREFIX>'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without modifying the file.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Rewrite in place without creating a .source_keys.bak copy.",
    )
    args = parser.parse_args()

    normalize_predictor_keys(
        args.path,
        from_prefix=args.from_prefix,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
