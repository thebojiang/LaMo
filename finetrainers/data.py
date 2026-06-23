# Copyright (c) 2026 Applied Intuition, Inc.
#
# This file is part of a modified version of finetrainers
# (https://github.com/huggingface/finetrainers), Copyright the
# finetrainers contributors, licensed under the Apache License, Version
# 2.0. Modifications by Applied Intuition, Inc.
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

from __future__ import annotations

import csv
import itertools
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .constants import COMMON_LLM_START_PHRASES, SUPPORTED_IMAGE_FILE_EXTENSIONS, SUPPORTED_VIDEO_FILE_EXTENSIONS
from .functional import remove_prefix, resize_to_nearest_bucket_video
from .logging import get_logger


logger = get_logger()


@dataclass
class ImageArtifact:
    value: Any


@dataclass
class VideoArtifact:
    value: Any


class _SkipSample(Exception):
    pass


class RawDataset(IterableDataset):
    def __init__(
        self,
        data: Iterable[Dict[str, Any]],
        *,
        dataset_type: str,
        data_root: Optional[str] = None,
        infinite: bool = True,
        caption_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._data = data
        self.dataset_type = dataset_type
        self.data_root = data_root
        self.infinite = infinite
        self.caption_options = caption_options or {}
        self._precomputable_once = not infinite

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while True:
            yielded = False
            for row in self._data:
                yielded = True
                yield dict(row)
            if not self.infinite:
                return
            if not yielded:
                raise ValueError("Dataset produced no rows.")


class PreprocessedDataset(IterableDataset):
    def __init__(self, dataset: RawDataset, dataset_type: str, config: Dict[str, Any]) -> None:
        self._dataset = dataset
        self._data = dataset._data
        self.dataset_type = dataset_type
        self.data_root = dataset.data_root
        self.infinite = dataset.infinite
        self.caption_options = dataset.caption_options
        self.config = dict(config)
        self._precomputable_once = dataset._precomputable_once

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        iterator = iter(self._dataset)
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            iterator = itertools.islice(iterator, worker.id, None, worker.num_workers)

        for row in iterator:
            try:
                yield self._prepare_row(row)
            except _SkipSample:
                continue
            except Exception as exc:
                logger.warning("Skipping sample after preprocessing error: %s", exc)
                continue

    def _prepare_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        dataset_type = self.dataset_type
        caption = _extract_caption(row, self.config, self.caption_options)
        sample: Dict[str, Any] = {"caption": caption}

        if dataset_type == "video":
            video_path = _resolve_media_path(row, self.data_root, media_type="video")
            video = _load_video(video_path)
            video = _process_video(video, self.config)
            sample["video"] = video
            sample["num_frames"], _, sample["height"], sample["width"] = video.shape
            sample["video_path"] = str(video_path)
            return sample

        if dataset_type == "image":
            image_path = _resolve_media_path(row, self.data_root, media_type="image")
            image = _load_image(image_path)
            image = _process_image(image, self.config)
            sample["image"] = image
            _, sample["height"], sample["width"] = image.shape
            sample["image_path"] = str(image_path)
            return sample

        raise ValueError(f"Unsupported dataset_type for preprocessing: {dataset_type}")


class CombinedIterableDataset(IterableDataset):
    def __init__(self, datasets: Sequence[IterableDataset], *, buffer_size: int = 1, shuffle: bool = True) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required.")
        self.datasets = list(datasets)
        self.buffer_size = max(int(buffer_size), 1)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        iterators = [iter(dataset) for dataset in self.datasets]
        source = _round_robin(iterators)
        if not self.shuffle or self.buffer_size <= 1:
            yield from source
            return

        buffer: List[Dict[str, Any]] = []
        for item in source:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                idx = random.randrange(len(buffer))
                yield buffer.pop(idx)
        while buffer:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)


class DPDataLoader(DataLoader):
    """DataLoader with a lightweight state_dict for torch.distributed.checkpoint."""

    def __init__(self, dp_rank: int, *args, **kwargs) -> None:
        self.dp_rank = dp_rank
        super().__init__(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        return {"dp_rank": self.dp_rank}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.dp_rank = int(state_dict.get("dp_rank", self.dp_rank))


class InMemoryDistributedDataPreprocessor:
    def __init__(
        self,
        *,
        rank: int,
        num_items: int,
        processor_fn: Dict[str, Any],
        save_dir: Optional[str] = None,
        enable_precomputation: bool = False,
        hash_save: bool = False,
        model_name: Optional[str] = None,
    ) -> None:
        self.rank = rank
        self.num_items = max(int(num_items), 1)
        self.processor_fn = processor_fn
        self.save_dir = save_dir
        self.enable_precomputation = enable_precomputation
        self.hash_save = hash_save
        self.model_name = model_name
        self._cached_samples: List[Dict[str, Any]] = []

    @property
    def requires_data(self) -> bool:
        return True

    def consume(
        self,
        key: str,
        *,
        components: Dict[str, Any],
        data_iterator: Iterator[Dict[str, Any]],
        generator: Optional[torch.Generator] = None,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
        first_samples: bool = False,
        **_: Any,
    ) -> Iterator[Dict[str, Any]]:
        del first_samples

        def _iterator() -> Iterator[Dict[str, Any]]:
            if use_cached_samples:
                samples = list(self._cached_samples)
            else:
                samples = []
                for _ in range(self.num_items):
                    try:
                        samples.append(next(data_iterator))
                    except StopIteration:
                        break
                if cache_samples:
                    self._cached_samples = list(samples)

            for sample in samples:
                yield self._process(key, sample, components, generator)

            if drop_samples:
                self._cached_samples = []

        return _iterator()

    def consume_hash_dict(self, *args, **kwargs):
        return self.consume(*args, **kwargs)

    def consume_save_hash_dict(self, *args, **kwargs):
        return self.consume(*args, **kwargs)

    def consume_once(self, *args, **kwargs):
        return self.consume(*args, **kwargs)

    def _process(
        self,
        key: str,
        sample: Dict[str, Any],
        components: Dict[str, Any],
        generator: Optional[torch.Generator],
    ) -> Dict[str, Any]:
        if key not in self.processor_fn:
            raise KeyError(f"Unknown preprocessor key: {key}")
        kwargs = {**components, **sample}
        if generator is not None:
            kwargs["generator"] = generator
        return self.processor_fn[key](**kwargs)


class PrecomputedDistributedDataPreprocessor(InMemoryDistributedDataPreprocessor):
    pass


class ResolutionSampler:
    def __init__(self, *, batch_size: int, dim_keys: Dict[str, Tuple[int, ...]]) -> None:
        self.batch_size = max(int(batch_size), 1)
        self.dim_keys = dim_keys
        self._buckets: Dict[Tuple[Tuple[str, Tuple[int, ...]], ...], List[Tuple[Dict[str, Any], ...]]] = {}

    def consume(self, *items: Dict[str, Any]) -> None:
        if not items:
            return
        key = self._resolution_key(items)
        self._buckets.setdefault(key, []).append(tuple(items))

    @property
    def is_ready(self) -> bool:
        return any(len(bucket) >= self.batch_size for bucket in self._buckets.values())

    def get_batch(self):
        for key, bucket in list(self._buckets.items()):
            if len(bucket) >= self.batch_size:
                batch_items = bucket[: self.batch_size]
                del bucket[: self.batch_size]
                if not bucket:
                    self._buckets.pop(key, None)
                streams = list(zip(*batch_items))
                return tuple([list(stream) for stream in streams])
        raise RuntimeError("ResolutionSampler.get_batch() called before a batch was ready.")

    def _resolution_key(self, items: Sequence[Dict[str, Any]]) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        key_parts: List[Tuple[str, Tuple[int, ...]]] = []
        merged: Dict[str, Any] = {}
        for item in items:
            merged.update(item)
        for tensor_key, dims in sorted(self.dim_keys.items()):
            value = merged.get(tensor_key)
            if isinstance(value, torch.Tensor):
                key_parts.append((tensor_key, tuple(int(value.shape[dim]) for dim in dims)))
        return tuple(key_parts)


class ValidationDataset(IterableDataset):
    def __init__(self, dataset_file: str) -> None:
        self.dataset_file = dataset_file

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.dataset_file, newline="") as handle:
            for row in csv.DictReader(handle):
                item = {"caption": row.get("caption") or row.get("prompt") or ""}
                if row.get("image_path"):
                    item["image"] = _load_image(Path(row["image_path"]))
                if row.get("video_path"):
                    item["video"] = _load_video(Path(row["video_path"]))
                if row.get("export_fps"):
                    item["export_fps"] = int(float(row["export_fps"]))
                yield item


def initialize_dataset(
    dataset_name_or_root: str,
    dataset_type: str,
    *,
    streaming: bool = True,
    infinite: bool = True,
    model_name: Optional[str] = None,
    enable_precomputation: bool = False,
    precompute_root: Optional[str] = None,
    _caption_options: Optional[Dict[str, Any]] = None,
    _data_root: Optional[str] = None,
) -> RawDataset:
    del model_name, enable_precomputation, precompute_root
    normalized_type = "video" if dataset_type == "openvid" else dataset_type
    data = _load_metadata_dataset(dataset_name_or_root, streaming=streaming)
    data_root = _data_root
    if data_root is None and os.path.isfile(dataset_name_or_root):
        data_root = str(Path(dataset_name_or_root).parent)
    return RawDataset(
        data,
        dataset_type=normalized_type,
        data_root=data_root,
        infinite=infinite,
        caption_options=_caption_options,
    )


def wrap_iterable_dataset_for_preprocessing(
    dataset: RawDataset,
    dataset_type: str,
    config: Dict[str, Any],
) -> PreprocessedDataset:
    normalized_type = "video" if dataset_type == "openvid" else dataset_type
    return PreprocessedDataset(dataset, normalized_type, config)


def combine_datasets(
    datasets: Sequence[IterableDataset],
    *,
    buffer_size: int = 1,
    shuffle: bool = True,
) -> IterableDataset:
    if len(datasets) == 1 and (not shuffle or buffer_size <= 1):
        return datasets[0]
    return CombinedIterableDataset(datasets, buffer_size=buffer_size, shuffle=shuffle)


def initialize_preprocessor(
    *,
    rank: int,
    num_items: int,
    processor_fn: Dict[str, Any],
    save_dir: Optional[str],
    enable_precomputation: bool,
    hash_save: bool,
    model_name: Optional[str],
):
    cls = PrecomputedDistributedDataPreprocessor if enable_precomputation else InMemoryDistributedDataPreprocessor
    if enable_precomputation:
        logger.warning("Disk precomputation is not implemented in the open-source data module; using in-memory preprocessing.")
    return cls(
        rank=rank,
        num_items=num_items,
        processor_fn=processor_fn,
        save_dir=save_dir,
        enable_precomputation=enable_precomputation,
        hash_save=hash_save,
        model_name=model_name,
    )


def _load_metadata_dataset(dataset_name_or_root: str, *, streaming: bool):
    path = Path(dataset_name_or_root)
    if path.exists() and path.is_file():
        ext = path.suffix.lower()
        if ext == ".csv":
            loader = "csv"
        elif ext in {".json", ".jsonl"}:
            loader = "json"
        elif ext == ".parquet":
            loader = "parquet"
        else:
            raise ValueError(f"Unsupported dataset metadata extension: {path.suffix}")
        return load_dataset(loader, data_files=str(path), split="train", streaming=streaming)
    return load_dataset(dataset_name_or_root, split="train", streaming=streaming)


def _round_robin(iterators: Sequence[Iterator[Dict[str, Any]]]) -> Iterator[Dict[str, Any]]:
    active = list(iterators)
    while active:
        next_active = []
        for iterator in active:
            try:
                yield next(iterator)
                next_active.append(iterator)
            except StopIteration:
                continue
        active = next_active


def _extract_caption(
    row: Dict[str, Any],
    config: Dict[str, Any],
    caption_options: Dict[str, Any],
) -> str:
    caption_key = (
        caption_options.get("caption_column")
        or config.get("caption_column")
        or next((key for key in ("caption", "text", "prompt", "short_caption") if row.get(key) is not None), None)
    )
    if caption_key is None:
        raise ValueError("No caption column found in dataset row.")
    caption = row.get(caption_key)
    if isinstance(caption, (list, tuple)):
        caption = caption[0] if caption else ""
    caption = str(caption or "")

    if config.get("remove_common_llm_caption_prefixes") or caption_options.get("remove_common_llm_caption_prefixes"):
        caption = remove_prefix(caption, list(COMMON_LLM_START_PHRASES))

    id_token = config.get("id_token") or caption_options.get("id_token")
    if id_token:
        caption = f"{id_token} {caption}".strip()
    return caption


def _resolve_media_path(row: Dict[str, Any], data_root: Optional[str], *, media_type: str) -> Path:
    if media_type == "video":
        keys = ("video", "video_path", "videopath", "path", "file", "filename", "mp4")
        extensions = SUPPORTED_VIDEO_FILE_EXTENSIONS
    else:
        keys = ("image", "image_path", "path", "file", "filename")
        extensions = SUPPORTED_IMAGE_FILE_EXTENSIONS

    value = None
    for key in keys:
        candidate = row.get(key)
        if candidate:
            value = candidate
            break
    if isinstance(value, dict):
        value = value.get("path") or value.get("bytes")
    if value is None:
        raise ValueError(f"No {media_type} path column found in dataset row.")

    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    if data_root is not None:
        rooted = Path(data_root) / path
        if rooted.exists():
            return rooted

    suffix = path.suffix.lower().lstrip(".")
    if suffix and suffix not in extensions:
        raise ValueError(f"Unsupported {media_type} extension: {path.suffix}")
    raise FileNotFoundError(f"{media_type.capitalize()} file not found: {path}")


def _load_video(path: Path) -> torch.Tensor:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError("decord is required for video training datasets.") from exc

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    if len(reader) == 0:
        raise ValueError(f"Video has zero frames: {path}")
    frames = reader.get_batch(list(range(len(reader)))).asnumpy()
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    video = video / 127.5 - 1.0
    return video.contiguous()


def _load_image(path: Path) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    array = np.asarray(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    return (tensor / 127.5 - 1.0).contiguous()


def _process_video(video: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    buckets = config.get("video_resolution_buckets")
    if not buckets:
        return video
    buckets = [tuple(int(v) for v in bucket) for bucket in buckets]
    min_frames = min(bucket[0] for bucket in buckets)
    short_video_handling = str(config.get("short_video_handling", "skip")).lower()
    if video.shape[0] < min_frames and short_video_handling == "skip":
        raise _SkipSample()

    mode = config.get("reshape_mode", "bicubic")
    if mode == "random_crop":
        bucket = _nearest_video_bucket(video, buckets)
        video = _resize_random_crop_video(video, bucket)
    else:
        result = resize_to_nearest_bucket_video(video, buckets, resize_mode=mode)
        video = result[0] if isinstance(result, tuple) else result
    return video.contiguous()


def _process_image(image: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    buckets = config.get("image_resolution_buckets")
    if not buckets:
        return image
    buckets = [tuple(int(v) for v in bucket) for bucket in buckets]
    _, height, width = image.shape
    aspect = width / height
    target_h, target_w = min(buckets, key=lambda bucket: abs((bucket[1] / bucket[0]) - aspect))
    image = image.unsqueeze(0)
    image = F.interpolate(image, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return image.squeeze(0).contiguous()


def _nearest_video_bucket(video: torch.Tensor, buckets: Sequence[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    num_frames, _, height, width = video.shape
    possible = [bucket for bucket in buckets if bucket[0] <= num_frames]
    frame_match = max(possible, key=lambda bucket: bucket[0]) if possible else min(buckets, key=lambda b: abs(b[0] - num_frames))
    same_frames = [bucket for bucket in buckets if bucket[0] == frame_match[0]]
    aspect = width / height
    return min(same_frames, key=lambda bucket: abs((bucket[2] / bucket[1]) - aspect))


def _resize_random_crop_video(video: torch.Tensor, bucket: Tuple[int, int, int]) -> torch.Tensor:
    target_frames, target_h, target_w = bucket
    if video.shape[0] > target_frames:
        indices = torch.linspace(0, video.shape[0] - 1, target_frames).long()
        video = video[indices]
    elif video.shape[0] < target_frames:
        indices = torch.arange(target_frames, device=video.device) % video.shape[0]
        video = video[indices].contiguous()

    _, _, height, width = video.shape
    scale = max(target_h / height, target_w / width)
    resized_h, resized_w = int(height * scale), int(width * scale)
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top = random.randint(0, max(resized_h - target_h, 0))
    left = random.randint(0, max(resized_w - target_w, 0))
    return video[:, :, top : top + target_h, left : left + target_w].contiguous()
