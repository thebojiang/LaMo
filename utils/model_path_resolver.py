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

"""Resolve user-provided model paths for the open-source release."""

try:
    from finetrainers import get_logger
    _log = get_logger()
except ImportError:
    import logging
    _log = logging.getLogger("lamo.model_path_resolver")


def resolve_pretrained_model_path(
    pretrained_model_name_or_path: str,
    cache_dir: str | None = None,
) -> str:
    """Return a local path or public model id unchanged.

    ``cache_dir`` is accepted for compatibility with standard model-loading
    call sites but is not needed by this resolver.
    """
    path = (pretrained_model_name_or_path or "").strip()
    _log.info("[resolve_pretrained_model_path] Using path as provided: %s", path)
    return path
