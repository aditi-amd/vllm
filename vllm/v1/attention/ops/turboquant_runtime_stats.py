# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant runtime statistics collection (opt-in telemetry).

These functions are called at key points in the store/decode/builder
pipeline. By default they are no-ops. Set VLLM_TQ_RUNTIME_STATS=1
to enable lightweight in-process stat collection for profiling.
"""

import os
from typing import Any


_ENABLED = os.environ.get("VLLM_TQ_RUNTIME_STATS", "0") == "1"

# Accumulator dicts - only populated when _ENABLED
_decode_stats: list[dict[str, Any]] = []
_store_stats: list[dict[str, Any]] = []
_builder_stats: list[dict[str, Any]] = []


def record_decode_call(**kwargs: Any) -> None:
    """Record a decode kernel invocation."""
    if _ENABLED:
        _decode_stats.append(kwargs)


def record_store_call(**kwargs: Any) -> None:
    """Record a KV store kernel invocation."""
    if _ENABLED:
        _store_stats.append(kwargs)


def record_builder_step(**kwargs: Any) -> None:
    """Record a metadata builder step."""
    if _ENABLED:
        _builder_stats.append(kwargs)


def get_stats() -> dict[str, list[dict[str, Any]]]:
    """Return all collected stats (for debugging)."""
    return {
        "decode": list(_decode_stats),
        "store": list(_store_stats),
        "builder": list(_builder_stats),
    }


def reset_stats() -> None:
    """Clear all accumulated stats."""
    _decode_stats.clear()
    _store_stats.clear()
    _builder_stats.clear()
