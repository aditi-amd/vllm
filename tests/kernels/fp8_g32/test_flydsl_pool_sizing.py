# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.attention.ops import (
    flydsl_fp8_g32_decode_v4,
    flydsl_fp8_g32_decode_v5_fused,
)


def _clear_env_snapshot(module):
    snapshot = getattr(module, "_ENV_SNAPSHOT", None)
    if snapshot is not None:
        snapshot.clear()


@pytest.mark.parametrize(
    "module",
    [flydsl_fp8_g32_decode_v4, flydsl_fp8_g32_decode_v5_fused],
)
@pytest.mark.parametrize(
    ("num_speculative_tokens", "expected"),
    [
        (None, 512),
        (2, 768),
    ],
)
def test_pool_sizing_counts_capture_tokens_and_scheduler_requests(
    monkeypatch, module, num_speculative_tokens, expected
):
    speculative_config = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 512]),
        scheduler_config=SimpleNamespace(max_num_seqs=256),
        speculative_config=speculative_config,
    )

    monkeypatch.delenv("VLLM_FP8_G32_DECODE_V4_B_BUCKET", raising=False)
    _clear_env_snapshot(module)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: config,
    )

    # Capture sizes already count token rows. Only max_num_seqs (requests)
    # expands by K = 1 + num_speculative_tokens.
    assert module._detect_max_capture_B() == expected


@pytest.mark.parametrize(
    "module",
    [flydsl_fp8_g32_decode_v4, flydsl_fp8_g32_decode_v5_fused],
)
def test_pool_sizing_honors_explicit_bucket(monkeypatch, module):
    monkeypatch.setenv("VLLM_FP8_G32_DECODE_V4_B_BUCKET", "123")
    _clear_env_snapshot(module)

    assert module._detect_max_capture_B() == 123
