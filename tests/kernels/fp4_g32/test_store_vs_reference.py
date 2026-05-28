# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Triton FP4-g32 store kernel.

Compares the Triton store output (codes + scales) against the PyTorch
reference (`reference.fp4_g32_encode`) byte/bit exactly. The reference
itself is validated separately (`test_reference.py`); this test ensures
the Triton kernel reproduces the reference behavior exactly.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops.fp4_g32.fp4_levels import (
    GROUP_SIZE,
    k_codes_offset,
    k_scales_offset,
    n_groups,
    slot_size,
    v_codes_offset,
    v_scales_offset,
)
from vllm.v1.attention.ops.fp4_g32.reference import fp4_g32_encode
from vllm.v1.attention.ops.fp4_g32.triton_store import fp4_g32_store


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _make_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate a zeroed uint8 KV cache of the right slot size."""
    slot_b = slot_size(head_dim)
    return torch.zeros(
        num_blocks, block_size, num_kv_heads, slot_b, dtype=torch.uint8, device=device
    )


def _read_codes_and_scales(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
):
    """Pull (k_codes, k_scales, v_codes, v_scales) for each (token, head) from
    the AoS slot layout. Returns tensors shaped [N, H, ...] matching the
    reference output.

    - k_codes / v_codes: [N, H, head_dim // 2] uint8
    - k_scales / v_scales: [N, H, n_groups] fp16
    """
    block_size = kv_cache.shape[1]
    N = slot_mapping.shape[0]
    Gk = n_groups(head_dim)

    k_codes_off = k_codes_offset(head_dim)
    k_scales_off = k_scales_offset(head_dim)
    v_codes_off = v_codes_offset(head_dim)
    v_scales_off = v_scales_offset(head_dim)
    codes_bytes = head_dim // 2

    # Zero-init so that skipped slots (slot_mapping[i] < 0) leave a clean
    # zero row that the test's "did-not-write" branch can assert against.
    k_codes = torch.zeros(
        N, num_kv_heads, codes_bytes, dtype=torch.uint8, device=kv_cache.device
    )
    k_scales = torch.zeros(
        N, num_kv_heads, Gk, dtype=torch.float16, device=kv_cache.device
    )
    v_codes = torch.zeros_like(k_codes)
    v_scales = torch.zeros_like(k_scales)

    for ti in range(N):
        slot = int(slot_mapping[ti].item())
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        for h in range(num_kv_heads):
            slot_bytes = kv_cache[blk, off, h]  # [slot_size_aligned] uint8

            kc = slot_bytes[k_codes_off : k_codes_off + codes_bytes]
            k_codes[ti, h] = kc

            ks_bytes = slot_bytes[k_scales_off : k_scales_off + 2 * Gk]
            k_scales[ti, h] = ks_bytes.view(torch.float16)

            vc = slot_bytes[v_codes_off : v_codes_off + codes_bytes]
            v_codes[ti, h] = vc

            vs_bytes = slot_bytes[v_scales_off : v_scales_off + 2 * Gk]
            v_scales[ti, h] = vs_bytes.view(torch.float16)

    return k_codes, k_scales, v_codes, v_scales


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_tokens", [1, 32, 17])  # include non-multiple
def test_store_matches_reference(dtype, head_dim, num_kv_heads, num_tokens):
    """Triton store output (codes + scales) matches PyTorch reference exactly."""
    device = torch.device("cuda")
    torch.manual_seed(0xF4)

    block_size = 32
    num_blocks = max(2, (num_tokens + block_size - 1) // block_size + 1)

    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Assign sequential slots; leave one negative to test the skip path.
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    if num_tokens >= 4:
        slot_mapping[1] = -1  # this slot must NOT be written

    kv_cache = _make_cache(num_blocks, block_size, num_kv_heads, head_dim, device)

    fp4_g32_store(key, value, kv_cache, slot_mapping)
    torch.cuda.synchronize()

    # Reference encoding — one head at a time to match the kernel's
    # per-(token, head) program model. Reference operates on [..., D].
    ref_k = fp4_g32_encode(key, rotate=True)
    ref_v = fp4_g32_encode(value, rotate=False)

    # Pull back from the cache.
    k_codes, k_scales, v_codes, v_scales = _read_codes_and_scales(
        kv_cache, slot_mapping, num_kv_heads, head_dim
    )

    # If a slot was set negative, verify the physical slot index 1 (where
    # the original `slot_mapping[1]` would have written) is still untouched
    # in the cache — i.e. the kernel correctly took the early-return path.
    if num_tokens >= 4:
        # Original slot_mapping[1] would have been 1 → physical slot at
        # (block=0, off=1). With slot_mapping[1] = -1, that physical slot
        # must remain all-zero (we initialized the cache to zero).
        assert (kv_cache[0, 1, :, :] == 0).all(), (
            "skipped slot's physical cache region was unexpectedly written"
        )

    # ── Compare token-by-token (skipping the negative slot) ────────────
    for ti in range(num_tokens):
        slot = int(slot_mapping[ti].item())
        if slot < 0:
            continue

        # K
        assert torch.equal(k_codes[ti], ref_k.codes_packed[ti]), (
            f"K codes mismatch at token {ti}\n"
            f"got:      {k_codes[ti].cpu().numpy().tolist()[:16]}\n"
            f"expected: {ref_k.codes_packed[ti].cpu().numpy().tolist()[:16]}"
        )
        # Scales must be bit-exact (both fp16).
        got_bits = k_scales[ti].view(torch.uint16)
        exp_bits = ref_k.scales[ti].view(torch.uint16)
        assert torch.equal(got_bits, exp_bits), (
            f"K scales mismatch at token {ti}\n"
            f"got fp16:      {k_scales[ti].cpu().numpy().tolist()}\n"
            f"expected fp16: {ref_k.scales[ti].cpu().numpy().tolist()}"
        )

        # V
        assert torch.equal(v_codes[ti], ref_v.codes_packed[ti]), (
            f"V codes mismatch at token {ti}"
        )
        got_bits = v_scales[ti].view(torch.uint16)
        exp_bits = ref_v.scales[ti].view(torch.uint16)
        assert torch.equal(got_bits, exp_bits), (
            f"V scales mismatch at token {ti}"
        )


def test_zero_input_produces_zero_codes_and_scales():
    """All-zero K/V → all-zero codes, all-zero scales (no NaNs, no garbage)."""
    device = torch.device("cuda")
    D, H, N = 128, 4, 8
    block_size = 16
    num_blocks = 2

    key = torch.zeros(N, H, D, dtype=torch.bfloat16, device=device)
    value = torch.zeros(N, H, D, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.arange(N, dtype=torch.int64, device=device)
    kv_cache = _make_cache(num_blocks, block_size, H, D, device)

    fp4_g32_store(key, value, kv_cache, slot_mapping)
    torch.cuda.synchronize()

    k_codes, k_scales, v_codes, v_scales = _read_codes_and_scales(
        kv_cache, slot_mapping, H, D
    )
    assert (k_codes == 0).all()
    assert (k_scales == 0).all()
    assert (v_codes == 0).all()
    assert (v_scales == 0).all()


def test_constant_c_env_override(monkeypatch):
    """Setting FP4FP16_CONSTANT_C changes the stored scales accordingly."""
    device = torch.device("cuda")
    D, H, N = 128, 2, 4
    block_size = 16
    num_blocks = 1

    torch.manual_seed(7)
    key = torch.randn(N, H, D, dtype=torch.bfloat16, device=device)
    value = torch.randn(N, H, D, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.arange(N, dtype=torch.int64, device=device)

    # Default c = 0.156
    cache_default = _make_cache(num_blocks, block_size, H, D, device)
    fp4_g32_store(key, value, cache_default, slot_mapping)
    torch.cuda.synchronize()
    _, ks_default, _, _ = _read_codes_and_scales(
        cache_default, slot_mapping, H, D
    )

    # Override c via env var (note: the kernel reads `constant_c` at launch
    # time via get_constant_c, so monkeypatching env BEFORE the launch call
    # is sufficient — no kernel cache reset needed).
    monkeypatch.setenv("FP4FP16_CONSTANT_C", "0.200")
    cache_alt = _make_cache(num_blocks, block_size, H, D, device)
    fp4_g32_store(key, value, cache_alt, slot_mapping)
    torch.cuda.synchronize()
    _, ks_alt, _, _ = _read_codes_and_scales(cache_alt, slot_mapping, H, D)

    # Scales should differ (absmax * 0.200 vs absmax * 0.156).
    assert not torch.equal(ks_default, ks_alt), (
        "FP4FP16_CONSTANT_C override had no effect on stored scales"
    )

    # Ratio should be approximately 0.200 / 0.156 ≈ 1.282 for nonzero scales.
    ratio = ks_alt.float() / ks_default.float().clamp_min(1e-8)
    nonzero = ks_default.float() > 0
    if nonzero.any():
        med = float(ratio[nonzero].median().item())
        assert abs(med - 0.200 / 0.156) < 0.02, f"unexpected scale ratio: {med}"
