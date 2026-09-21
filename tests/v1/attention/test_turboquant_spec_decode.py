# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.attention.backends.turboquant_attn import (
    _uniform_decode_query_len,
)


@pytest.mark.parametrize(
    ("num_decodes", "num_decode_tokens", "expected"),
    [
        (0, 0, 1),
        (4, 4, 1),
        (4, 8, 2),
        (4, 12, 3),
    ],
)
def test_uniform_decode_query_len(num_decodes, num_decode_tokens, expected):
    assert _uniform_decode_query_len(num_decodes, num_decode_tokens) == expected


def test_uniform_decode_query_len_rejects_ragged_batch():
    with pytest.raises(RuntimeError, match="ragged 1-and-K mix"):
        _uniform_decode_query_len(num_decodes=3, num_decode_tokens=7)
