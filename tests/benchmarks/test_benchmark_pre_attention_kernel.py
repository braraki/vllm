# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from benchmarks.kernels.benchmark_pre_attention_kernel import (
    _resolve_signatures,
    benchmark_provider,
    validate_outputs,
    write_csv,
)


def test_resolve_signatures_prefers_explicit_values():
    assert _resolve_signatures("ignored-model", [256, 512], 8, 1) == [
        (256, 8, 1),
        (512, 8, 1),
    ]


def test_write_csv_smoke(tmp_path: Path):
    output_path = tmp_path / "pre_attention_kernel_benchmark.csv"
    write_csv(
        [
            {
                "provider": "pre_attention_kernel_custom_op",
                "num_tokens": 1,
                "head_dim": 256,
                "num_heads": 8,
                "num_kv_heads": 1,
                "dtype": "bfloat16",
                "median_ms": 0.1,
                "min_ms": 0.09,
                "max_ms": 0.11,
            }
        ],
        output_path,
    )
    assert output_path.exists()
    assert "pre_attention_kernel_custom_op" in output_path.read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark smoke only")
def test_benchmark_provider_pre_attention_kernel_smoke():
    if not hasattr(torch.ops, "vllm") or not hasattr(
        torch.ops.vllm, "fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update"
    ):
        pytest.skip("pre-attention-kernel custom op not available")

    median_ms, min_ms, max_ms = benchmark_provider(
        provider="pre_attention_kernel_custom_op",
        num_tokens=1,
        head_dim=256,
        num_heads=8,
        num_kv_heads=1,
        dtype=torch.bfloat16,
        eps=1e-6,
    )
    assert median_ms >= 0.0
    assert min_ms >= 0.0
    assert max_ms >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness only")
def test_validate_outputs_smoke():
    if not hasattr(torch.ops, "vllm") or not hasattr(
        torch.ops.vllm, "fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update"
    ):
        pytest.skip("pre-attention-kernel custom op not available")

    validate_outputs(
        head_dim=256,
        num_heads=8,
        num_kv_heads=1,
        dtype=torch.bfloat16,
        eps=1e-6,
        num_tokens=4,
    )
