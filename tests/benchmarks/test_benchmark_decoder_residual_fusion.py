# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from benchmarks.kernels.benchmark_decoder_residual_fusion import (
    _resolve_hidden_sizes,
    benchmark_provider,
    validate_outputs,
    write_csv,
)


def test_resolve_hidden_sizes_prefers_explicit_values():
    assert _resolve_hidden_sizes("ignored-model", [2304, 4096]) == [2304, 4096]


def test_write_csv_smoke(tmp_path: Path):
    output_path = tmp_path / "decoder_residual_fusion_benchmark.csv"
    write_csv(
        [
            {
                "provider": "fusion_custom_op",
                "num_tokens": 1,
                "hidden_size": 2304,
                "dtype": "bfloat16",
                "median_ms": 0.1,
                "min_ms": 0.09,
                "max_ms": 0.11,
            }
        ],
        output_path,
    )
    assert output_path.exists()
    assert "fusion_custom_op" in output_path.read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark smoke only")
def test_benchmark_provider_fusion_custom_op_smoke():
    if not hasattr(torch.ops._C, "fused_add_rms_norm"):
        pytest.skip("fused_add_rms_norm custom op not available")

    median_ms, min_ms, max_ms = benchmark_provider(
        provider="fusion_custom_op",
        num_tokens=1,
        hidden_size=64,
        dtype=torch.bfloat16,
        eps=1e-6,
    )
    assert median_ms >= 0.0
    assert min_ms >= 0.0
    assert max_ms >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness only")
def test_validate_outputs_smoke():
    if not hasattr(torch.ops._C, "fused_add_rms_norm"):
        pytest.skip("fused_add_rms_norm custom op not available")

    validate_outputs(hidden_size=64, dtype=torch.bfloat16, eps=1e-6, num_tokens=4)
