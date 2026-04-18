# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from benchmarks.kernels.benchmark_ple_gelu_and_mul import (
    _resolve_ple_dims,
    benchmark_provider,
    write_csv,
)


def test_resolve_ple_dims_prefers_explicit_dims():
    assert _resolve_ple_dims("ignored-model", [64, 128]) == [64, 128]


def test_write_csv_smoke(tmp_path: Path):
    output_path = tmp_path / "ple_gelu_and_mul_benchmark.csv"
    write_csv(
        [
            {
                "provider": "custom_two_input",
                "num_tokens": 1,
                "ple_dim": 256,
                "median_ms": 0.1,
                "min_ms": 0.09,
                "max_ms": 0.11,
            }
        ],
        output_path,
    )
    assert output_path.exists()
    assert "custom_two_input" in output_path.read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark smoke only")
def test_benchmark_provider_custom_two_input_smoke():
    if not hasattr(torch.ops._C, "ple_gelu_tanh_and_mul"):
        pytest.skip("ple_gelu_tanh_and_mul custom op not available")

    median_ms, min_ms, max_ms = benchmark_provider(
        provider="custom_two_input",
        num_tokens=1,
        ple_dim=64,
        dtype=torch.bfloat16,
    )
    assert median_ms >= 0.0
    assert min_ms >= 0.0
    assert max_ms >= 0.0
