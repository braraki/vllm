# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.config import CompilationConfig, CompilationMode, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import PLEGeluAndMul
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
from vllm.platforms import current_platform


class _Gemma4PLEPathModule(nn.Module):
    def __init__(self, use_fusion: bool):
        super().__init__()
        layer = object.__new__(Gemma4DecoderLayer)
        nn.Module.__init__(layer)
        layer.use_ple_gelu_and_mul_fusion = use_fusion
        layer.ple_gelu_and_mul = PLEGeluAndMul()
        self.layer = layer

    def forward(self, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.layer._apply_ple_gelu_and_mul(gate, value)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Gemma4 PLE fusion test requires a CUDA-like backend",
)
def test_gemma4_ple_gelu_and_mul_matches_baseline(default_vllm_config, compiled, dtype):
    if not hasattr(torch.ops._C, "ple_gelu_tanh_and_mul"):
        pytest.skip("ple_gelu_tanh_and_mul custom op not available")

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_tokens = 11
    ple_dim = 256
    gate = torch.randn(num_tokens, ple_dim, dtype=dtype)
    value = torch.randn_like(gate)

    if compiled:
        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                custom_ops=["+ple_gelu_and_mul"],
            )
        )
        context = set_current_vllm_config(vllm_config)
    else:
        context = set_current_vllm_config(VllmConfig())

    with context:
        baseline = _Gemma4PLEPathModule(use_fusion=False)
        fused = _Gemma4PLEPathModule(use_fusion=True)
        if compiled:
            baseline = torch.compile(baseline)
            fused = torch.compile(fused)

        baseline_out = baseline(gate.clone(), value.clone())
        fused_out = fused(gate.clone(), value.clone())

    torch.testing.assert_close(
        baseline_out,
        fused_out,
        atol=get_default_atol(baseline_out),
        rtol=get_default_rtol(baseline_out),
    )
