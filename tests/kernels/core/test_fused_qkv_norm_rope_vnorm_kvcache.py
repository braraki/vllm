# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name, set_random_seed
from vllm.v1.attention.backends.registry import AttentionBackendEnum

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
NUM_TOKENS = [1, 4, 16, 256]
HEAD_DIMS = [256, 512]


def _allocate_kv_cache(
    attn: Attention,
    block_size: int,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    max_blocks = (num_tokens + block_size - 1) // block_size
    kv_cache_shape = attn.attn_backend.get_kv_cache_shape(
        max_blocks,
        block_size,
        attn.num_kv_heads,
        attn.head_size,
    )
    try:
        stride_order = attn.attn_backend.get_kv_cache_stride_order()
    except (AttributeError, NotImplementedError):
        stride_order = tuple(range(len(kv_cache_shape)))

    permuted_shape = tuple(kv_cache_shape[i] for i in stride_order)
    inv_order = [stride_order.index(i) for i in range(len(kv_cache_shape))]
    raw_tensor = torch.zeros(
        int(torch.tensor(kv_cache_shape).prod().item()),
        dtype=dtype,
        device=device,
    )
    raw_tensor = raw_tensor.view(permuted_shape)
    return raw_tensor.permute(*inv_order)


def _baseline_post_gemm(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(q.shape[0], num_heads, head_dim)
    q = RMSNorm.forward_static(
        q_by_head,
        eps,
        head_dim,
        q.dtype,
        q_weight,
    ).view(q.shape)

    k_by_head = k.view(k.shape[0], num_kv_heads, head_dim)
    k = RMSNorm.forward_static(
        k_by_head,
        eps,
        head_dim,
        k.dtype,
        k_weight,
    ).view(k.shape)

    q, k = RotaryEmbedding.forward_static(
        positions,
        q,
        k,
        head_dim,
        head_dim,
        cos_sin_cache,
        is_neox,
    )

    v_by_head = v.view(v.shape[0], num_kv_heads, head_dim)
    v = RMSNorm.forward_static(
        v_by_head,
        eps,
        head_dim,
        v.dtype,
        None,
    ).view(v.shape)

    q_heads = q.view(-1, num_heads, head_dim)
    k_heads = k.view(-1, num_kv_heads, head_dim)
    v_heads = v.view(-1, num_kv_heads, head_dim)
    torch.ops.vllm.unified_kv_cache_update(
        k_heads,
        v_heads,
        _encode_layer_name(layer_name),
    )
    return q_heads, k_heads, v_heads


def _fused_post_gemm(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = qkv.clone()
    ops.fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update(
        qkv,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        positions.view(-1),
        _encode_layer_name(layer_name),
        -1,
    )
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return (
        q.view(-1, num_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        v.view(-1, num_kv_heads, head_dim),
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Part 4 fused post-GEMM op requires CUDA",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@torch.inference_mode()
def test_fused_qkv_norm_rope_vnorm_kvcache_matches_reference(
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    num_tokens: int,
    head_dim: int,
):
    if not hasattr(torch.ops, "vllm") or not hasattr(
        torch.ops.vllm, "fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update"
    ):
        pytest.skip("Part 4 fused post-GEMM custom op not available")

    set_random_seed(7)
    num_heads = 8
    num_kv_heads = 1
    device = torch.device("cuda")
    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
    )

    with set_current_vllm_config(vllm_config):
        attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / head_dim**0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
            attn_backend=AttentionBackendEnum.FLASH_ATTN.get_class(),
        )

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    baseline_cache = _allocate_kv_cache(
        attn,
        block_size=vllm_config.cache_config.block_size,
        num_tokens=num_tokens,
        dtype=dtype,
        device=device,
    )
    fused_cache = baseline_cache.clone()

    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config,
        slot_mapping={attn.layer_name: slot_mapping},
    ):
        attn.kv_cache = baseline_cache
        baseline_q, baseline_k, baseline_v = _baseline_post_gemm(
            qkv,
            positions,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            eps,
            num_heads,
            num_kv_heads,
            head_dim,
            is_neox,
            attn.layer_name,
        )

        attn.kv_cache = fused_cache
        fused_q, fused_k, fused_v = _fused_post_gemm(
            qkv,
            positions,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            eps,
            num_heads,
            num_kv_heads,
            head_dim,
            is_neox,
            attn.layer_name,
        )

    atol, rtol = ((2e-3, 2e-3) if dtype == torch.float16 else (2e-2, 1e-2))
    torch.testing.assert_close(baseline_q, fused_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(baseline_k, fused_k, atol=atol, rtol=rtol)
    torch.testing.assert_close(baseline_v, fused_v, atol=atol, rtol=rtol)
    torch.testing.assert_close(baseline_cache, fused_cache, atol=atol, rtol=rtol)
