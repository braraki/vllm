# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_SUPPORTED_HEAD_DIMS = {256, 512}
_SUPPORTED_NUM_HEADS_Q = 8
_SUPPORTED_NUM_HEADS_KV = 1


def _raise_if_unsupported(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
) -> None:
    if not HAS_TRITON:
        raise RuntimeError(
            "fused_qkv_norm_rope_vnorm requires Triton, but Triton is not available"
        )
    if not current_platform.is_cuda():
        raise RuntimeError("fused_qkv_norm_rope_vnorm Triton path is CUDA-only")
    if not qkv.is_cuda:
        raise RuntimeError("qkv must be a CUDA tensor")
    if not q_weight.is_cuda or not k_weight.is_cuda:
        raise RuntimeError("q_weight and k_weight must be CUDA tensors")
    if not cos_sin_cache.is_cuda or not position_ids.is_cuda:
        raise RuntimeError("cos_sin_cache and position_ids must be CUDA tensors")
    if (
        q_weight.device != qkv.device
        or k_weight.device != qkv.device
        or cos_sin_cache.device != qkv.device
        or position_ids.device != qkv.device
    ):
        raise RuntimeError("All fused_qkv_norm_rope_vnorm inputs must share a device")
    if qkv.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"Unsupported dtype for fused_qkv_norm_rope_vnorm: {qkv.dtype}"
        )
    if qkv.dim() != 2:
        raise RuntimeError("qkv must be a 2D packed tensor")
    if not qkv.is_contiguous():
        raise RuntimeError("qkv must be contiguous")
    if position_ids.dim() != 1 or not position_ids.is_contiguous():
        raise RuntimeError("position_ids must be a contiguous 1D tensor")
    if position_ids.dtype != torch.long:
        raise RuntimeError("position_ids must be torch.int64")
    if q_weight.dim() != 1 or k_weight.dim() != 1:
        raise RuntimeError("q_weight and k_weight must be 1D tensors")
    if not q_weight.is_contiguous() or not k_weight.is_contiguous():
        raise RuntimeError("q_weight and k_weight must be contiguous")
    if q_weight.dtype != qkv.dtype or k_weight.dtype != qkv.dtype:
        raise RuntimeError("q_weight and k_weight must match qkv dtype")
    if cos_sin_cache.dim() != 2 or not cos_sin_cache.is_contiguous():
        raise RuntimeError("cos_sin_cache must be a contiguous 2D tensor")
    if cos_sin_cache.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError("cos_sin_cache must be float32, float16, or bfloat16")
    if cos_sin_cache.size(1) != head_dim:
        raise RuntimeError(
            "Gemma4 Triton fused_qkv_norm_rope_vnorm requires rotary_dim == head_dim"
        )
    if num_heads_q != _SUPPORTED_NUM_HEADS_Q:
        raise RuntimeError(
            "Gemma4 Triton fused_qkv_norm_rope_vnorm requires num_heads_q == 8"
        )
    if (
        num_heads_k != _SUPPORTED_NUM_HEADS_KV
        or num_heads_v != _SUPPORTED_NUM_HEADS_KV
    ):
        raise RuntimeError(
            "Gemma4 Triton fused_qkv_norm_rope_vnorm requires num_heads_k == num_heads_v == 1"
        )
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise RuntimeError(
            f"Unsupported head_dim for Triton fused_qkv_norm_rope_vnorm: {head_dim}"
        )
    if q_weight.numel() != head_dim or k_weight.numel() != head_dim:
        raise RuntimeError("q_weight and k_weight sizes must match head_dim")
    total_dim = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    if qkv.size(1) != total_dim:
        raise RuntimeError("qkv packed width does not match heads * head_dim")
    if position_ids.numel() != qkv.size(0):
        raise RuntimeError("position_ids length must match qkv rows")


@triton.jit
def _fused_qkv_norm_rope_vnorm_neox_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    position_ids_ptr,
    stride_qkv_row,
    stride_cache_row,
    q_size,
    k_offset,
    head_dim,
    eps,
    HALF_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.program_id(axis=1)

    row_ptr = qkv_ptr + token_idx * stride_qkv_row
    is_q = slot_idx < 8
    is_k = slot_idx == 8
    head_offset = tl.where(is_q, slot_idx * head_dim, tl.where(is_k, q_size, k_offset))
    base_ptr = row_ptr + head_offset

    half_offsets = tl.arange(0, HALF_BLOCK)
    first_ptrs = base_ptr + half_offsets
    second_ptrs = first_ptrs + HALF_BLOCK

    first = tl.load(first_ptrs).to(tl.float32)
    second = tl.load(second_ptrs).to(tl.float32)
    sum_sq = tl.sum(first * first + second * second, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / head_dim + eps)

    q_weight_first = tl.load(q_weight_ptr + half_offsets).to(tl.float32)
    q_weight_second = tl.load(q_weight_ptr + HALF_BLOCK + half_offsets).to(
        tl.float32
    )
    k_weight_first = tl.load(k_weight_ptr + half_offsets).to(tl.float32)
    k_weight_second = tl.load(k_weight_ptr + HALF_BLOCK + half_offsets).to(
        tl.float32
    )

    weight_first = tl.where(
        is_q,
        q_weight_first,
        tl.where(is_k, k_weight_first, 1.0),
    )
    weight_second = tl.where(
        is_q,
        q_weight_second,
        tl.where(is_k, k_weight_second, 1.0),
    )

    norm_first = first * inv_rms * weight_first
    norm_second = second * inv_rms * weight_second

    pos_idx = tl.load(position_ids_ptr + token_idx)
    cache_ptr = cos_sin_cache_ptr + pos_idx * stride_cache_row
    cos = tl.load(cache_ptr + half_offsets).to(tl.float32)
    sin = tl.load(cache_ptr + HALF_BLOCK + half_offsets).to(tl.float32)

    rope_first = norm_first * cos - norm_second * sin
    rope_second = norm_second * cos + norm_first * sin
    qk_mask = slot_idx < 9

    tl.store(
        first_ptrs,
        tl.where(qk_mask, rope_first, norm_first),
    )
    tl.store(
        second_ptrs,
        tl.where(qk_mask, rope_second, norm_second),
    )


@triton.jit
def _fused_qkv_norm_rope_vnorm_gptj_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    position_ids_ptr,
    stride_qkv_row,
    stride_cache_row,
    q_size,
    k_offset,
    head_dim,
    eps,
    HALF_BLOCK: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.program_id(axis=1)

    row_ptr = qkv_ptr + token_idx * stride_qkv_row
    is_q = slot_idx < 8
    is_k = slot_idx == 8
    head_offset = tl.where(is_q, slot_idx * head_dim, tl.where(is_k, q_size, k_offset))
    base_ptr = row_ptr + head_offset

    pair_offsets = tl.arange(0, HALF_BLOCK)
    even_ptrs = base_ptr + pair_offsets * 2
    odd_ptrs = even_ptrs + 1

    even = tl.load(even_ptrs).to(tl.float32)
    odd = tl.load(odd_ptrs).to(tl.float32)
    sum_sq = tl.sum(even * even + odd * odd, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / head_dim + eps)

    even_cols = pair_offsets * 2
    odd_cols = even_cols + 1
    q_weight_even = tl.load(q_weight_ptr + even_cols).to(tl.float32)
    q_weight_odd = tl.load(q_weight_ptr + odd_cols).to(tl.float32)
    k_weight_even = tl.load(k_weight_ptr + even_cols).to(tl.float32)
    k_weight_odd = tl.load(k_weight_ptr + odd_cols).to(tl.float32)

    weight_even = tl.where(is_q, q_weight_even, tl.where(is_k, k_weight_even, 1.0))
    weight_odd = tl.where(is_q, q_weight_odd, tl.where(is_k, k_weight_odd, 1.0))

    norm_even = even * inv_rms * weight_even
    norm_odd = odd * inv_rms * weight_odd

    pos_idx = tl.load(position_ids_ptr + token_idx)
    cache_ptr = cos_sin_cache_ptr + pos_idx * stride_cache_row
    cos = tl.load(cache_ptr + pair_offsets).to(tl.float32)
    sin = tl.load(cache_ptr + HALF_BLOCK + pair_offsets).to(tl.float32)

    rope_even = norm_even * cos - norm_odd * sin
    rope_odd = norm_odd * cos + norm_even * sin
    qk_mask = slot_idx < 9

    tl.store(
        even_ptrs,
        tl.where(qk_mask, rope_even, norm_even),
    )
    tl.store(
        odd_ptrs,
        tl.where(qk_mask, rope_odd, norm_odd),
    )


def _fused_qkv_norm_rope_vnorm_impl(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    del forced_token_heads_per_warp
    _raise_if_unsupported(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_weight,
        k_weight,
        cos_sin_cache,
        position_ids,
    )

    num_tokens = qkv.size(0)
    total_heads = num_heads_q + num_heads_k + num_heads_v
    q_size = num_heads_q * head_dim
    k_offset = q_size + num_heads_k * head_dim
    grid = (num_tokens, total_heads)
    num_warps = 4 if head_dim == 256 else 8

    if is_neox:
        _fused_qkv_norm_rope_vnorm_neox_kernel[grid](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            position_ids,
            qkv.stride(0),
            cos_sin_cache.stride(0),
            q_size,
            k_offset,
            head_dim,
            eps,
            HALF_BLOCK=head_dim // 2,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _fused_qkv_norm_rope_vnorm_gptj_kernel[grid](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            position_ids,
            qkv.stride(0),
            cos_sin_cache.stride(0),
            q_size,
            k_offset,
            head_dim,
            eps,
            HALF_BLOCK=head_dim // 2,
            num_warps=num_warps,
            num_stages=1,
        )


def _fused_qkv_norm_rope_vnorm_fake(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    del (
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
        forced_token_heads_per_warp,
    )
    return None


direct_register_custom_op(
    op_name="fused_qkv_norm_rope_vnorm",
    op_func=_fused_qkv_norm_rope_vnorm_impl,
    mutates_args=["qkv"],
    fake_impl=_fused_qkv_norm_rope_vnorm_fake,
)
