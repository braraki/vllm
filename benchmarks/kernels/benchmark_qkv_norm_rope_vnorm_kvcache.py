# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Gemma4 full post-GEMM attention-prep + KV-cache fusion.

This benchmark isolates the non-KV-shared decoder block between the QKV GEMM
and FlashAttention:

Baseline:
    q = q_norm(q)
    k = k_norm(k)
    q, k = rotary_emb(positions, q, k)
    v = v_norm(v)
    unified_kv_cache_update(k, v, layer_name)

New kernel:
    fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update(qkv, ..., layer_name)
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    _encode_layer_name,
    set_random_seed,
)

DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_NUM_TOKENS = [1, 4, 16, 64, 256, 1024]
DEFAULT_PROVIDERS = [
    "baseline_compiled",
    "post_gemm_kvcache_custom_op",
]

# Provider glossary:
# - baseline_compiled: the unfused post-GEMM attention-prep + KV-cache block
#   under torch.compile.
# - post_gemm_kvcache_custom_op: the Part 4 fused op that mutates packed qkv
#   and writes K/V directly to unified KV cache.

FUSED_OP_MESSAGE = (
    "fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update Triton custom op "
    "is not available. Ensure local vLLM imports succeeded and Triton is "
    "available on CUDA."
)


def _load_gemma4_attention_signatures(
    model: str,
) -> tuple[list[tuple[int, int, int]], float]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config

    num_heads = int(getattr(config, "num_attention_heads", 0))
    num_kv_heads = int(getattr(config, "num_key_value_heads", 0))
    head_dims = {
        int(getattr(config, "head_dim", 0)),
        int(getattr(config, "global_head_dim", getattr(config, "head_dim", 0))),
    }
    head_dims.discard(0)
    eps = getattr(config, "rms_norm_eps", None)

    if not num_heads or not num_kv_heads or not head_dims or eps is None:
        raise ValueError(
            f"Model {model!r} does not expose Gemma4 attention signatures cleanly"
        )

    signatures = sorted((head_dim, num_heads, num_kv_heads) for head_dim in head_dims)
    return signatures, float(eps)


def _resolve_signatures(
    model: str | None,
    head_dims: list[int] | None,
    num_heads: int | None,
    num_kv_heads: int | None,
) -> list[tuple[int, int, int]]:
    if head_dims:
        resolved_num_heads = num_heads if num_heads is not None else 8
        resolved_num_kv_heads = num_kv_heads if num_kv_heads is not None else 1
        return sorted(
            (head_dim, resolved_num_heads, resolved_num_kv_heads)
            for head_dim in head_dims
        )
    if model:
        return _load_gemma4_attention_signatures(model)[0]
    return [(256, 8, 1), (512, 8, 1)]


def _resolve_eps(model: str | None, eps: float | None) -> float:
    if eps is not None:
        return float(eps)
    if model:
        return _load_gemma4_attention_signatures(model)[1]
    return 1e-6


def _make_rope_cache(
    max_position_embeddings: int,
    head_dim: int,
    dtype: torch.dtype,
    is_neox: bool,
) -> torch.Tensor:
    with default_vllm_config():
        rope = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=10000.0,
            is_neox_style=is_neox,
            dtype=dtype,
        )
    return rope.cos_sin_cache.to(device="cuda", dtype=dtype)


def _make_vllm_config(dtype: torch.dtype, block_size: int = 16) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(
            block_size=block_size,
            cache_dtype="auto",
        ),
    )


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
        kv_cache_stride_order = attn.attn_backend.get_kv_cache_stride_order()
    except (AttributeError, NotImplementedError):
        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

    permuted_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    inv_order = [kv_cache_stride_order.index(i) for i in range(len(kv_cache_shape))]

    raw_tensor = torch.zeros(
        int(torch.tensor(kv_cache_shape).prod().item()),
        dtype=dtype,
        device=device,
    )
    raw_tensor = raw_tensor.view(permuted_shape)
    return raw_tensor.permute(*inv_order)


def baseline_post_gemm(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    dummy = torch.ops.vllm.unified_kv_cache_update(
        k_heads,
        v_heads,
        _encode_layer_name(layer_name),
    )
    return q_heads, k_heads, v_heads, dummy


def post_gemm_kvcache_custom_op(
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
    forced_token_heads_per_warp: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = qkv.clone()
    dummy = ops.fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update(
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
        forced_token_heads_per_warp,
    )
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return (
        q.view(-1, num_heads, head_dim),
        k.view(-1, num_kv_heads, head_dim),
        v.view(-1, num_kv_heads, head_dim),
        dummy,
    )


def _make_case(
    *,
    num_tokens: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    eps: float,
    is_neox: bool,
) -> tuple[VllmConfig, Attention, dict[str, torch.Tensor | str | float | int | bool]]:
    set_random_seed(42)
    torch.set_default_device("cuda")

    vllm_config = _make_vllm_config(dtype)
    with set_current_vllm_config(vllm_config):
        attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / head_dim**0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
        )
        kv_cache = _allocate_kv_cache(
            attn=attn,
            block_size=vllm_config.cache_config.block_size,
            num_tokens=num_tokens,
            dtype=dtype,
            device=torch.device("cuda"),
        )

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device="cuda")
    positions = torch.arange(num_tokens, dtype=torch.long, device="cuda")
    q_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    k_weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    cos_sin_cache = _make_rope_cache(4096, head_dim, dtype, is_neox)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device="cuda")
    attn.kv_cache = kv_cache

    case: dict[str, torch.Tensor | str | float | int | bool] = {
        "qkv": qkv,
        "positions": positions,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos_sin_cache": cos_sin_cache,
        "eps": eps,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "is_neox": is_neox,
        "layer_name": attn.layer_name,
        "slot_mapping": slot_mapping,
    }
    return vllm_config, attn, case


def benchmark_provider(
    provider: str,
    num_tokens: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    eps: float,
    is_neox: bool = True,
) -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_qkv_norm_rope_vnorm_kvcache requires CUDA")

    if not hasattr(torch.ops, "vllm") or not hasattr(
        torch.ops.vllm, "fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update"
    ):
        raise RuntimeError(FUSED_OP_MESSAGE)

    vllm_config, attn, case = _make_case(
        num_tokens=num_tokens,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        eps=eps,
        is_neox=is_neox,
    )
    layer_name = str(case["layer_name"])
    slot_mapping = case["slot_mapping"]

    args = (
        case["qkv"],
        case["positions"],
        case["q_weight"],
        case["k_weight"],
        case["cos_sin_cache"],
        eps,
        num_heads,
        num_kv_heads,
        head_dim,
        is_neox,
        layer_name,
    )

    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config,
        slot_mapping={layer_name: slot_mapping},
    ):
        attn.kv_cache.zero_()
        if provider == "baseline_compiled":
            compiled_fn = torch.compile(baseline_post_gemm)
            fn = lambda: compiled_fn(*args)
        elif provider == "post_gemm_kvcache_compiled":
            compiled_fn = torch.compile(post_gemm_kvcache_custom_op)
            fn = lambda: compiled_fn(*args)
        elif provider == "post_gemm_kvcache_custom_op":
            fn = lambda: post_gemm_kvcache_custom_op(*args)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            fn, quantiles=[0.5, 0.2, 0.8]
        )
    return ms, min_ms, max_ms


def validate_outputs(
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    eps: float,
    is_neox: bool = True,
    num_tokens: int = 8,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("validate_outputs requires CUDA")
    if not hasattr(torch.ops, "vllm") or not hasattr(
        torch.ops.vllm, "fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update"
    ):
        raise RuntimeError(FUSED_OP_MESSAGE)

    vllm_config, attn, case = _make_case(
        num_tokens=num_tokens,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        eps=eps,
        is_neox=is_neox,
    )
    layer_name = str(case["layer_name"])
    slot_mapping = case["slot_mapping"]

    baseline_cache = attn.kv_cache.clone()
    fused_cache = attn.kv_cache.clone()

    with set_current_vllm_config(vllm_config), set_forward_context(
        None,
        vllm_config,
        slot_mapping={layer_name: slot_mapping},
    ):
        attn.kv_cache = baseline_cache
        baseline_q, baseline_k, baseline_v, _ = baseline_post_gemm(
            case["qkv"],
            case["positions"],
            case["q_weight"],
            case["k_weight"],
            case["cos_sin_cache"],
            eps,
            num_heads,
            num_kv_heads,
            head_dim,
            is_neox,
            layer_name,
        )

        attn.kv_cache = fused_cache
        fused_q, fused_k, fused_v, _ = post_gemm_kvcache_custom_op(
            case["qkv"],
            case["positions"],
            case["q_weight"],
            case["k_weight"],
            case["cos_sin_cache"],
            eps,
            num_heads,
            num_kv_heads,
            head_dim,
            is_neox,
            layer_name,
        )

    atol, rtol = ((2e-3, 2e-3) if dtype == torch.float16 else (2e-2, 1e-2))
    torch.testing.assert_close(baseline_q, fused_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(baseline_k, fused_k, atol=atol, rtol=rtol)
    torch.testing.assert_close(baseline_v, fused_v, atol=atol, rtol=rtol)
    torch.testing.assert_close(baseline_cache, fused_cache, atol=atol, rtol=rtol)


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "num_tokens",
                "head_dim",
                "num_heads",
                "num_kv_heads",
                "dtype",
                "median_ms",
                "min_ms",
                "max_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plots(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    head_dims = sorted({int(row["head_dim"]) for row in rows})
    provider_order = [
        "baseline_compiled",
        "post_gemm_kvcache_custom_op",
        "post_gemm_kvcache_compiled",
    ]

    for head_dim in head_dims:
        plt.figure(figsize=(10, 5))
        dim_rows = [row for row in rows if int(row["head_dim"]) == head_dim]
        token_counts = sorted({int(row["num_tokens"]) for row in dim_rows})
        providers = [
            provider
            for provider in provider_order
            if any(row["provider"] == provider for row in dim_rows)
        ]
        if not providers:
            plt.close()
            continue

        grouped_rows = {
            (str(row["provider"]), int(row["num_tokens"])): float(row["median_ms"])
            for row in dim_rows
        }
        group_positions = list(range(len(token_counts)))
        bar_width = 0.8 / len(providers)

        for provider_idx, provider in enumerate(providers):
            x_positions = [
                group_pos - 0.4 + (provider_idx + 0.5) * bar_width
                for group_pos in group_positions
            ]
            latencies_us = [
                grouped_rows[(provider, token_count)] * 1000.0
                for token_count in token_counts
            ]
            bars = plt.bar(
                x_positions,
                latencies_us,
                width=bar_width,
                label=provider,
            )
            plt.bar_label(
                bars,
                labels=[f"{latency_us:.1f}" for latency_us in latencies_us],
                padding=3,
                fontsize=8,
                rotation=0,
            )
        plt.title(f"Post-GEMM Fusion Benchmark (head_dim={head_dim})")
        plt.xlabel("num_tokens")
        plt.ylabel("median latency (us)")
        plt.xticks(group_positions, [str(token_count) for token_count in token_counts])
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"qkv_norm_rope_vnorm_kvcache_head_dim_{head_dim}.png")
        plt.close()


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark Gemma4 full post-GEMM fusion implementations."
    )
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_TOKENS,
        help="List of flattened token counts to benchmark.",
    )
    parser.add_argument(
        "--head-dim",
        nargs="+",
        type=int,
        default=None,
        help="List of attention head dimensions to benchmark.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Override the number of Q heads when --head-dim is provided.",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="Override the number of KV heads when --head-dim is provided.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Optional Hugging Face model id used to derive Gemma4 head_dim "
            "signatures and RMSNorm epsilon when explicit overrides are omitted."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Override the RMSNorm epsilon instead of deriving it from the model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=DEFAULT_PROVIDERS + ["post_gemm_kvcache_compiled"],
        default=DEFAULT_PROVIDERS,
        help="Providers to benchmark.",
    )
    parser.add_argument(
        "--skip-correctness-check",
        action="store_true",
        help="Skip the fused-vs-baseline correctness check before benchmarking.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_qkv_norm_rope_vnorm_kvcache_results"),
        help="Directory that will receive CSV and plot outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    signatures = _resolve_signatures(
        args.model, args.head_dim, args.num_heads, args.num_kv_heads
    )
    eps = _resolve_eps(args.model, args.eps)

    if not args.skip_correctness_check:
        for head_dim, num_heads, num_kv_heads in signatures:
            validate_outputs(
                head_dim=head_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dtype=dtype,
                eps=eps,
            )

    rows: list[dict[str, float | int | str]] = []
    for head_dim, num_heads, num_kv_heads in signatures:
        for num_tokens in args.num_tokens:
            for provider in args.providers:
                median_ms, min_ms, max_ms = benchmark_provider(
                    provider=provider,
                    num_tokens=num_tokens,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    dtype=dtype,
                    eps=eps,
                )
                rows.append(
                    {
                        "provider": provider,
                        "num_tokens": num_tokens,
                        "head_dim": head_dim,
                        "num_heads": num_heads,
                        "num_kv_heads": num_kv_heads,
                        "dtype": args.dtype,
                        "median_ms": median_ms,
                        "min_ms": min_ms,
                        "max_ms": max_ms,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "qkv_norm_rope_vnorm_kvcache_benchmark.csv")
    write_plots(rows, args.output_dir)

    for row in rows:
        print(
            f"{row['provider']:>28}  tokens={row['num_tokens']:>5}  "
            f"head_dim={row['head_dim']:>4}  median_ms={row['median_ms']:.4f}"
        )
