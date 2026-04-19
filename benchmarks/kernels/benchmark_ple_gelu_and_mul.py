# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Gemma4 PLE GELU x multiply implementations.

Example:
    python benchmarks/kernels/benchmark_ple_gelu_and_mul.py \
        --num-tokens 1 16 128 1024 \
        --ple-dim 256 512 \
        --dtype bfloat16 \
        --output-dir /tmp/ple_gelu_and_mul_bench
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.activation  # noqa: F401
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.activation import GeluAndMul, PLEGeluAndMul
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

DEFAULT_NUM_TOKENS = [1, 16, 128, 1024, 4096]


def _load_gemma4_ple_dim(model: str) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config
    ple_dim = getattr(config, "hidden_size_per_layer_input", 0)
    if not ple_dim:
        raise ValueError(
            f"Model {model!r} does not expose hidden_size_per_layer_input in config"
        )
    return int(ple_dim)


def _resolve_ple_dims(model: str | None, ple_dims: list[int] | None) -> list[int]:
    if ple_dims:
        return ple_dims
    if model:
        return [_load_gemma4_ple_dim(model)]
    return [256]


def _native_separate(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.gelu(gate, approximate="tanh") * value


@default_vllm_config()
def benchmark_provider(
    provider: str,
    num_tokens: int,
    ple_dim: int,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_ple_gelu_and_mul requires CUDA")

    set_random_seed(42)
    torch.set_default_device("cuda")

    gate = torch.randn(num_tokens, ple_dim, dtype=dtype, device="cuda")
    value = torch.randn_like(gate)
    cat_op = GeluAndMul(approximate="tanh")
    custom_op = PLEGeluAndMul()
    compiled_native = torch.compile(_native_separate)

    if provider == "native_separate":
        fn = lambda: _native_separate(gate, value)
    elif provider == "native_compiled":
        fn = lambda: compiled_native(gate, value)
    elif provider == "cat_plus_existing_custom":
        fn = lambda: cat_op(torch.cat((gate, value), dim=-1))
    elif provider == "custom_two_input":
        fn = lambda: custom_op(gate, value)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, min_ms, max_ms


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "num_tokens",
                "ple_dim",
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
    ple_dims = sorted({int(row["ple_dim"]) for row in rows})
    providers = sorted({str(row["provider"]) for row in rows})

    for ple_dim in ple_dims:
        plt.figure(figsize=(9, 5))
        dim_rows = [row for row in rows if int(row["ple_dim"]) == ple_dim]
        for provider in providers:
            provider_rows = sorted(
                (row for row in dim_rows if row["provider"] == provider),
                key=lambda row: int(row["num_tokens"]),
            )
            if not provider_rows:
                continue
            plt.plot(
                [int(row["num_tokens"]) for row in provider_rows],
                [float(row["median_ms"]) for row in provider_rows],
                marker="o",
                label=provider,
            )
        plt.title(f"PLE GELU x Mul Benchmark (ple_dim={ple_dim})")
        plt.xlabel("num_tokens")
        plt.ylabel("median latency (ms)")
        plt.xscale("log")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"ple_gelu_and_mul_ple_dim_{ple_dim}.png")
        plt.close()


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark Gemma4 PLE GELU x multiply implementations."
    )
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_TOKENS,
        help="List of token counts to benchmark.",
    )
    parser.add_argument(
        "--ple-dim",
        nargs="+",
        type=int,
        default=None,
        help="List of PLE hidden dimensions to benchmark.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face model id used to derive hidden_size_per_layer_input "
            "when --ple-dim is omitted."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./benchmark_ple_gelu_and_mul_results"),
        help="Directory that will receive CSV and plot outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    ple_dims = _resolve_ple_dims(args.model, args.ple_dim)
    providers = [
        "native_separate",
        "native_compiled",
        "cat_plus_existing_custom",
        "custom_two_input",
    ]

    rows: list[dict[str, float | int | str]] = []
    for ple_dim in ple_dims:
        for num_tokens in args.num_tokens:
            for provider in providers:
                median_ms, min_ms, max_ms = benchmark_provider(
                    provider=provider,
                    num_tokens=num_tokens,
                    ple_dim=ple_dim,
                    dtype=dtype,
                )
                rows.append(
                    {
                        "provider": provider,
                        "num_tokens": num_tokens,
                        "ple_dim": ple_dim,
                        "median_ms": median_ms,
                        "min_ms": min_ms,
                        "max_ms": max_ms,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "ple_gelu_and_mul_benchmark.csv")
    write_plots(rows, args.output_dir)

    for row in rows:
        print(
            f"{row['provider']:>24}  tokens={row['num_tokens']:>5}  "
            f"ple_dim={row['ple_dim']:>5}  median_ms={row['median_ms']:.4f}"
        )
