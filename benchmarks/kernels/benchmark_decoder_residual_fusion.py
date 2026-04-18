# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Gemma4 decoder residual fusion implementations.

This benchmark isolates the pre-feedforward residual handoff in
``Gemma4DecoderLayer.forward``:

Baseline:
    hidden_states = hidden_states + residual
    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)

Fusion:
    hidden_states, residual = self.pre_feedforward_layernorm(
        hidden_states, residual
    )

Example:
    python benchmarks/kernels/benchmark_decoder_residual_fusion.py \
        --num-tokens 1 4 16 64 256 1024 \
        --model google/gemma-4-E2B-it \
        --dtype bfloat16 \
        --output-dir /tmp/decoder_residual_fusion_bench
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm, fused_add_rms_norm
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_NUM_TOKENS = [1, 4, 16, 64, 256, 1024]
DEFAULT_PROVIDERS = [
    "baseline_eager",
    "baseline_compiled",
    "fusion_custom_op",
]


def _load_gemma4_hidden_size_and_eps(model: str) -> tuple[int, float]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if hasattr(config, "text_config"):
        config = config.text_config

    hidden_size = getattr(config, "hidden_size", 0)
    if not hidden_size:
        raise ValueError(f"Model {model!r} does not expose hidden_size")

    eps = getattr(config, "rms_norm_eps", None)
    if eps is None:
        raise ValueError(f"Model {model!r} does not expose rms_norm_eps")

    return int(hidden_size), float(eps)


def _resolve_hidden_sizes(
    model: str | None, hidden_sizes: list[int] | None
) -> list[int]:
    if hidden_sizes:
        return hidden_sizes
    if model:
        hidden_size, _ = _load_gemma4_hidden_size_and_eps(model)
        return [hidden_size]
    raise ValueError("Either --hidden-size or --model must be provided")


def _resolve_eps(model: str | None, eps: float | None) -> float:
    if eps is not None:
        return float(eps)
    if model:
        _, model_eps = _load_gemma4_hidden_size_and_eps(model)
        return model_eps
    return 1e-6


def baseline_residual_rmsnorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_residual = hidden_states + residual
    normalized_hidden_states = RMSNorm.forward_static(
        updated_residual,
        variance_epsilon=eps,
        hidden_size=hidden_states.shape[-1],
        orig_dtype=hidden_states.dtype,
        weight=weight,
    )
    return normalized_hidden_states, updated_residual


def fusion_residual_rmsnorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = hidden_states.clone()
    residual = residual.clone()
    return fused_add_rms_norm(hidden_states, residual, weight, eps)


def layer_api_residual_rmsnorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = RMSNorm(
        hidden_size=hidden_states.shape[-1],
        eps=eps,
        dtype=weight.dtype,
    ).to(device=hidden_states.device, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer(hidden_states.clone(), residual.clone())


@default_vllm_config()
def run_provider(
    provider: str,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if provider == "baseline_eager":
        return baseline_residual_rmsnorm(hidden_states, residual, weight, eps)
    if provider == "baseline_compiled":
        compiled_fn = torch.compile(baseline_residual_rmsnorm)
        return compiled_fn(hidden_states, residual, weight, eps)
    if provider == "fusion_custom_op":
        return fusion_residual_rmsnorm(hidden_states, residual, weight, eps)
    if provider == "fusion_layer_api":
        return layer_api_residual_rmsnorm(hidden_states, residual, weight, eps)
    raise ValueError(f"Unknown provider: {provider}")


@default_vllm_config()
def benchmark_provider(
    provider: str,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
) -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_decoder_residual_fusion requires CUDA")

    set_random_seed(42)
    torch.set_default_device("cuda")

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(hidden_states)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    if provider == "baseline_eager":
        fn = lambda: baseline_residual_rmsnorm(
            hidden_states.clone(), residual.clone(), weight, eps
        )
    elif provider == "baseline_compiled":
        compiled_fn = torch.compile(baseline_residual_rmsnorm)
        fn = lambda: compiled_fn(
            hidden_states.clone(), residual.clone(), weight, eps
        )
    elif provider == "fusion_custom_op":
        fn = lambda: fusion_residual_rmsnorm(hidden_states, residual, weight, eps)
    elif provider == "fusion_layer_api":
        layer = RMSNorm(hidden_size=hidden_size, eps=eps, dtype=dtype).to(
            device="cuda", dtype=dtype
        )
        with torch.no_grad():
            layer.weight.copy_(weight)
        fn = lambda: layer(hidden_states.clone(), residual.clone())
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, min_ms, max_ms


def validate_outputs(
    hidden_size: int,
    dtype: torch.dtype,
    eps: float,
    num_tokens: int = 8,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("validate_outputs requires CUDA")

    set_random_seed(7)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(hidden_states)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    baseline_hidden_states, baseline_residual = run_provider(
        "baseline_eager",
        hidden_states,
        residual,
        weight,
        eps,
    )
    fused_hidden_states, fused_residual = run_provider(
        "fusion_custom_op",
        hidden_states,
        residual,
        weight,
        eps,
    )

    if not torch.allclose(
        baseline_hidden_states, fused_hidden_states, atol=1e-2, rtol=1e-2
    ):
        raise AssertionError(
            "Fused decoder residual output does not match the baseline"
        )

    if not torch.allclose(
        baseline_residual, fused_residual, atol=1e-2, rtol=1e-2
    ):
        raise AssertionError(
            "Fused decoder residual residual-threading does not match the baseline"
        )


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "num_tokens",
                "hidden_size",
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
    hidden_sizes = sorted({int(row["hidden_size"]) for row in rows})
    providers = sorted({str(row["provider"]) for row in rows})

    for hidden_size in hidden_sizes:
        plt.figure(figsize=(9, 5))
        hidden_rows = [row for row in rows if int(row["hidden_size"]) == hidden_size]
        for provider in providers:
            provider_rows = sorted(
                (row for row in hidden_rows if row["provider"] == provider),
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
        plt.title(f"Decoder Residual Fusion Benchmark (hidden_size={hidden_size})")
        plt.xlabel("num_tokens")
        plt.ylabel("median latency (ms)")
        plt.xscale("log")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"decoder_residual_fusion_hidden_size_{hidden_size}.png")
        plt.close()


def parse_args():
    parser = FlexibleArgumentParser(
        description="Benchmark Gemma4 decoder residual fusion implementations."
    )
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=DEFAULT_NUM_TOKENS,
        help="List of flattened token counts to benchmark.",
    )
    parser.add_argument(
        "--hidden-size",
        nargs="+",
        type=int,
        default=None,
        help="List of hidden sizes to benchmark. Defaults to the model hidden size.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Optional Hugging Face model id used to derive hidden_size and eps "
            "when explicit overrides are omitted."
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
        choices=DEFAULT_PROVIDERS + ["fusion_layer_api"],
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
        default=Path("./benchmark_decoder_residual_fusion_results"),
        help="Directory that will receive CSV and plot outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    hidden_sizes = _resolve_hidden_sizes(args.model, args.hidden_size)
    eps = _resolve_eps(args.model, args.eps)

    if not args.skip_correctness_check:
        for hidden_size in hidden_sizes:
            validate_outputs(hidden_size=hidden_size, dtype=dtype, eps=eps)

    rows: list[dict[str, float | int | str]] = []
    for hidden_size in hidden_sizes:
        for num_tokens in args.num_tokens:
            for provider in args.providers:
                median_ms, min_ms, max_ms = benchmark_provider(
                    provider=provider,
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    eps=eps,
                )
                rows.append(
                    {
                        "provider": provider,
                        "num_tokens": num_tokens,
                        "hidden_size": hidden_size,
                        "dtype": args.dtype,
                        "median_ms": median_ms,
                        "min_ms": min_ms,
                        "max_ms": max_ms,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "decoder_residual_fusion_benchmark.csv")
    write_plots(rows, args.output_dir)

    for row in rows:
        print(
            f"{row['provider']:>18}  tokens={row['num_tokens']:>5}  "
            f"hidden_size={row['hidden_size']:>5}  median_ms={row['median_ms']:.4f}"
        )
