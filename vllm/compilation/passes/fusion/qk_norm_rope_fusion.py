# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels import qkv_norm_rope_vnorm_triton as _qkv_norm_rope_vnorm_triton  # noqa: F401
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)

FUSED_QK_ROPE_OP = torch.ops._C.fused_qk_norm_rope.default
FUSED_QKV_ROPE_VNORM_OP = (
    torch.ops.vllm.fused_qkv_norm_rope_vnorm.default
    if hasattr(torch.ops, "vllm")
    and hasattr(torch.ops.vllm, "fused_qkv_norm_rope_vnorm")
    else None
)
RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default
LT_512_FUSED_QK_ROPE_HEAD_DIMS = {64, 128, 256}
CUDA_512_FUSED_QK_ROPE_HEAD_DIMS = LT_512_FUSED_QK_ROPE_HEAD_DIMS | {512}

P = ParamSpec("P")


class QkNormRopePattern:
    """
    Match the unfused sequence in attention blocks and replace with the fused op.

    Unfused (conceptually):
      q, k, v = split(qkv, [qsz, kvsz, kvsz], -1)
      qh = reshape(q, [-1, num_heads, head_dim])
      kh = reshape(k, [-1, num_kv_heads, head_dim])
      qn = rms_norm(qh, q_weight, eps)
      kn = rms_norm(kh, k_weight, eps)
      qf = reshape(qn, [-1, num_heads * head_dim])
      kf = reshape(kn, [-1, num_kv_heads * head_dim])
      qf, kf = rotary_embedding(positions, qf, kf, head_dim, cos_sin_cache, is_neox)
      return qf, kf, v

    Fused replacement:
      fused_qk_norm_rope(qkv, num_heads, num_kv_heads, num_kv_heads, head_dim,
                         eps, q_weight, k_weight, cos_sin_cache, is_neox,
                         positions.view(-1))
      return split(qkv, [qsz, kvsz, kvsz], -1)
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        rope_flashinfer: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rotary_op = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=self.rope_flashinfer,
        ).rotary_op

    def get_inputs(self) -> list[torch.Tensor]:
        # Sample inputs to help pattern tracing
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(4096, self.head_dim)
        else:
            cos_sin_cache = empty_bf16(4096, self.head_dim)
        return [
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
        ]

    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> fx.GraphModule:
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)

            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def signature_matches(match: pm.Match) -> bool:
            rms_input_shapes: list[tuple[int, int]] = []
            for node in match.nodes:
                if node.target != RMS_NORM_OP:
                    continue
                x, weight = node.args[0], node.args[1]
                if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
                    return False
                x_shape = tuple(x.meta["val"].shape)
                weight_shape = tuple(weight.meta["val"].shape)
                if x_shape[-1] != self.head_dim or weight_shape[-1] != self.head_dim:
                    return False
                rms_input_shapes.append((x_shape[-2], x_shape[-1]))

            return (
                (self.num_heads, self.head_dim) in rms_input_shapes
                and (self.num_kv_heads, self.head_dim) in rms_input_shapes
            )

        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            head_dim = q_weight.shape[-1]
            kv_size = self.num_kv_heads * head_dim
            q_size = qkv.shape[-1] - 2 * kv_size
            # split qkv -> q,k,v
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            num_heads_q = q.shape[-1] // head_dim

            # Q path: view -> RMS -> view back to q.shape
            q_by_head = q.view(*q.shape[:-1], num_heads_q, head_dim)
            q_normed_by_head = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
            q_flat = q_normed_by_head.view(q.shape)

            # K path: view -> RMS -> view back to k.shape
            k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, head_dim)
            k_normed_by_head = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
            k_flat = k_normed_by_head.view(k.shape)

            # RoPE: apply to flattened q/k
            result = auto_functionalized(
                self.rotary_op,
                positions=positions,
                query=q_flat,
                key=k_flat,
                head_size=head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
            )
            q_rope = result[1]
            k_rope = result[2]
            return q_rope, k_rope, v

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            head_dim = q_weight.shape[-1]
            kv_size = self.num_kv_heads * head_dim
            q_size = qkv.shape[-1] - 2 * kv_size
            num_heads_q = q_size // head_dim
            # Run fused qk_norm_rope op
            result = auto_functionalized(
                FUSED_QK_ROPE_OP,
                qkv=qkv,
                num_heads_q=num_heads_q,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=head_dim,
                eps=self.eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                position_ids=positions.view(-1),
                forced_token_heads_per_warp=-1,
            )
            result_qkv = result[1]

            # Split back to q,k,v and return
            return result_qkv.split([q_size, kv_size, kv_size], dim=-1)  # type: ignore[no-any-return]

        # NOTE: use fx_view_to_reshape to unify view/reshape to simplify
        # pattern and increase matching opportunities
        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            QkNormRopePattern.wrap_trace_fn(
                pm.fwd_only,
                QkNormRopePattern.fx_view_to_reshape,
            ),
            pm_pass,
            extra_check=signature_matches,
        )


class QKNormRoPEFusionPass(VllmPatternMatcherPass):
    """Fuse Q/K RMSNorm + RoPE into fused_qk_norm_rope when the custom op exists."""

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_fusion_pass"
        )
        experiment_mode = str(
            config.additional_config.get("gemma4_kernel_experiment", "baseline")
        )
        if experiment_mode in (
            "qk-norm-rope-fusion-512",
            "qkv-norm-rope-vnorm-fusion",
        ) and current_platform.is_cuda():
            supported_head_dims = CUDA_512_FUSED_QK_ROPE_HEAD_DIMS
        else:
            supported_head_dims = LT_512_FUSED_QK_ROPE_HEAD_DIMS

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE fusion not enabled: unsupported dtype %s", dtype
            )
            return

        attn_layers: dict[str, Attention] = get_layers_from_vllm_config(
            config, Attention
        )
        if len(attn_layers) == 0:
            logger.warning_once(
                "QK Norm+RoPE fusion enabled, but no Attention layers were discovered."
            )
            return

        attn_signatures = sorted(
            {
                (layer.head_size, layer.num_heads, layer.num_kv_heads)
                for layer in attn_layers.values()
            }
        )
        supported_attn_signatures = [
            signature
            for signature in attn_signatures
            if signature[0] in supported_head_dims
        ]
        skipped_attn_signatures = [
            signature
            for signature in attn_signatures
            if signature[0] not in supported_head_dims
        ]
        if skipped_attn_signatures:
            logger.info(
                "QK Norm+RoPE fusion skipping unsupported attention signatures: %s",
                skipped_attn_signatures,
            )
        if not supported_attn_signatures:
            logger.warning_once(
                "QK Norm+RoPE fusion not enabled: no attention signatures with "
                "supported head dimensions %s",
                sorted(supported_head_dims),
            )
            return
        if len(attn_signatures) > 1:
            logger.info(
                "QK Norm+RoPE fusion registering %d attention signatures: %s",
                len(supported_attn_signatures),
                supported_attn_signatures,
            )

        for epsilon in [1e-5, 1e-6]:
            for neox in [True, False]:
                for head_dim, num_heads, num_kv_heads in supported_attn_signatures:
                    if experiment_mode == "qkv-norm-rope-vnorm-fusion":
                        if FUSED_QKV_ROPE_VNORM_OP is None:
                            logger.warning(
                                "Skipping qkv-norm-rope-vnorm-fusion pattern "
                                "registration because fused_qkv_norm_rope_vnorm "
                                "is not available in torch.ops.vllm."
                            )
                            continue
                        QKVNormRopeVNormPattern(
                            head_dim=head_dim,
                            num_heads=num_heads,
                            num_kv_heads=num_kv_heads,
                            eps=epsilon,
                            is_neox=neox,
                        ).register(self.patterns)
                    else:
                        if RotaryEmbedding.enabled():
                            for rope_flashinfer in [False, True]:
                                QkNormRopePattern(
                                    head_dim=head_dim,
                                    num_heads=num_heads,
                                    num_kv_heads=num_kv_heads,
                                    eps=epsilon,
                                    is_neox=neox,
                                    rope_flashinfer=rope_flashinfer,
                                ).register(self.patterns)
                        else:
                            QkNormRopePattern(
                                head_dim=head_dim,
                                num_heads=num_heads,
                                num_kv_heads=num_kv_heads,
                                eps=epsilon,
                                is_neox=neox,
                            ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused QK Norm+RoPE on %s sites", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, QkNormRopePattern)


class QKVNormRopeVNormPattern:
    """Match Gemma4 non-KV-shared attention prep and replace it with one op."""

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.is_neox = is_neox
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rotary_op = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        ).rotary_op

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(1, self.head_dim)
        k_weight = empty_bf16(1, self.head_dim)
        cos_sin_cache = empty_bf16(4096, self.head_dim)
        return [
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def signature_matches(match: pm.Match) -> bool:
            weighted_shapes: list[tuple[int, int]] = []
            weightless_shapes: list[tuple[int, int]] = []
            for node in match.nodes:
                if node.target != RMS_NORM_OP:
                    continue
                x = node.args[0]
                weight = node.args[1]
                if not isinstance(x, fx.Node):
                    return False
                x_shape = tuple(x.meta["val"].shape)
                if x_shape[-1] != self.head_dim:
                    return False
                if weight is None:
                    weightless_shapes.append((x_shape[-2], x_shape[-1]))
                    continue
                if not isinstance(weight, fx.Node):
                    return False
                weight_shape = tuple(weight.meta["val"].shape)
                if weight_shape[-1] != self.head_dim:
                    return False
                weighted_shapes.append((x_shape[-2], x_shape[-1]))

            return (
                (self.num_heads, self.head_dim) in weighted_shapes
                and (self.num_kv_heads, self.head_dim) in weighted_shapes
                and weightless_shapes.count((self.num_kv_heads, self.head_dim)) == 1
            )

        def pattern(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            head_dim = q_weight.shape[-1]
            kv_size = self.num_kv_heads * head_dim
            q_size = qkv.shape[-1] - 2 * kv_size
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            num_heads_q = q.shape[-1] // head_dim

            q_by_head = q.view(*q.shape[:-1], num_heads_q, head_dim)
            q_normed_by_head = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
            q_flat = q_normed_by_head.view(q.shape)

            k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, head_dim)
            k_normed_by_head = vllm.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
            k_flat = k_normed_by_head.view(k.shape)

            result = auto_functionalized(
                self.rotary_op,
                positions=positions,
                query=q_flat,
                key=k_flat,
                head_size=head_dim,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
            )
            q_rope = result[1]
            k_rope = result[2]

            v_by_head = v.view(*v.shape[:-1], self.num_kv_heads, head_dim)
            v_normed_by_head = vllm.ir.ops.rms_norm(v_by_head, None, self.eps)
            v_flat = v_normed_by_head.view(v.shape)
            return q_rope, k_rope, v_flat

        def replacement(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            cos_sin_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert FUSED_QKV_ROPE_VNORM_OP is not None
            head_dim = q_weight.shape[-1]
            kv_size = self.num_kv_heads * head_dim
            q_size = qkv.shape[-1] - 2 * kv_size
            num_heads_q = q_size // head_dim
            result = auto_functionalized(
                FUSED_QKV_ROPE_VNORM_OP,
                qkv=qkv,
                num_heads_q=num_heads_q,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=head_dim,
                eps=self.eps,
                q_weight=q_weight,
                k_weight=k_weight,
                cos_sin_cache=cos_sin_cache,
                is_neox=self.is_neox,
                position_ids=positions.view(-1),
                forced_token_heads_per_warp=-1,
            )
            result_qkv = result[1]
            return result_qkv.split([q_size, kv_size, kv_size], dim=-1)  # type: ignore[no-any-return]

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            QkNormRopePattern.wrap_trace_fn(
                pm.fwd_only,
                QkNormRopePattern.fx_view_to_reshape,
            ),
            pm_pass,
            extra_check=signature_matches,
        )
