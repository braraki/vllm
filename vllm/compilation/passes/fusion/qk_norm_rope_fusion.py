# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels import qkv_norm_rope_vnorm_triton as _qkv_norm_rope_vnorm_triton  # noqa: F401
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _USE_LAYERNAME, _encode_layer_name, _resolve_layer_name
from vllm.utils.torch_utils import is_quantized_kv_cache

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from ..fx_utils import find_getitem_maybe, is_auto_func, is_func
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
FUSED_QKV_ROPE_VNORM_KVCACHE_OP = (
    torch.ops.vllm.fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update.default
    if hasattr(torch.ops, "vllm")
    and hasattr(
        torch.ops.vllm, "fused_qkv_norm_rope_vnorm_and_unified_kv_cache_update"
    )
    else None
)
RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default
LT_512_FUSED_QK_ROPE_HEAD_DIMS = {64, 128, 256}
CUDA_512_FUSED_QK_ROPE_HEAD_DIMS = LT_512_FUSED_QK_ROPE_HEAD_DIMS | {512}

P = ParamSpec("P")


def _pattern_debug_enabled() -> bool:
    return envs.VLLM_PATTERN_MATCH_DEBUG is not None


@dataclass
class Part4FusionCandidate:
    qkv: fx.Node
    positions: fx.Node | Any
    q_weight: fx.Node
    k_weight: fx.Node
    v_weight: fx.Node
    cos_sin_cache: fx.Node | Any
    layer_name: fx.Node | Any
    split: fx.Node
    q_split: fx.Node
    k_split: fx.Node
    v_split: fx.Node
    q_heads: fx.Node
    k_heads: fx.Node
    v_heads: fx.Node
    kv_cache_dummy: fx.Node
    q_heads_shape: Any
    k_heads_shape: Any
    v_heads_shape: Any
    num_heads: int
    num_kv_heads: int
    q_size: int
    kv_size: int
    head_dim: int
    eps: float
    is_neox: bool
    resolved_layer_name: str | None


def _copy_meta(dst: fx.Node, src: fx.Node) -> None:
    if src.meta:
        dst.meta.update(src.meta)


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
        q_weight = empty_bf16(self.head_dim)
        k_weight = empty_bf16(self.head_dim)
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

    @staticmethod
    def _supports_part4_full_fusion(layer: Attention, head_dim: int) -> bool:
        impl = layer.impl
        if layer.kv_sharing_target_layer_name is not None:
            return False
        if head_dim not in CUDA_512_FUSED_QK_ROPE_HEAD_DIMS:
            return False
        if "triton_attn" not in impl.__class__.__module__:
            return False
        if getattr(impl, "attn_type", None) != "decoder":
            return False
        if getattr(impl, "dcp_world_size", 1) != 1:
            return False
        if is_quantized_kv_cache(getattr(impl, "kv_cache_dtype", "auto")):
            return False
        return True

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_fusion_pass"
        )
        self.experiment_mode = str(
            config.additional_config.get("gemma4_kernel_experiment", "baseline")
        )
        self._part4_reject_log_budget = 0
        self._part4_supported_layers: dict[str, Attention] = {}
        self._part4_supported_signatures: set[tuple[int, int, int]] = set()
        self._part4_ambiguous_signatures: set[tuple[int, int, int]] = set()
        self._part4_custom_rewrite_enabled = (
            self.experiment_mode == "qkv-norm-rope-vnorm-kvcache-fusion"
        )
        if self.experiment_mode in (
            "qk-norm-rope-fusion-512",
            "qkv-norm-rope-vnorm-fusion",
            "qkv-norm-rope-vnorm-kvcache-fusion",
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

        if self._part4_custom_rewrite_enabled:
            if FUSED_QKV_ROPE_VNORM_KVCACHE_OP is None:
                logger.warning(
                    "Skipping qkv-norm-rope-vnorm-kvcache-fusion custom "
                    "rewrite because the fused Part 4 custom op is not "
                    "available in torch.ops.vllm."
                )
                return

            supported_layers = [
                layer
                for layer in attn_layers.values()
                if self._supports_part4_full_fusion(layer, layer.head_size)
            ]
            supported_attn_signatures = sorted(
                {
                    (layer.head_size, layer.num_heads, layer.num_kv_heads)
                    for layer in supported_layers
                }
            )
            skipped_attn_signatures = sorted(
                {
                    (layer.head_size, layer.num_heads, layer.num_kv_heads)
                    for layer in attn_layers.values()
                    if (layer.head_size, layer.num_heads, layer.num_kv_heads)
                    not in supported_attn_signatures
                }
            )
            if skipped_attn_signatures:
                logger.info(
                    "Part 4 full post-GEMM fusion skipping unsupported "
                    "attention signatures: %s",
                    skipped_attn_signatures,
                )
            if not supported_attn_signatures:
                logger.warning_once(
                    "Part 4 full post-GEMM fusion not enabled: no supported "
                    "Gemma4 TritonAttention decoder signatures found"
                )
                return
            self._part4_supported_layers = {
                layer.layer_name: layer for layer in supported_layers
            }
            self._part4_supported_signatures = set(supported_attn_signatures)
            unsupported_signatures = {
                (layer.head_size, layer.num_heads, layer.num_kv_heads)
                for layer in attn_layers.values()
                if not self._supports_part4_full_fusion(layer, layer.head_size)
            }
            self._part4_ambiguous_signatures = (
                self._part4_supported_signatures & unsupported_signatures
            )
            if self._part4_ambiguous_signatures:
                logger.info(
                    "Part 4 full post-GEMM fusion requires exact layer "
                    "resolution for ambiguous signatures: %s",
                    sorted(self._part4_ambiguous_signatures),
                )
            if _pattern_debug_enabled():
                logger.debug(
                    "Part 4 full post-GEMM fusion registering signatures: %s",
                    supported_attn_signatures,
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
                    if self.experiment_mode == "qkv-norm-rope-vnorm-fusion":
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

    def _part4_debug_log(self, message: str, **extra: Any) -> None:
        if not _pattern_debug_enabled():
            return
        if self._part4_reject_log_budget >= 64:
            return
        self._part4_reject_log_budget += 1
        logger.debug(
            "Part 4 custom rewrite: %s%s",
            message,
            f" | {extra}" if extra else "",
        )

    @staticmethod
    def _meta_shape(node: fx.Node | Any) -> tuple[Any, ...] | None:
        if not isinstance(node, fx.Node):
            return None
        val = node.meta.get("val")
        shape = getattr(val, "shape", None)
        if shape is None:
            return None
        return tuple(shape)

    @staticmethod
    def _meta_last_dim(node: fx.Node | Any) -> Any:
        shape = QKNormRoPEFusionPass._meta_shape(node)
        return None if shape is None else shape[-1]

    @staticmethod
    def _iter_non_output_users(node: fx.Node) -> set[fx.Node]:
        return {user for user in node.users if user.op != "output"}

    @staticmethod
    def _output_user_count(node: fx.Node) -> int:
        return sum(1 for user in node.users if user.op == "output")

    @staticmethod
    def _has_expected_users(
        node: fx.Node,
        *,
        expected_non_output: set[fx.Node],
        output_count: int,
    ) -> bool:
        return (
            QKNormRoPEFusionPass._iter_non_output_users(node) == expected_non_output
            and QKNormRoPEFusionPass._output_user_count(node) == output_count
        )

    @staticmethod
    def _erase_if_unused(graph: fx.Graph, node: fx.Node) -> None:
        stack = [node]
        while stack:
            current = stack.pop()
            if current._erased or current.users:
                continue
            inputs = list(current.all_input_nodes)
            graph.erase_node(current)
            stack.extend(inputs)

    def _decode_part4_layer_name(self, layer_name: fx.Node | Any) -> str | None:
        raw_value = layer_name
        if isinstance(layer_name, fx.Node):
            raw_value = layer_name.meta.get("val")
            if raw_value is None:
                raw_value = layer_name.meta.get("example_value")
        if raw_value is None:
            return None
        try:
            return _resolve_layer_name(raw_value)
        except Exception:
            return None

    def _match_part4_candidate(
        self, kv_cache_dummy: fx.Node
    ) -> Part4FusionCandidate | None:
        if not is_func(
            kv_cache_dummy, torch.ops.vllm.unified_kv_cache_update.default
        ):
            return None

        if len(kv_cache_dummy.args) != 3:
            self._part4_debug_log("reject: unexpected unified_kv_cache_update args")
            return None

        k_heads, v_heads, layer_name = kv_cache_dummy.args
        if not isinstance(k_heads, fx.Node) or not isinstance(v_heads, fx.Node):
            self._part4_debug_log(
                "reject: kv cache update inputs are not fx.Nodes",
                node=kv_cache_dummy,
            )
            return None

        if not is_func(v_heads, RMS_NORM_OP):
            self._part4_debug_log("reject: v path is not weighted rms_norm")
            return None
        v_heads_shape = self._meta_shape(v_heads)
        if (
            v_heads_shape is None
            or len(v_heads_shape) < 2
            or not isinstance(v_heads_shape[-1], int)
            or not isinstance(v_heads_shape[-2], int)
        ):
            self._part4_debug_log(
                "reject: v norm output shape missing or dynamic",
                shape=v_heads_shape,
            )
            return None
        num_kv_heads = int(v_heads_shape[-2])
        head_dim = int(v_heads_shape[-1])
        kv_size = num_kv_heads * head_dim

        v_reshape = v_heads.args[0]
        v_weight = v_heads.args[1]
        eps = v_heads.args[2]
        if (
            not isinstance(v_reshape, fx.Node)
            or not is_func(v_reshape, torch.ops.aten.reshape.default)
            or not isinstance(v_weight, fx.Node)
            or self._meta_last_dim(v_weight) != head_dim
        ):
            self._part4_debug_log(
                "reject: invalid v reshape or v weight",
                head_dim=head_dim,
            )
            return None

        if not is_func(k_heads, torch.ops.aten.reshape.default):
            self._part4_debug_log("reject: k_heads is not a reshape")
            return None
        k_rope_getitem = k_heads.args[0]
        if (
            not isinstance(k_rope_getitem, fx.Node)
            or not is_func(k_rope_getitem, operator.getitem)
            or k_rope_getitem.args[1] != 2
        ):
            self._part4_debug_log(
                "reject: k branch does not come from rotary getitem"
            )
            return None

        rotary = k_rope_getitem.args[0]
        if not isinstance(rotary, fx.Node) or not is_auto_func(
            rotary, torch.ops._C.rotary_embedding.default
        ):
            self._part4_debug_log("reject: rotary node mismatch")
            return None

        q_rope_getitem = find_getitem_maybe(rotary, 1)
        if q_rope_getitem is None:
            self._part4_debug_log("reject: missing q rotary getitem")
            return None
        q_heads = next(iter(q_rope_getitem.users), None)
        if (
            not isinstance(q_heads, fx.Node)
            or not is_func(q_heads, torch.ops.aten.reshape.default)
        ):
            self._part4_debug_log("reject: q_heads is not reshape")
            return None

        q_flat = rotary.kwargs.get("query")
        k_flat = rotary.kwargs.get("key")
        positions = rotary.kwargs.get("positions")
        cos_sin_cache = rotary.kwargs.get("cos_sin_cache")
        is_neox = rotary.kwargs.get("is_neox")
        rotary_head_size = rotary.kwargs.get("head_size")

        if rotary_head_size != head_dim or not isinstance(is_neox, bool):
            self._part4_debug_log(
                "reject: rotary metadata mismatch",
                head_size=rotary_head_size,
                is_neox=is_neox,
            )
            return None

        if (
            not isinstance(q_flat, fx.Node)
            or not isinstance(k_flat, fx.Node)
            or not isinstance(positions, fx.Node)
            or not is_func(q_flat, torch.ops.aten.reshape.default)
            or not is_func(k_flat, torch.ops.aten.reshape.default)
        ):
            self._part4_debug_log(
                "reject: q/k flats or positions do not match expected nodes"
            )
            return None

        q_norm = q_flat.args[0]
        k_norm = k_flat.args[0]
        q_weight = q_norm.args[1] if isinstance(q_norm, fx.Node) else None
        k_weight = k_norm.args[1] if isinstance(k_norm, fx.Node) else None
        if (
            not isinstance(q_norm, fx.Node)
            or not isinstance(k_norm, fx.Node)
            or not is_func(q_norm, RMS_NORM_OP)
            or not is_func(k_norm, RMS_NORM_OP)
            or not isinstance(q_weight, fx.Node)
            or not isinstance(k_weight, fx.Node)
            or self._meta_last_dim(q_weight) != head_dim
            or self._meta_last_dim(k_weight) != head_dim
        ):
            self._part4_debug_log("reject: q/k weighted rms_norm mismatch")
            return None

        q_heads_shape = self._meta_shape(q_heads)
        k_heads_shape = self._meta_shape(k_heads)
        if (
            q_heads_shape is None
            or len(q_heads_shape) < 2
            or not isinstance(q_heads_shape[-1], int)
            or not isinstance(q_heads_shape[-2], int)
        ):
            self._part4_debug_log(
                "reject: q_heads shape missing or dynamic",
                shape=q_heads_shape,
            )
            return None
        if (
            k_heads_shape is None
            or len(k_heads_shape) < 2
            or not isinstance(k_heads_shape[-1], int)
            or not isinstance(k_heads_shape[-2], int)
        ):
            self._part4_debug_log(
                "reject: k_heads shape missing or dynamic",
                shape=k_heads_shape,
            )
            return None

        num_heads = int(q_heads_shape[-2])
        q_head_dim = int(q_heads_shape[-1])
        k_num_kv_heads = int(k_heads_shape[-2])
        k_head_dim = int(k_heads_shape[-1])
        if q_head_dim != head_dim:
            self._part4_debug_log(
                "reject: q_heads head_dim mismatch",
                q_head_dim=q_head_dim,
                head_dim=head_dim,
            )
            return None
        if k_num_kv_heads != num_kv_heads or k_head_dim != head_dim:
            self._part4_debug_log(
                "reject: k_heads shape mismatch",
                k_heads_shape=k_heads_shape,
                expected=(num_kv_heads, head_dim),
            )
            return None

        signature = (head_dim, num_heads, num_kv_heads)
        if signature not in self._part4_supported_signatures:
            self._part4_debug_log(
                "reject: unsupported Part 4 signature",
                signature=signature,
            )
            return None

        resolved_layer_name = self._decode_part4_layer_name(layer_name)
        if resolved_layer_name is not None:
            supported_layer = self._part4_supported_layers.get(resolved_layer_name)
            if supported_layer is None:
                self._part4_debug_log(
                    "reject: resolved layer is not eligible for Part 4",
                    resolved_layer_name=resolved_layer_name,
                    signature=signature,
                )
                return None
        elif signature in self._part4_ambiguous_signatures:
            self._part4_debug_log(
                "reject: ambiguous signature requires exact layer resolution",
                signature=signature,
            )
            return None

        q_size = num_heads * head_dim
        layer_debug = resolved_layer_name or f"signature={signature}"

        if q_norm.args[2] != eps or k_norm.args[2] != eps:
            self._part4_debug_log(
                "reject: q/k/v epsilon mismatch",
                q_eps=q_norm.args[2],
                k_eps=k_norm.args[2],
                v_eps=eps,
            )
            return None

        q_reshape = q_norm.args[0]
        k_reshape = k_norm.args[0]
        if (
            not isinstance(q_reshape, fx.Node)
            or not isinstance(k_reshape, fx.Node)
            or not is_func(q_reshape, torch.ops.aten.reshape.default)
            or not is_func(k_reshape, torch.ops.aten.reshape.default)
        ):
            self._part4_debug_log("reject: q/k pre-norm reshapes missing")
            return None

        q_split = q_reshape.args[0]
        k_split = k_reshape.args[0]
        v_split = v_reshape.args[0]
        if (
            not isinstance(q_split, fx.Node)
            or not isinstance(k_split, fx.Node)
            or not isinstance(v_split, fx.Node)
            or not is_func(q_split, operator.getitem)
            or not is_func(k_split, operator.getitem)
            or not is_func(v_split, operator.getitem)
            or q_split.args[1] != 0
            or k_split.args[1] != 1
            or v_split.args[1] != 2
        ):
            self._part4_debug_log("reject: split getitems mismatch")
            return None

        split = q_split.args[0]
        if (
            not isinstance(split, fx.Node)
            or split != k_split.args[0]
            or split != v_split.args[0]
            or not is_func(split, torch.ops.aten.split_with_sizes.default)
        ):
            self._part4_debug_log("reject: q/k/v do not share split_with_sizes")
            return None

        split_sizes = tuple(split.args[1])
        if split_sizes != (q_size, kv_size, kv_size) or split.args[2] != -1:
            self._part4_debug_log(
                "reject: split sizes mismatch",
                split_sizes=split_sizes,
                layer=layer_debug,
            )
            return None

        qkv = split.args[0]
        if not isinstance(qkv, fx.Node):
            self._part4_debug_log("reject: qkv source is not fx.Node")
            return None

        if not self._has_expected_users(
            split,
            expected_non_output={q_split, k_split, v_split},
            output_count=0,
        ):
            self._part4_debug_log("reject: split has unexpected users")
            return None
        if not self._has_expected_users(
            q_split,
            expected_non_output={q_reshape},
            output_count=0,
        ):
            self._part4_debug_log("reject: q split output has unexpected users")
            return None
        if not self._has_expected_users(
            k_split,
            expected_non_output={k_reshape},
            output_count=0,
        ):
            self._part4_debug_log("reject: k split output has unexpected users")
            return None
        if not self._has_expected_users(
            v_split,
            expected_non_output={v_reshape},
            output_count=0,
        ):
            self._part4_debug_log("reject: v split output has unexpected users")
            return None
        if not self._has_expected_users(
            q_reshape,
            expected_non_output={q_norm},
            output_count=0,
        ):
            self._part4_debug_log("reject: q reshape has unexpected users")
            return None
        if not self._has_expected_users(
            k_reshape,
            expected_non_output={k_norm},
            output_count=0,
        ):
            self._part4_debug_log("reject: k reshape has unexpected users")
            return None
        if not self._has_expected_users(
            v_reshape,
            expected_non_output={v_heads},
            output_count=0,
        ):
            self._part4_debug_log("reject: v reshape has unexpected users")
            return None
        if not self._has_expected_users(
            q_norm,
            expected_non_output={q_flat},
            output_count=0,
        ):
            self._part4_debug_log("reject: q norm has unexpected users")
            return None
        if not self._has_expected_users(
            k_norm,
            expected_non_output={k_flat},
            output_count=0,
        ):
            self._part4_debug_log("reject: k norm has unexpected users")
            return None
        if not self._has_expected_users(
            rotary,
            expected_non_output={q_rope_getitem, k_rope_getitem},
            output_count=0,
        ):
            self._part4_debug_log("reject: rotary has unexpected users")
            return None
        if not self._has_expected_users(
            q_rope_getitem,
            expected_non_output={q_heads},
            output_count=0,
        ):
            self._part4_debug_log("reject: q rotary getitem has unexpected users")
            return None
        if not self._has_expected_users(
            k_rope_getitem,
            expected_non_output={k_heads},
            output_count=0,
        ):
            self._part4_debug_log("reject: k rotary getitem has unexpected users")
            return None
        if not self._has_expected_users(
            q_heads,
            expected_non_output=set(),
            output_count=1,
        ):
            self._part4_debug_log("reject: q_heads has unexpected users")
            return None
        if not self._has_expected_users(
            k_heads,
            expected_non_output={kv_cache_dummy},
            output_count=1,
        ):
            self._part4_debug_log("reject: k_heads has unexpected users")
            return None
        if not self._has_expected_users(
            v_heads,
            expected_non_output={kv_cache_dummy},
            output_count=1,
        ):
            self._part4_debug_log("reject: v_heads has unexpected users")
            return None
        if not self._has_expected_users(
            kv_cache_dummy,
            expected_non_output=set(),
            output_count=1,
        ):
            self._part4_debug_log("reject: kv cache dummy has unexpected users")
            return None

        self._part4_debug_log(
            "accepted",
            layer=layer_debug,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            is_neox=is_neox,
        )
        return Part4FusionCandidate(
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            cos_sin_cache=cos_sin_cache,
            layer_name=layer_name,
            split=split,
            q_split=q_split,
            k_split=k_split,
            v_split=v_split,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            kv_cache_dummy=kv_cache_dummy,
            q_heads_shape=q_heads.args[1],
            k_heads_shape=k_heads.args[1],
            v_heads_shape=v_reshape.args[1],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            q_size=q_size,
            kv_size=kv_size,
            head_dim=head_dim,
            eps=float(eps),
            is_neox=is_neox,
            resolved_layer_name=resolved_layer_name,
        )

    def _apply_part4_candidate(
        self, graph: fx.Graph, candidate: Part4FusionCandidate
    ) -> None:
        with graph.inserting_before(candidate.kv_cache_dummy):
            position_ids = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(candidate.positions, [-1]),
            )
            fused = graph.call_function(
                auto_functionalized,
                args=(FUSED_QKV_ROPE_VNORM_KVCACHE_OP,),
                kwargs={
                    "qkv": candidate.qkv,
                    "num_heads_q": candidate.num_heads,
                    "num_heads_k": candidate.num_kv_heads,
                    "num_heads_v": candidate.num_kv_heads,
                    "head_dim": candidate.head_dim,
                    "eps": candidate.eps,
                    "q_weight": candidate.q_weight,
                    "k_weight": candidate.k_weight,
                    "v_weight": candidate.v_weight,
                    "cos_sin_cache": candidate.cos_sin_cache,
                    "is_neox": candidate.is_neox,
                    "position_ids": position_ids,
                    "layer_name": candidate.layer_name,
                    "forced_token_heads_per_warp": -1,
                },
            )
            fused_dummy = graph.call_function(operator.getitem, args=(fused, 0))
            fused_qkv = graph.call_function(operator.getitem, args=(fused, 1))
            split = graph.call_function(
                torch.ops.aten.split_with_sizes.default,
                args=(
                    fused_qkv,
                    [candidate.q_size, candidate.kv_size, candidate.kv_size],
                    -1,
                ),
            )
            q_flat = graph.call_function(operator.getitem, args=(split, 0))
            k_flat = graph.call_function(operator.getitem, args=(split, 1))
            v_flat = graph.call_function(operator.getitem, args=(split, 2))
            q_heads = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(q_flat, candidate.q_heads_shape),
            )
            k_heads = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(k_flat, candidate.k_heads_shape),
            )
            v_heads = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(v_flat, candidate.v_heads_shape),
            )

        _copy_meta(
            position_ids,
            candidate.positions
            if isinstance(candidate.positions, fx.Node)
            else candidate.qkv,
        )
        _copy_meta(fused_dummy, candidate.kv_cache_dummy)
        _copy_meta(fused_qkv, candidate.qkv)
        _copy_meta(q_flat, candidate.q_split)
        _copy_meta(k_flat, candidate.k_split)
        _copy_meta(v_flat, candidate.v_split)
        _copy_meta(q_heads, candidate.q_heads)
        _copy_meta(k_heads, candidate.k_heads)
        _copy_meta(v_heads, candidate.v_heads)

        candidate.q_heads.replace_all_uses_with(q_heads)
        candidate.k_heads.replace_all_uses_with(k_heads)
        candidate.v_heads.replace_all_uses_with(v_heads)
        candidate.kv_cache_dummy.replace_all_uses_with(fused_dummy)

        for node in (
            candidate.kv_cache_dummy,
            candidate.q_heads,
            candidate.k_heads,
            candidate.v_heads,
        ):
            self._erase_if_unused(graph, node)

    def _rewrite_part4(self, graph: fx.Graph) -> int:
        matched = 0
        for node in list(graph.nodes):
            if node._erased or not is_func(
                node, torch.ops.vllm.unified_kv_cache_update.default
            ):
                continue
            candidate = self._match_part4_candidate(node)
            if candidate is None:
                continue
            self._apply_part4_candidate(graph, candidate)
            matched += 1
        return matched

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self._part4_reject_log_budget = 0
        if self._part4_custom_rewrite_enabled:
            self.matched_count = self._rewrite_part4(graph)
        else:
            self.matched_count = self.patterns.apply(graph)
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        logger.debug("Fused QK Norm+RoPE on %s sites", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            QkNormRopePattern,
            QKVNormRopeVNormPattern,
            QKVNormRopeVNormKVCachePattern,
        )


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
        q_weight = empty_bf16(self.head_dim)
        k_weight = empty_bf16(self.head_dim)
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


class QKVNormRopeVNormKVCachePattern:
    """Match the full post-GEMM prep + KV cache update sequence."""

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        is_neox: bool,
        layer_name: str,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps
        self.is_neox = is_neox
        self.layer_name = layer_name
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rotary_op = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        ).rotary_op

    def get_inputs(self) -> list:
        T = 5
        qkv = empty_bf16(T, self.q_size + 2 * self.kv_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(self.head_dim)
        k_weight = empty_bf16(self.head_dim)
        v_weight = empty_bf16(self.head_dim)
        cos_sin_cache = empty_bf16(4096, self.head_dim)
        inputs: list = [qkv, positions, q_weight, k_weight, v_weight, cos_sin_cache]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self.layer_name))
        return inputs

    def register(self, pm_pass: PatternMatcherPass) -> None:
        debug_log_budget = {"count": 0}

        def log_signature_check(message: str, **extra) -> None:
            if not _pattern_debug_enabled():
                return
            if debug_log_budget["count"] >= 24:
                return
            debug_log_budget["count"] += 1
            logger.debug(
                "Part 4 signature_matches[%s hd=%s h=%s kv=%s neox=%s]: %s%s",
                self.layer_name,
                self.head_dim,
                self.num_heads,
                self.num_kv_heads,
                self.is_neox,
                message,
                f" | {extra}" if extra else "",
            )

        def signature_matches(match: pm.Match) -> bool:
            weighted_shapes: list[tuple[int, int]] = []
            weightless_shapes: list[tuple[int, int]] = []
            log_signature_check("called", nodes=len(match.nodes))
            for node in match.nodes:
                if node.target != RMS_NORM_OP:
                    continue
                x = node.args[0]
                weight = node.args[1]
                if not isinstance(x, fx.Node):
                    log_signature_check("rejected: rms_norm input is not fx.Node")
                    return False
                x_shape = tuple(x.meta["val"].shape)
                if x_shape[-1] != self.head_dim:
                    log_signature_check(
                        "rejected: rms_norm input last dim mismatch",
                        x_shape=x_shape,
                    )
                    return False
                if weight is None:
                    weightless_shapes.append((x_shape[-2], x_shape[-1]))
                    continue
                if not isinstance(weight, fx.Node):
                    log_signature_check(
                        "rejected: rms_norm weight is not fx.Node",
                        x_shape=x_shape,
                    )
                    return False
                weight_shape = tuple(weight.meta["val"].shape)
                if weight_shape[-1] != self.head_dim:
                    log_signature_check(
                        "rejected: rms_norm weight last dim mismatch",
                        x_shape=x_shape,
                        weight_shape=weight_shape,
                    )
                    return False
                weighted_shapes.append((x_shape[-2], x_shape[-1]))

            matched = (
                weighted_shapes.count((self.num_heads, self.head_dim)) >= 1
                and weighted_shapes.count((self.num_kv_heads, self.head_dim)) >= 2
                and weightless_shapes.count((self.num_kv_heads, self.head_dim)) == 0
            )
            if matched:
                log_signature_check(
                    "accepted",
                    weighted_shapes=weighted_shapes,
                    weightless_shapes=weightless_shapes,
                )
            else:
                log_signature_check(
                    "rejected: weighted/weightless shape counts",
                    weighted_shapes=weighted_shapes,
                    weightless_shapes=weightless_shapes,
                )
            return matched

        encoded_layer_name = _encode_layer_name(self.layer_name)

        if _USE_LAYERNAME:

            def pattern(
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
                layer_name,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                head_dim = q_weight.shape[-1]
                kv_size = self.num_kv_heads * head_dim
                q_size = qkv.shape[-1] - 2 * kv_size
                q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
                num_heads_q = q.shape[-1] // head_dim

                q_by_head = q.view(*q.shape[:-1], num_heads_q, head_dim)
                q_normed_by_head = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
                q_flat = q_normed_by_head.view(q.shape)

                k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, head_dim)
                k_normed_by_head = vllm.ir.ops.rms_norm(
                    k_by_head, k_weight, self.eps
                )
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

                v_heads = v.view(*v.shape[:-1], self.num_kv_heads, self.head_dim)
                v_normed_heads = vllm.ir.ops.rms_norm(v_heads, v_weight, self.eps)

                q_heads = q_rope.view(-1, self.num_heads, self.head_dim)
                k_heads = k_rope.view(-1, self.num_kv_heads, self.head_dim)
                kv_cache_dummy = torch.ops.vllm.unified_kv_cache_update(
                    k_heads, v_normed_heads, layer_name
                )
                return q_heads, k_heads, v_normed_heads, kv_cache_dummy

            def replacement(
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
                layer_name,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                assert FUSED_QKV_ROPE_VNORM_KVCACHE_OP is not None
                head_dim = q_weight.shape[-1]
                kv_size = self.num_kv_heads * head_dim
                q_size = qkv.shape[-1] - 2 * kv_size
                num_heads_q = q_size // head_dim
                result = auto_functionalized(
                    FUSED_QKV_ROPE_VNORM_KVCACHE_OP,
                    qkv=qkv,
                    num_heads_q=num_heads_q,
                    num_heads_k=self.num_kv_heads,
                    num_heads_v=self.num_kv_heads,
                    head_dim=head_dim,
                    eps=self.eps,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    v_weight=v_weight,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=self.is_neox,
                    position_ids=positions.view(-1),
                    layer_name=layer_name,
                    forced_token_heads_per_warp=-1,
                )
                result_qkv = result[1]
                q, k, v = result_qkv.split([q_size, kv_size, kv_size], dim=-1)
                q = q.view(-1, self.num_heads, self.head_dim)
                k = k.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)
                return q, k, v, result[0]

        else:

            def pattern(
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                head_dim = q_weight.shape[-1]
                kv_size = self.num_kv_heads * head_dim
                q_size = qkv.shape[-1] - 2 * kv_size
                q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
                num_heads_q = q.shape[-1] // head_dim

                q_by_head = q.view(*q.shape[:-1], num_heads_q, head_dim)
                q_normed_by_head = vllm.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
                q_flat = q_normed_by_head.view(q.shape)

                k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, head_dim)
                k_normed_by_head = vllm.ir.ops.rms_norm(
                    k_by_head, k_weight, self.eps
                )
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

                v_heads = v.view(*v.shape[:-1], self.num_kv_heads, self.head_dim)
                v_normed_heads = vllm.ir.ops.rms_norm(v_heads, v_weight, self.eps)

                q_heads = q_rope.view(-1, self.num_heads, self.head_dim)
                k_heads = k_rope.view(-1, self.num_kv_heads, self.head_dim)
                kv_cache_dummy = torch.ops.vllm.unified_kv_cache_update(
                    k_heads, v_normed_heads, encoded_layer_name
                )
                return q_heads, k_heads, v_normed_heads, kv_cache_dummy

            def replacement(
                qkv: torch.Tensor,
                positions: torch.Tensor,
                q_weight: torch.Tensor,
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                cos_sin_cache: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                assert FUSED_QKV_ROPE_VNORM_KVCACHE_OP is not None
                head_dim = q_weight.shape[-1]
                kv_size = self.num_kv_heads * head_dim
                q_size = qkv.shape[-1] - 2 * kv_size
                num_heads_q = q_size // head_dim
                result = auto_functionalized(
                    FUSED_QKV_ROPE_VNORM_KVCACHE_OP,
                    qkv=qkv,
                    num_heads_q=num_heads_q,
                    num_heads_k=self.num_kv_heads,
                    num_heads_v=self.num_kv_heads,
                    head_dim=head_dim,
                    eps=self.eps,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    v_weight=v_weight,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=self.is_neox,
                    position_ids=positions.view(-1),
                    layer_name=encoded_layer_name,
                    forced_token_heads_per_warp=-1,
                )
                result_qkv = result[1]
                q, k, v = result_qkv.split([q_size, kv_size, kv_size], dim=-1)
                q = q.view(-1, self.num_heads, self.head_dim)
                k = k.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)
                return q, k, v, result[0]

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
