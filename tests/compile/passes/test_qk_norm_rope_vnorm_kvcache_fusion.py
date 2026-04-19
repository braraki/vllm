# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm.engine.arg_utils import EngineArgs
from vllm.compilation.passes.fusion.qk_norm_rope_fusion import (
    FUSED_QKV_ROPE_VNORM_KVCACHE_OP,
    QKNormRoPEFusionPass,
    RMS_NORM_OP,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.compilation.passes.utility.split_coalescing import SplitCoalescingPass
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.registry import AttentionBackendEnum

VLLM_UNIFIED_KV_CACHE_UPDATE_OP = torch.ops.vllm.unified_kv_cache_update.default


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


class QKVNormRoPEVNormKVCacheTestModel(torch.nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float,
        is_neox: bool,
        vllm_config: VllmConfig,
        dtype: torch.dtype,
        include_v_norm: bool = True,
        prefix: str = "model.layers.0.self_attn.attn",
        attn_backend=AttentionBackendEnum.TRITON_ATTN.get_class(),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.eps = eps
        self.include_v_norm = include_v_norm
        self.layer_name = prefix

        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / head_dim**0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=prefix,
            attn_backend=attn_backend,
        )
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        # Match the effective serving graph shape, which carries a tensor
        # in the V-norm weight slot.
        self.v_norm = RMSNorm(head_dim, eps=eps)
        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=is_neox,
            dtype=dtype,
        )

    def forward(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if self.include_v_norm:
            v = self.v_norm(v)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        kv_cache_dummy = torch.ops.vllm.unified_kv_cache_update(
            k,
            v,
            _encode_layer_name(self.layer_name),
        )
        return kv_cache_dummy, q, k, v


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Part 4 compile-pass coverage is CUDA-only",
)
def test_qkv_norm_rope_vnorm_kvcache_fusion(dtype, head_dim, is_neox, eps):
    if FUSED_QKV_ROPE_VNORM_KVCACHE_OP is None:
        pytest.skip("Part 4 fused post-GEMM custom op not available")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+rotary_embedding"],
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                eliminate_noops=True,
            ),
        ),
        additional_config={
            "gemma4_kernel_experiment": "qkv-norm-rope-vnorm-kvcache-fusion"
        },
    )

    T = 5
    slot_mapping = torch.arange(T, dtype=torch.long, device="cuda")

    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = QKVNormRoPEVNormKVCacheTestModel(
            num_heads=8,
            num_kv_heads=1,
            head_dim=head_dim,
            eps=eps,
            is_neox=is_neox,
            vllm_config=vllm_config,
            dtype=dtype,
        )

        noop_pass = NoOpEliminationPass(vllm_config)
        coalesce_pass = SplitCoalescingPass(vllm_config)
        fusion_pass = QKNormRoPEFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, coalesce_pass, fusion_pass, cleanup_pass)
        backend_baseline = TestBackend(noop_pass, cleanup_pass)

        qkv = torch.randn(T, model.q_size + 2 * model.kv_size)
        pos = torch.arange(T, dtype=torch.long, device=qkv.device)
        qkv_unfused = qkv.clone()
        pos_unfused = pos.clone()

        baseline_cache = _allocate_kv_cache(
            model.attn,
            block_size=vllm_config.cache_config.block_size,
            num_tokens=T,
            dtype=dtype,
            device=qkv.device,
        )
        fused_cache = baseline_cache.clone()

        with set_forward_context(
            None,
            vllm_config,
            slot_mapping={model.layer_name: slot_mapping},
        ):
            model.attn.kv_cache = fused_cache
            torch._dynamo.mark_dynamic(qkv, 0)
            torch._dynamo.mark_dynamic(pos, 0)
            model_fused = torch.compile(model, backend=backend)
            fused_outputs = model_fused(qkv, pos)
            fused_cache_after = model.attn.kv_cache.clone()

        with set_forward_context(
            None,
            vllm_config,
            slot_mapping={model.layer_name: slot_mapping},
        ):
            model.attn.kv_cache = baseline_cache
            torch._dynamo.mark_dynamic(qkv_unfused, 0)
            torch._dynamo.mark_dynamic(pos_unfused, 0)
            model_unfused = torch.compile(model, backend=backend_baseline)
            unfused_outputs = model_unfused(qkv_unfused, pos_unfused)
            baseline_cache_after = model.attn.kv_cache.clone()

        atol, rtol = ((2e-3, 2e-3) if dtype == torch.float16 else (2e-2, 1e-2))
        for fused, unfused in zip(fused_outputs, unfused_outputs):
            torch.testing.assert_close(unfused, fused, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            baseline_cache_after,
            fused_cache_after,
            atol=atol,
            rtol=rtol,
        )

        assert fusion_pass.matched_count == 1
        backend.check_before_ops([RMS_NORM_OP, VLLM_UNIFIED_KV_CACHE_UPDATE_OP])
        backend.check_after_ops([FUSED_QKV_ROPE_VNORM_KVCACHE_OP])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Part 4 compile-pass coverage is CUDA-only",
)
def test_qkv_norm_rope_vnorm_kvcache_does_not_match_without_v_norm(dtype):
    if FUSED_QKV_ROPE_VNORM_KVCACHE_OP is None:
        pytest.skip("Part 4 fused post-GEMM custom op not available")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+rotary_embedding"],
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                eliminate_noops=True,
            ),
        ),
        additional_config={
            "gemma4_kernel_experiment": "qkv-norm-rope-vnorm-kvcache-fusion"
        },
    )

    T = 5
    slot_mapping = torch.arange(T, dtype=torch.long, device="cuda")

    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = QKVNormRoPEVNormKVCacheTestModel(
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            eps=1e-6,
            is_neox=True,
            vllm_config=vllm_config,
            dtype=dtype,
            include_v_norm=False,
        )

        noop_pass = NoOpEliminationPass(vllm_config)
        coalesce_pass = SplitCoalescingPass(vllm_config)
        fusion_pass = QKNormRoPEFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, coalesce_pass, fusion_pass, cleanup_pass)

        qkv = torch.randn(T, model.q_size + 2 * model.kv_size)
        pos = torch.arange(T, dtype=torch.long, device=qkv.device)
        model.attn.kv_cache = _allocate_kv_cache(
            model.attn,
            block_size=vllm_config.cache_config.block_size,
            num_tokens=T,
            dtype=dtype,
            device=qkv.device,
        )

        with set_forward_context(
            None,
            vllm_config,
            slot_mapping={model.layer_name: slot_mapping},
        ):
            torch._dynamo.mark_dynamic(qkv, 0)
            torch._dynamo.mark_dynamic(pos, 0)
            model_fused = torch.compile(model, backend=backend)
            _ = model_fused(qkv, pos)

        assert fusion_pass.matched_count == 0
        assert backend.op_count(FUSED_QKV_ROPE_VNORM_KVCACHE_OP) == 0


def test_part4_engine_config_removes_unified_kv_cache_update_split():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        gemma4_kernel_experiment="qkv-norm-rope-vnorm-kvcache-fusion",
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "splitting_ops": [
                "vllm::unified_attention_with_output",
                "vllm::unified_kv_cache_update",
                "vllm::unified_mla_kv_cache_update",
            ],
        },
    )
    vllm_config = engine_args.create_engine_config()
    assert (
        "vllm::unified_kv_cache_update"
        not in vllm_config.compilation_config.splitting_ops
    )
    assert (
        "vllm::unified_mla_kv_cache_update"
        in vllm_config.compilation_config.splitting_ops
    )


def test_non_part4_engine_config_keeps_unified_kv_cache_update_split():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        gemma4_kernel_experiment="qkv-norm-rope-vnorm-fusion",
        compilation_config={
            "mode": CompilationMode.VLLM_COMPILE,
            "splitting_ops": [
                "vllm::unified_attention_with_output",
                "vllm::unified_kv_cache_update",
                "vllm::unified_mla_kv_cache_update",
            ],
        },
    )
    vllm_config = engine_args.create_engine_config()
    assert (
        "vllm::unified_kv_cache_update"
        in vllm_config.compilation_config.splitting_ops
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Part 4 compile-pass coverage is CUDA-only",
)
def test_qkv_norm_rope_vnorm_kvcache_does_not_match_flash_attention(dtype):
    if FUSED_QKV_ROPE_VNORM_KVCACHE_OP is None:
        pytest.skip("Part 4 fused post-GEMM custom op not available")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+rotary_embedding"],
            pass_config=PassConfig(
                enable_qk_norm_rope_fusion=True,
                eliminate_noops=True,
            ),
        ),
        additional_config={
            "gemma4_kernel_experiment": "qkv-norm-rope-vnorm-kvcache-fusion"
        },
    )

    T = 5
    slot_mapping = torch.arange(T, dtype=torch.long, device="cuda")

    with (
        set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        model = QKVNormRoPEVNormKVCacheTestModel(
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            eps=1e-6,
            is_neox=True,
            vllm_config=vllm_config,
            dtype=dtype,
            attn_backend=AttentionBackendEnum.FLASH_ATTN.get_class(),
        )

        noop_pass = NoOpEliminationPass(vllm_config)
        coalesce_pass = SplitCoalescingPass(vllm_config)
        fusion_pass = QKNormRoPEFusionPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        backend = TestBackend(noop_pass, coalesce_pass, fusion_pass, cleanup_pass)

        qkv = torch.randn(T, model.q_size + 2 * model.kv_size)
        pos = torch.arange(T, dtype=torch.long, device=qkv.device)
        model.attn.kv_cache = _allocate_kv_cache(
            model.attn,
            block_size=vllm_config.cache_config.block_size,
            num_tokens=T,
            dtype=dtype,
            device=qkv.device,
        )

        with set_forward_context(
            None,
            vllm_config,
            slot_mapping={model.layer_name: slot_mapping},
        ):
            torch._dynamo.mark_dynamic(qkv, 0)
            torch._dynamo.mark_dynamic(pos, 0)
            model_fused = torch.compile(model, backend=backend)
            _ = model_fused(qkv, pos)

        assert fusion_pass.matched_count == 0
        assert backend.op_count(FUSED_QKV_ROPE_VNORM_KVCACHE_OP) == 0
