"""MoT 模块单元测试。

测试内容：
- 每个模块的 forward pass（随机 tensor，不加载真实模型）
- 输出 shape 正确
- gate 初始化为 0
- 冻结参数不更新
"""

import pytest
import torch
import torch.nn as nn

from acp.mot.config import MoTConfig
from acp.mot.perceiver import PerceiverResampler
from acp.mot.gated_cross_attention import GatedCrossAttentionBlock
from acp.mot.action_head import ActionHead
from acp.mot.action_projector import ActionProjector
from acp.mot.mot_model import ACPMoT


# ─────────────────────────── 通用常量 ───────────────────────────
B = 2        # batch size
T = 16       # token 序列长度
N = 8        # UI 元素数量
N_PATCHES = 49   # ViT patch 数（7×7）
D_V = 1152       # Qwen2-VL ViT 输出维度
D_L = 2048       # Qwen2.5-3B hidden size
D_ACTION = 512   # ActionProjector 输出维度


# ─────────────────────────── Mock 模型 ───────────────────────────

class MockVisionEncoder(nn.Module):
    """模拟 ViT：screenshot → visual_hidden。"""

    def __init__(self, d_visual: int = D_V) -> None:
        super().__init__()
        self.d_visual = d_visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B = x.size(0)
        return torch.randn(B, N_PATCHES, self.d_visual)


class MockDecoderLayer(nn.Module):
    """模拟单个 LLM decoder layer：透传 hidden_states。"""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (hidden_states,)


class MockLanguageModel(nn.Module):
    """模拟 HuggingFace Qwen2.5 风格的语言模型。"""

    def __init__(self, d_lang: int = D_L, num_layers: int = 4, vocab_size: int = 1000) -> None:
        super().__init__()
        # 模拟 model.embed_tokens, model.layers, model.norm
        self.model = _MockInnerLM(d_lang, num_layers, vocab_size)


class _MockInnerLM(nn.Module):
    def __init__(self, d_lang: int, num_layers: int, vocab_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_lang)
        self.layers = nn.ModuleList([MockDecoderLayer() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_lang)


def _make_elements(B: int, N: int, vocab_size: int = 8192) -> dict:
    """构造测试用元素 dict。"""
    return {
        "bbox": torch.rand(B, N, 4),
        "type": torch.rand(B, N, 16),
        "label_ids": torch.randint(0, vocab_size, (B, N)),
    }


# ─────────────────────────── Config ───────────────────────────

def test_config_defaults() -> None:
    cfg = MoTConfig()
    assert cfg.d_visual == 1152
    assert cfg.d_lang == 2048
    assert cfg.injection_layers == [8, 12, 16, 20]
    assert cfg.num_action_types == 5
    assert cfg.use_4bit is True


# ─────────────────────────── PerceiverResampler ───────────────────────────

class TestPerceiverResampler:

    def setup_method(self) -> None:
        self.model = PerceiverResampler(
            d_visual=D_V,
            d_lang=D_L,
            num_latents=64,
            depth=4,
            rank=16,
            num_heads=8,
        )

    def test_output_shape(self) -> None:
        visual_hidden = torch.randn(B, N_PATCHES, D_V)
        out = self.model(visual_hidden)
        assert out.shape == (B, 64, D_L), f"Expected ({B}, 64, {D_L}), got {out.shape}"

    def test_proj_b_init_zero(self) -> None:
        """proj_B 初始化为 0，训练初期视觉信号为 0。"""
        assert torch.all(self.model.proj_B.weight == 0), "proj_B.weight 应初始化为 0"

    def test_batch_size_1(self) -> None:
        visual_hidden = torch.randn(1, N_PATCHES, D_V)
        out = self.model(visual_hidden)
        assert out.shape == (1, 64, D_L)

    def test_different_patch_counts(self) -> None:
        """支持不同数量的 patch（可变输入长度）。"""
        for n_patches in [49, 196, 256]:
            visual_hidden = torch.randn(B, n_patches, D_V)
            out = self.model(visual_hidden)
            assert out.shape == (B, 64, D_L)

    def test_gradient_flows(self) -> None:
        """胶水层参数可以接收梯度。"""
        visual_hidden = torch.randn(B, N_PATCHES, D_V, requires_grad=False)
        out = self.model(visual_hidden)
        loss = out.sum()
        loss.backward()
        assert self.model.latents.grad is not None


# ─────────────────────────── GatedCrossAttentionBlock ───────────────────────

class TestGatedCrossAttentionBlock:

    def setup_method(self) -> None:
        self.block = GatedCrossAttentionBlock(d_lang=D_L, heads=8)

    def test_output_shape(self) -> None:
        lang_hidden = torch.randn(B, T, D_L)
        visual_tokens = torch.randn(B, 64, D_L)
        out = self.block(lang_hidden, visual_tokens)
        assert out.shape == (B, T, D_L), f"Expected ({B}, {T}, {D_L}), got {out.shape}"

    def test_gate_init_zero(self) -> None:
        """attn_gate 和 ff_gate 必须初始化为 0。"""
        assert self.block.attn_gate.item() == 0.0, "attn_gate 应初始化为 0"
        assert self.block.ff_gate.item() == 0.0, "ff_gate 应初始化为 0"

    def test_gate_zero_preserves_residual(self) -> None:
        """gate=0 时，tanh(0)=0，输入应通过残差完整保留（attn 和 FF 输出为 0）。

        注：由于 LayerNorm 等操作，输出不完全等于输入。
        但 attn_gate=0 时 cross-attn 贡献为 0，ff_gate=0 时 FFN 贡献为 0。
        """
        lang_hidden = torch.randn(B, T, D_L)
        visual_tokens = torch.randn(B, 64, D_L)
        with torch.no_grad():
            out = self.block(lang_hidden, visual_tokens)
        # gate=0 时输出应等于输入（残差连接 + 0 贡献）
        assert torch.allclose(out, lang_hidden, atol=1e-5), (
            "gate=0 时输出应与输入完全一致（tanh(0)=0 使注入为 0）"
        )

    def test_gradient_flows_through_gates(self) -> None:
        lang_hidden = torch.randn(B, T, D_L)
        visual_tokens = torch.randn(B, 64, D_L)
        out = self.block(lang_hidden, visual_tokens)
        out.sum().backward()
        assert self.block.attn_gate.grad is not None
        assert self.block.ff_gate.grad is not None


# ─────────────────────────── ActionProjector ───────────────────────────

class TestActionProjector:

    def setup_method(self) -> None:
        self.proj = ActionProjector(d_lang=D_L, d_action=D_ACTION, rank=8)

    def test_output_shape(self) -> None:
        last_token = torch.randn(B, D_L)
        out = self.proj(last_token)
        assert out.shape == (B, D_ACTION), f"Expected ({B}, {D_ACTION}), got {out.shape}"

    def test_proj_b_init_zero(self) -> None:
        assert torch.all(self.proj.proj_B.weight == 0), "proj_B.weight 应初始化为 0"

    def test_initial_output_is_zero(self) -> None:
        """proj_B 零初始化 → 初始输出为 0。"""
        last_token = torch.randn(B, D_L)
        with torch.no_grad():
            out = self.proj(last_token)
        assert torch.all(out == 0), "初始输出应为全 0（proj_B 零初始化）"


# ─────────────────────────── ActionHead ───────────────────────────

class TestActionHead:

    def setup_method(self) -> None:
        self.head = ActionHead(
            d_lang=D_ACTION,   # ActionProjector 输出维度
            d_action=256,
            num_action_types=5,
            max_elements=N,
            label_vocab_size=8192,
        )
        self.elements = _make_elements(B, N)

    def test_output_shapes(self) -> None:
        instruction_emb = torch.randn(B, D_ACTION)
        action_type, elem_scores, coord = self.head(instruction_emb, self.elements)
        assert action_type.shape == (B, 5), f"action_type: {action_type.shape}"
        assert elem_scores.shape == (B, N), f"element_scores: {elem_scores.shape}"
        assert coord.shape == (B, 2), f"coord_offset: {coord.shape}"

    def test_action_type_is_probability(self) -> None:
        instruction_emb = torch.randn(B, D_ACTION)
        action_type, _, _ = self.head(instruction_emb, self.elements)
        sums = action_type.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), "action_type 应为概率分布"

    def test_element_scores_is_probability(self) -> None:
        instruction_emb = torch.randn(B, D_ACTION)
        _, elem_scores, _ = self.head(instruction_emb, self.elements)
        sums = elem_scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), "element_scores 应为概率分布"

    def test_coord_offset_in_range(self) -> None:
        instruction_emb = torch.randn(B, D_ACTION)
        _, _, coord = self.head(instruction_emb, self.elements)
        assert coord.min() >= 0.0 and coord.max() <= 1.0, "coord_offset 应在 [0,1] 内"


# ─────────────────────────── ACPMoT (集成) ───────────────────────────

class TestACPMoT:

    def setup_method(self) -> None:
        # Mock V 和 L 模型（4 层，便于快速测试）
        self.vision_enc = MockVisionEncoder(d_visual=D_V)
        # 使用 4 层 mock LM，injection_layers 对应第 1、2、3 层
        self.lang_model = MockLanguageModel(d_lang=D_L, num_layers=4, vocab_size=1000)
        injection_layers = [1, 2, 3]

        perceiver = PerceiverResampler(
            d_visual=D_V, d_lang=D_L, num_latents=64, depth=2, rank=16, num_heads=8
        )
        cross_attn_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(d_lang=D_L, heads=8)
            for _ in injection_layers
        ])
        action_projector = ActionProjector(d_lang=D_L, d_action=D_ACTION, rank=8)
        action_head = ActionHead(
            d_lang=D_ACTION, d_action=256, num_action_types=5, max_elements=N
        )

        self.model = ACPMoT(
            vision_encoder=self.vision_enc,
            language_model=self.lang_model,
            action_head=action_head,
            perceiver=perceiver,
            cross_attn_blocks=cross_attn_blocks,
            action_projector=action_projector,
            injection_layers=injection_layers,
        )

    def _make_inputs(self):
        screenshot = torch.randn(B, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (B, T))
        elements = _make_elements(B, N)
        return screenshot, input_ids, elements

    def test_output_shapes(self) -> None:
        screenshot, input_ids, elements = self._make_inputs()
        out = self.model(screenshot, input_ids, elements)
        assert out["action_type"].shape == (B, 5)
        assert out["element_scores"].shape == (B, N)
        assert out["coord_offset"].shape == (B, 2)
        assert out["visual_tokens"].shape == (B, 64, D_L)

    def test_vision_encoder_frozen(self) -> None:
        """V 模型参数应被冻结（requires_grad=False）。"""
        for name, param in self.vision_enc.named_parameters():
            assert not param.requires_grad, f"vision_encoder.{name} 应被冻结"

    def test_language_model_frozen(self) -> None:
        """L 模型参数应被冻结（requires_grad=False）。"""
        for name, param in self.lang_model.named_parameters():
            assert not param.requires_grad, f"language_model.{name} 应被冻结"

    def test_glue_layers_trainable(self) -> None:
        """胶水层参数应可训练（requires_grad=True）。"""
        trainable_modules = {
            "perceiver": self.model.perceiver,
            "cross_attn_blocks": self.model.cross_attn_blocks,
            "action_projector": self.model.action_projector,
            "action_head": self.model.action_head,
        }
        for module_name, module in trainable_modules.items():
            for param_name, param in module.named_parameters():
                assert param.requires_grad, (
                    f"{module_name}.{param_name} 应可训练（requires_grad=True）"
                )

    def test_frozen_params_no_grad_update(self) -> None:
        """冻结参数在反向传播后不产生梯度。"""
        screenshot, input_ids, elements = self._make_inputs()
        out = self.model(screenshot, input_ids, elements)
        loss = out["action_type"].sum() + out["coord_offset"].sum()
        loss.backward()

        # V 和 L 的参数不应有梯度
        for param in self.vision_enc.parameters():
            assert param.grad is None, "视觉编码器参数不应有梯度"
        for param in self.lang_model.parameters():
            assert param.grad is None, "语言模型参数不应有梯度"

    def test_injection_layer_count_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            ACPMoT(
                vision_encoder=MockVisionEncoder(),
                language_model=MockLanguageModel(num_layers=4),
                action_head=ActionHead(d_lang=D_ACTION),
                perceiver=PerceiverResampler(),
                cross_attn_blocks=nn.ModuleList([GatedCrossAttentionBlock()]),
                action_projector=ActionProjector(),
                injection_layers=[1, 2],  # 2 层但只有 1 个 block，应报错
            )

    def test_from_config(self) -> None:
        """from_config 工厂方法可以正确构建模型。"""
        cfg = MoTConfig(
            d_visual=D_V,
            d_lang=D_L,
            num_latents=64,
            perceiver_depth=2,
            num_layers=4,
            injection_layers=[1, 2, 3],
            d_action=D_ACTION,
            label_vocab_size=8192,
        )
        # 使用 4 层 mock LM（injection_layers 内的索引需 < num_layers）
        lang_model = MockLanguageModel(d_lang=D_L, num_layers=4)
        model = ACPMoT.from_config(cfg, MockVisionEncoder(d_visual=D_V), lang_model)

        screenshot, input_ids, elements = self._make_inputs()
        out = model(screenshot, input_ids, elements)
        assert out["action_type"].shape == (B, 5)


# ─────────────────────────── 参数量验证 ───────────────────────────

class TestParameterCount:

    def test_perceiver_has_parameters(self) -> None:
        model = PerceiverResampler(d_visual=D_V, d_lang=D_L, num_latents=64, depth=4)
        n_params = sum(p.numel() for p in model.parameters())
        # 实际参数量远大于 67M（d_lang=2048, depth=4），约 200M+
        # 此处只验证参数量在合理范围内（>50M）
        assert n_params > 50_000_000, f"Perceiver 参数量异常偏低: {n_params:,}"

    def test_cross_attn_block_has_parameters(self) -> None:
        block = GatedCrossAttentionBlock(d_lang=D_L, heads=8)
        n_params = sum(p.numel() for p in block.parameters())
        # 单个 block: MHA(~16M) + FFN(~33M) + LayerNorms + gates ≈ 50M
        assert n_params > 40_000_000, f"CrossAttnBlock 参数量异常偏低: {n_params:,}"

    def test_action_projector_param_count(self) -> None:
        proj = ActionProjector(d_lang=D_L, d_action=D_ACTION, rank=8)
        n_params = sum(p.numel() for p in proj.parameters())
        expected = D_L * 8 + 8 * D_ACTION  # proj_A + proj_B = 20,480
        assert n_params == expected, f"ActionProjector 参数量: {n_params}, 期望: {expected}"

    def test_action_head_has_parameters(self) -> None:
        head = ActionHead(d_lang=D_ACTION, d_action=256, num_action_types=5)
        n_params = sum(p.numel() for p in head.parameters())
        # label_embedding(8192*384) + element_encoder + proj + heads ≈ 3M+
        assert n_params > 3_000_000, f"ActionHead 参数量异常偏低: {n_params:,}"
