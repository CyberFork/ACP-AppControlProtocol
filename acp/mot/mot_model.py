"""MoT 模型：组装 V + L + 胶水层 + A。

架构流程：
    1. V (ViT):  screenshot → visual_hidden [B, N_patches, D_v]
    2. Perceiver: visual_hidden → visual_tokens [B, 64, D_l]
    3. L (LLM):  instruction token → 逐层前向，在 injection_layers 注入 cross_attn
    4. ActionProjector: L 最后 token hidden → action embedding [B, D_action]
    5. ActionHead: action_embedding + elements → (action_type, element_scores, coord)

冻结策略：
    - V（ViT）和 L（LLM）参数冻结（requires_grad=False）
    - 只训练胶水层：PerceiverResampler、GatedCrossAttentionBlock × N、ActionProjector、ActionHead

语言模型接口约定（HuggingFace Qwen2.5 兼容）：
    language_model.model.embed_tokens: token embedding
    language_model.model.layers: list[DecoderLayer]（每层接受 (hidden, ...) 返回 tuple）
    language_model.model.norm: final LayerNorm

    对于测试/mock，只需实现 get_lm_layers() 接口（见下方）。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .perceiver import PerceiverResampler
from .gated_cross_attention import GatedCrossAttentionBlock
from .action_head import ActionHead
from .action_projector import ActionProjector


class ACPMoT(nn.Module):
    """MoT 多模态统一模型。

    Args:
        vision_encoder:    ViT 编码器（调用 forward 返回 hidden states）
        language_model:    语言模型（HuggingFace 兼容接口）
        action_head:       动作预测头
        perceiver:         PerceiverResampler
        cross_attn_blocks: GatedCrossAttentionBlock 列表，与 injection_layers 一一对应
        action_projector:  ActionProjector
        injection_layers:  在 L 的哪些层之后注入视觉信息（默认 [8, 12, 16, 20]）
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        language_model: nn.Module,
        action_head: ActionHead,
        perceiver: PerceiverResampler,
        cross_attn_blocks: nn.ModuleList,
        action_projector: ActionProjector,
        injection_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if injection_layers is None:
            injection_layers = [8, 12, 16, 20]

        assert len(cross_attn_blocks) == len(injection_layers), (
            f"cross_attn_blocks 数量 ({len(cross_attn_blocks)}) "
            f"必须与 injection_layers 数量 ({len(injection_layers)}) 一致"
        )

        self.injection_layers = sorted(injection_layers)
        self._injection_layer_set = set(self.injection_layers)

        # 冻结 V 和 L 的参数
        vision_encoder.requires_grad_(False)
        language_model.requires_grad_(False)

        # 可训练的胶水层（不冻结）
        self.perceiver = perceiver
        self.cross_attn_blocks = cross_attn_blocks
        self.action_projector = action_projector
        self.action_head = action_head

        # 冻结的基础模型（注册为 buffer-like，不参与梯度计算）
        self.vision_encoder = vision_encoder
        self.language_model = language_model

        # 构建 injection_layer → cross_attn_block 的映射
        self._layer_to_block = {
            layer_idx: block
            for layer_idx, block in zip(self.injection_layers, cross_attn_blocks)
        }

    def _get_lm_components(self):
        """提取语言模型内部组件（兼容 HuggingFace 和 mock 模型）。

        HuggingFace Qwen2.5:
            embed_tokens = lm.model.embed_tokens
            layers = lm.model.layers
            norm = lm.model.norm

        Mock 模型需实现 .embed_tokens, .layers, .norm 属性。
        """
        lm = self.language_model
        # HuggingFace 风格
        if hasattr(lm, "model"):
            inner = lm.model
            return inner.embed_tokens, inner.layers, inner.norm
        # mock/自定义风格：直接在 lm 上找
        return lm.embed_tokens, lm.layers, lm.norm

    @torch.no_grad()
    def _encode_visual(self, screenshot: torch.Tensor) -> torch.Tensor:
        """截图 → visual_hidden（冻结）。

        Args:
            screenshot: [B, C, H, W] 归一化图像张量
        Returns:
            visual_hidden: [B, N_patches, D_v]
        """
        return self.vision_encoder(screenshot)

    def forward(
        self,
        screenshot: torch.Tensor,
        input_ids: torch.Tensor,
        elements: dict,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """端到端前向传播。

        Args:
            screenshot:      截图 [B, C, H, W]
            input_ids:       指令 token ids [B, T]
            elements:        UI 元素特征 dict，见 ActionHead 文档
            attention_mask:  optional [B, T]

        Returns:
            dict with:
                'action_type':    [B, num_action_types]
                'element_scores': [B, N]
                'coord_offset':   [B, 2]
                'visual_tokens':  [B, 64, D_l]  — 中间产物，供调试
        """
        # ── Step 1: V → visual_hidden ──────────────────────────────────────
        with torch.no_grad():
            visual_hidden = self.vision_encoder(screenshot)  # [B, N_patches, D_v]

        # ── Step 2: Perceiver → visual_tokens ──────────────────────────────
        visual_tokens = self.perceiver(visual_hidden)  # [B, 64, D_l]

        # ── Step 3: L 逐层前向 + cross_attn 注入 ───────────────────────────
        embed_tokens, lm_layers, lm_norm = self._get_lm_components()

        with torch.no_grad():
            hidden = embed_tokens(input_ids)  # [B, T, D_l]

        for layer_idx, layer in enumerate(lm_layers):
            with torch.no_grad():
                # HuggingFace decoder layer 返回 tuple，第一个元素是 hidden_states
                layer_out = layer(hidden, attention_mask=attention_mask)
                hidden = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

            # 在指定层之后注入视觉信息（胶水层有梯度）
            if layer_idx in self._injection_layer_set:
                block = self._layer_to_block[layer_idx]
                hidden = block(hidden, visual_tokens)  # 有梯度

        with torch.no_grad():
            hidden = lm_norm(hidden)  # [B, T, D_l]

        # ── Step 4: 取最后 token → ActionProjector ─────────────────────────
        last_token = hidden[:, -1, :]        # [B, D_l]
        action_emb = self.action_projector(last_token)  # [B, D_action]

        # ── Step 5: ActionHead → 动作预测 ───────────────────────────────────
        action_type, element_scores, coord_offset = self.action_head(
            action_emb, elements
        )

        return {
            "action_type": action_type,
            "element_scores": element_scores,
            "coord_offset": coord_offset,
            "visual_tokens": visual_tokens,
        }

    @classmethod
    def from_config(cls, config, vision_encoder: nn.Module, language_model: nn.Module) -> "ACPMoT":
        """从 MoTConfig 构建模型（加载真实预训练模型后调用）。

        Args:
            config:          MoTConfig 实例
            vision_encoder:  已加载的 ViT 模型
            language_model:  已加载的语言模型
        """
        from .config import MoTConfig
        assert isinstance(config, MoTConfig)

        perceiver = PerceiverResampler(
            d_visual=config.d_visual,
            d_lang=config.d_lang,
            num_latents=config.num_latents,
            depth=config.perceiver_depth,
            rank=config.perceiver_rank,
            num_heads=config.perceiver_heads,
        )

        cross_attn_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(d_lang=config.d_lang, heads=config.num_heads)
            for _ in config.injection_layers
        ])

        action_projector = ActionProjector(
            d_lang=config.d_lang,
            d_action=config.d_action,
            rank=config.action_proj_rank,
        )

        action_head = ActionHead(
            d_lang=config.d_action,  # ActionProjector 输出维度
            d_action=256,
            num_action_types=config.num_action_types,
            max_elements=config.max_elements,
            label_vocab_size=config.label_vocab_size,
        )

        return cls(
            vision_encoder=vision_encoder,
            language_model=language_model,
            action_head=action_head,
            perceiver=perceiver,
            cross_attn_blocks=cross_attn_blocks,
            action_projector=action_projector,
            injection_layers=config.injection_layers,
        )
