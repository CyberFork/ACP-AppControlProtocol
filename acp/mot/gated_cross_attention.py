"""在 L 模型中后层注入 V 模型信息。

Gate 机制：初始化为 0，tanh(0)=0，训练初期不影响原始 LLM。
插入位置：L 模型第 8、12、16、20 层（32 层模型的 25%-63%）。

用法：
    block = GatedCrossAttentionBlock(d_lang=2048, heads=8)
    lang_hidden = block(lang_hidden, visual_tokens)  # 残差连接
"""

import torch
import torch.nn as nn


class GatedCrossAttentionBlock(nn.Module):
    """在语言模型隐层注入视觉信息的门控 cross-attention 块。

    核心设计：
    - `attn_gate` 和 `ff_gate` 初始化为 0，tanh(0)=0，训练初期输出为 0
    - Q=lang_hidden, K=V=visual_tokens（视觉信息单向注入语言流）
    - Pre-LN 风格：LayerNorm 在 attention 前，保持梯度稳定

    Args:
        d_lang: 语言模型 hidden size，默认 2048
        heads:  attention heads，默认 8
    """

    def __init__(self, d_lang: int = 2048, heads: int = 8) -> None:
        super().__init__()

        # 门控参数：标量，初始化为 0
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

        # Cross-Attention（Q=lang, K=V=visual）
        self.norm_q = nn.LayerNorm(d_lang)
        self.norm_kv = nn.LayerNorm(d_lang)
        self.cross_attn = nn.MultiheadAttention(d_lang, heads, batch_first=True)

        # FFN
        self.norm_ff = nn.LayerNorm(d_lang)
        self.ff = nn.Sequential(
            nn.Linear(d_lang, d_lang * 4),
            nn.GELU(),
            nn.Linear(d_lang * 4, d_lang),
        )

    def forward(
        self,
        lang_hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lang_hidden:   语言模型中间层输出 [B, T, D_l]
            visual_tokens: PerceiverResampler 输出 [B, num_latents, D_l]
        Returns:
            更新后的 lang_hidden [B, T, D_l]
        """
        # Cross-attention：语言 token 关注视觉 token
        attn_out, _ = self.cross_attn(
            self.norm_q(lang_hidden),
            self.norm_kv(visual_tokens),
            self.norm_kv(visual_tokens),
        )
        lang_hidden = lang_hidden + torch.tanh(self.attn_gate) * attn_out

        # FFN
        ff_out = self.ff(self.norm_ff(lang_hidden))
        lang_hidden = lang_hidden + torch.tanh(self.ff_gate) * ff_out

        return lang_hidden
