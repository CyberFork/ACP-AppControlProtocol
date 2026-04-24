"""V→L 空间映射：将 ViT 输出的 visual_hidden 压缩为固定长度 visual_tokens。

输入: visual_hidden [B, N_patches, D_v]  (如 [B, 196, 1152])
输出: visual_tokens [B, num_latents, D_l]  (如 [B, 64, 2048])

核心：64 个 learnable queries 通过 cross-attention 从 ViT 输出中提取信息。
维度映射用 QLoRA 风格低秩投影（rank=16），proj_B 初始化为 0 确保训练初期
视觉信号不影响模型（渐进式接入）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PerceiverLayer(nn.Module):
    """单层 Perceiver：latents × visual_features 的 cross-attention + FFN。"""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        # FFN 维度保持 d_model（不做 4x 扩展以控制参数量）
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  latent queries [B, num_latents, D]
            kv: projected visual features [B, N_patches, D]
        Returns:
            updated latents [B, num_latents, D]
        """
        # Cross-attention（Pre-LN 风格）
        attn_out, _ = self.cross_attn(
            self.norm_q(x),
            self.norm_kv(kv),
            self.norm_kv(kv),
        )
        x = x + attn_out

        # FFN
        x = x + self.ff(self.norm_ff(x))
        return x


class PerceiverResampler(nn.Module):
    """将可变长度 ViT 特征压缩为固定长度 visual_tokens。

    QLoRA 风格低秩投影将视觉空间（D_v）映射到语言空间（D_l）：
      proj_A: D_v → rank  (随机初始化)
      proj_B: rank → D_l  (零初始化，训练初期无视觉信号)

    Args:
        d_visual:    ViT 输出维度，默认 1152（Qwen2-VL ViT）
        d_lang:      语言模型 hidden size，默认 2048（Qwen2.5-3B）
        num_latents: 输出 token 数，默认 64
        depth:       cross-attention 层数，默认 4
        rank:        低秩投影秩，默认 16
        num_heads:   attention heads，默认 8
    """

    def __init__(
        self,
        d_visual: int = 1152,
        d_lang: int = 2048,
        num_latents: int = 64,
        depth: int = 4,
        rank: int = 16,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.d_lang = d_lang
        self.num_latents = num_latents

        # Learnable latent queries（语言空间）
        self.latents = nn.Parameter(torch.randn(num_latents, d_lang) * 0.02)

        # QLoRA 风格低秩投影：视觉 → 语言空间
        self.proj_A = nn.Linear(d_visual, rank, bias=False)
        self.proj_B = nn.Linear(rank, d_lang, bias=False)
        nn.init.zeros_(self.proj_B.weight)  # 零初始化，训练初期无视觉信号

        # Perceiver 层
        self.layers = nn.ModuleList(
            [_PerceiverLayer(d_lang, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_lang)

    def forward(self, visual_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_hidden: ViT 输出 [B, N_patches, D_v]
        Returns:
            visual_tokens: 压缩后的视觉 token [B, num_latents, D_l]
        """
        B = visual_hidden.size(0)

        # 低秩投影：视觉空间 → 语言空间
        kv = self.proj_B(self.proj_A(visual_hidden))  # [B, N_patches, D_l]

        # 扩展 latent queries 到 batch
        x = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, num_latents, D_l]

        # 逐层 cross-attention
        for layer in self.layers:
            x = layer(x, kv)

        return self.norm(x)  # [B, num_latents, D_l]
