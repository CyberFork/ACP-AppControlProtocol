"""MoT (Mixture of Transformers) 融合层。

将 V（视觉）、L（语言）、A（动作）三个专家模型通过 QLoRA 风格的胶水层连接：

    V (ShowUI-2B ViT) → PerceiverResampler → visual_tokens
    L (Qwen2.5-3B) ← GatedCrossAttentionBlock × 4 (注入视觉信息)
    L 最后 token → ActionProjector → ActionHead → action
"""

from .config import MoTConfig
from .perceiver import PerceiverResampler
from .gated_cross_attention import GatedCrossAttentionBlock
from .action_head import ActionHead
from .action_projector import ActionProjector
from .mot_model import ACPMoT

__all__ = [
    "MoTConfig",
    "PerceiverResampler",
    "GatedCrossAttentionBlock",
    "ActionHead",
    "ActionProjector",
    "ACPMoT",
]
