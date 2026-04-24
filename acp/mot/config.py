"""MoT 模型配置。"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MoTConfig:
    # V model
    vision_model: str = "showlab/ShowUI-2B"
    d_visual: int = 1152  # Qwen2-VL ViT 输出维度

    # L model
    language_model: str = "Qwen/Qwen2.5-3B-Instruct"
    d_lang: int = 2048  # Qwen2.5-3B hidden size
    num_layers: int = 32

    # Perceiver
    num_latents: int = 64
    perceiver_depth: int = 4
    perceiver_rank: int = 16
    perceiver_heads: int = 8

    # Cross-Attention injection
    injection_layers: List[int] = field(default_factory=lambda: [8, 12, 16, 20])
    num_heads: int = 8

    # Action
    d_action: int = 512
    action_proj_rank: int = 8
    num_action_types: int = 5  # click, type, scroll, press_key, wait
    max_elements: int = 64
    label_vocab_size: int = 8192  # 元素标签词表大小

    # QLoRA
    qlora_rank: int = 16
    qlora_alpha: int = 32
    use_4bit: bool = True
