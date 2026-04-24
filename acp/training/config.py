"""训练配置 — TrainingConfig dataclass。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """MoT 三阶段训练配置。

    三个训练阶段：
      Stage 1: V-L 对齐 — 只训练 PerceiverResampler（冻结 V/L 基座）
      Stage 2: UI 融合 — 训练 Perceiver + GatedCrossAttn（胶水层）
      Stage 3: 端到端 — 训练全部胶水层 + ActionHead
    """

    # ── 基础 ─────────────────────────────────────────────────────────────────
    output_dir: str = "models/adapters/"
    seed: int = 42

    # ── 阶段控制 ─────────────────────────────────────────────────────────────
    stage: int = 1  # 1=对齐, 2=融合, 3=端到端

    # ── Stage 1: V-L 对齐 ────────────────────────────────────────────────────
    stage1_data: str = ""          # 图文对数据集路径
    stage1_epochs: int = 3
    stage1_lr: float = 1e-3
    stage1_trainable: list = field(
        default_factory=lambda: ["perceiver"]
    )

    # ── Stage 2: UI 融合 ─────────────────────────────────────────────────────
    stage2_data: str = ""
    stage2_epochs: int = 5
    stage2_lr: float = 2e-4
    stage2_trainable: list = field(
        default_factory=lambda: ["perceiver", "cross_attn"]
    )

    # ── Stage 3: 端到端 ───────────────────────────────────────────────────────
    stage3_data: str = ""
    stage3_epochs: int = 3
    stage3_lr: float = 1e-4
    stage3_trainable: list = field(
        default_factory=lambda: [
            "perceiver", "cross_attn", "action_head", "action_projector"
        ]
    )

    # ── QLoRA ────────────────────────────────────────────────────────────────
    qlora_rank: int = 16
    qlora_alpha: int = 32
    use_4bit: bool = True

    # ── 训练超参 ─────────────────────────────────────────────────────────────
    batch_size: int = 4
    gradient_accumulation: int = 8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    fp16: bool = True
    gradient_checkpointing: bool = True

    # ── 硬件 ─────────────────────────────────────────────────────────────────
    device: str = "cuda"

    # ── 便利属性 ─────────────────────────────────────────────────────────────

    @property
    def effective_batch_size(self) -> int:
        """实际批大小（batch_size × gradient_accumulation）。"""
        return self.batch_size * self.gradient_accumulation

    @property
    def stage_data(self) -> str:
        """当前阶段的数据路径。"""
        return {1: self.stage1_data, 2: self.stage2_data, 3: self.stage3_data}[self.stage]

    @property
    def stage_epochs(self) -> int:
        """当前阶段的训练轮数。"""
        return {1: self.stage1_epochs, 2: self.stage2_epochs, 3: self.stage3_epochs}[self.stage]

    @property
    def stage_lr(self) -> float:
        """当前阶段的学习率。"""
        return {1: self.stage1_lr, 2: self.stage2_lr, 3: self.stage3_lr}[self.stage]

    @property
    def stage_trainable(self) -> list:
        """当前阶段可训练的模块名列表。"""
        return {
            1: self.stage1_trainable,
            2: self.stage2_trainable,
            3: self.stage3_trainable,
        }[self.stage]
