"""MoTTrainer — MoT 三阶段 QLoRA 训练器。

三阶段训练策略（来自 research-mot-glue 调研）：
  Stage 1: V-L 对齐   — 只解冻 PerceiverResampler，大学习率暖身
  Stage 2: 胶水层融合  — 解冻 Perceiver + GatedCrossAttn，中学习率
  Stage 3: 端到端微调  — 解冻全部胶水层 + ActionHead，小学习率

Flamingo 风格：gate 初始为 0（tanh(0)=0），确保训练初期不破坏预训练权重。
只保存可训练参数（不存冻结的 V/L 基座），checkpoint 体积 < 100MB。
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from acp.training.config import TrainingConfig

if TYPE_CHECKING:
    # 避免 import 时强依赖 torch
    pass

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # stub，让类定义不报错
    class AdamW:  # type: ignore[no-redef]
        pass
    class LambdaLR:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _cosine_schedule_with_warmup(
    optimizer: Any,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Any:
    """Cosine LR 调度器（带 warmup）。"""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# MoTTrainer
# ---------------------------------------------------------------------------

class MoTTrainer:
    """MoT 三阶段训练器。

    用法：
        trainer = MoTTrainer(config, model)
        trainer.train_stage1()   # V-L 对齐
        trainer.train_stage2()   # 胶水层融合
        trainer.train_stage3()   # 端到端

        # 或一次性完成
        trainer.train_all()
    """

    # 模块名前缀 → 对应 config.stage_trainable 中的 key
    _MODULE_PREFIXES = {
        "perceiver":        ["perceiver_resampler", "perceiver"],
        "cross_attn":       ["gated_cross_attn", "cross_attn"],
        "action_head":      ["action_head"],
        "action_projector": ["action_projector"],
    }

    def __init__(self, config: TrainingConfig, model: Any):
        self.config = config
        self.model = model

        # 懒初始化（每个 stage 调用时重建）
        self._optimizer: Optional[Any] = None
        self._scheduler: Optional[Any] = None

    # ── 公共训练接口 ─────────────────────────────────────────────────────────

    def train_stage1(self) -> None:
        """Stage 1：训练 PerceiverResampler（V-L 对齐）。"""
        logger.info("=== Stage 1: V-L 对齐 ===")
        self.config.stage = 1
        self._run_stage(
            data_path=self.config.stage1_data,
            epochs=self.config.stage1_epochs,
            lr=self.config.stage1_lr,
            trainable_keys=self.config.stage1_trainable,
        )

    def train_stage2(self) -> None:
        """Stage 2：训练 Perceiver + GatedCrossAttn（胶水层融合）。"""
        logger.info("=== Stage 2: UI 融合 ===")
        self.config.stage = 2
        self._run_stage(
            data_path=self.config.stage2_data,
            epochs=self.config.stage2_epochs,
            lr=self.config.stage2_lr,
            trainable_keys=self.config.stage2_trainable,
        )

    def train_stage3(self) -> None:
        """Stage 3：训练全部胶水层 + ActionHead（端到端）。"""
        logger.info("=== Stage 3: 端到端 ===")
        self.config.stage = 3
        self._run_stage(
            data_path=self.config.stage3_data,
            epochs=self.config.stage3_epochs,
            lr=self.config.stage3_lr,
            trainable_keys=self.config.stage3_trainable,
        )

    def train_all(self) -> None:
        """连续完成三个阶段训练。"""
        self.train_stage1()
        self.train_stage2()
        self.train_stage3()

    # ── 冻结 / 解冻 ──────────────────────────────────────────────────────────

    def _freeze_for_stage(self, trainable_keys: list[str]) -> None:
        """按阶段冻结/解冻参数。

        规则：
        - 先将全部参数冻结（requires_grad=False）
        - 再按 trainable_keys 对应的模块名前缀解冻

        Args:
            trainable_keys: 来自 TrainingConfig.stage_trainable 的模块 key 列表
                            如 ["perceiver", "cross_attn"]
        """
        if not _TORCH_AVAILABLE:
            logger.debug("_freeze_for_stage: torch 不可用，跳过（CPU mock 模式）")
            return

        # 解析需要解冻的参数名前缀
        unfreeze_prefixes: list[str] = []
        for key in trainable_keys:
            unfreeze_prefixes.extend(self._MODULE_PREFIXES.get(key, [key]))

        frozen_count = 0
        unfrozen_count = 0

        for name, param in self.model.named_parameters():
            should_train = any(name.startswith(pfx) for pfx in unfreeze_prefixes)
            param.requires_grad = should_train
            if should_train:
                unfrozen_count += 1
            else:
                frozen_count += 1

        logger.info(
            "参数冻结完成：%d 冻结，%d 可训练（解冻前缀：%s）",
            frozen_count, unfrozen_count, unfreeze_prefixes,
        )

    def _build_optimizer(self, lr: float) -> Any:
        """构建 AdamW 优化器（只包含可训练参数）。"""
        if not _TORCH_AVAILABLE:
            return None

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning("没有可训练参数，优化器为空")
            return None

        return AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    def _build_scheduler(
        self, optimizer: Any, num_steps: int
    ) -> Any:
        """构建 cosine LR 调度器（带 warmup）。"""
        if not _TORCH_AVAILABLE or optimizer is None:
            return None

        warmup_steps = max(1, int(num_steps * self.config.warmup_ratio))
        return _cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)

    # ── 内部训练循环 ─────────────────────────────────────────────────────────

    def _run_stage(
        self,
        data_path: str,
        epochs: int,
        lr: float,
        trainable_keys: list[str],
    ) -> None:
        """内部：执行单阶段训练。"""
        self._freeze_for_stage(trainable_keys)

        if not data_path:
            logger.warning("data_path 为空，跳过训练（stage=%d）", self.config.stage)
            return

        from acp.training.data_loader import UITrajectoryDataset

        # 加载数据集
        dataset = UITrajectoryDataset.from_custom(data_path)
        if len(dataset) == 0:
            logger.warning("数据集为空，跳过训练")
            return

        if not _TORCH_AVAILABLE:
            logger.info("torch 不可用，CPU mock 模式——跳过实际训练循环")
            return

        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
        )

        num_steps = len(loader) * epochs
        self._optimizer = self._build_optimizer(lr)
        self._scheduler = self._build_scheduler(self._optimizer, num_steps)

        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            self._optimizer.zero_grad()

            for step, batch in enumerate(loader):
                loss = self._forward_step(batch)

                if self.config.fp16:
                    # 实际使用时应配合 torch.cuda.amp.GradScaler
                    loss = loss / self.config.gradient_accumulation
                    loss.backward()
                else:
                    (loss / self.config.gradient_accumulation).backward()

                epoch_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        logger.info(
                            "Stage %d | epoch %d/%d | step %d | loss %.4f",
                            self.config.stage, epoch + 1, epochs,
                            global_step, epoch_loss / (step + 1),
                        )

            logger.info("Stage %d epoch %d/%d 完成，avg_loss=%.4f",
                        self.config.stage, epoch + 1, epochs,
                        epoch_loss / max(1, len(loader)))

    def _forward_step(self, batch: Any) -> Any:
        """单步前向（由调用方传入实际模型时覆盖/子类化）。

        默认实现：直接调用 model(batch)，期望返回包含 loss 的对象。
        """
        output = self.model(batch)
        if hasattr(output, "loss"):
            return output.loss
        return output

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """只保存可训练参数（不保存冻结的 V/L 基座）。

        保存格式：
            {
                "config": <TrainingConfig.__dict__>,
                "state_dict": {name: tensor, ...},  # 只含 requires_grad=True 的参数
            }
        """
        if not _TORCH_AVAILABLE:
            logger.info("torch 不可用，保存 mock checkpoint（JSON）")
            import json
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"config": self.config.__dict__, "state_dict": {}}, f, indent=2, default=str)
            return

        import torch

        trainable_state = {
            name: param.data
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        checkpoint = {
            "config": self.config.__dict__,
            "state_dict": trainable_state,
        }

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(
            "Checkpoint 已保存：%s（%d 个可训练参数张量）",
            save_path, len(trainable_state),
        )

    def load_checkpoint(self, path: str) -> None:
        """加载胶水层权重（严格模式：只加载当前模型中存在的键）。"""
        if not _TORCH_AVAILABLE:
            logger.info("torch 不可用，跳过 checkpoint 加载")
            return

        import torch

        checkpoint = torch.load(path, map_location=self.config.device)
        state_dict = checkpoint.get("state_dict", {})

        current_params = dict(self.model.named_parameters())
        loaded, skipped = 0, 0

        for name, tensor in state_dict.items():
            if name in current_params:
                current_params[name].data.copy_(tensor)
                loaded += 1
            else:
                logger.debug("跳过不存在的参数：%s", name)
                skipped += 1

        logger.info("Checkpoint 加载完成：%d 加载，%d 跳过（路径：%s）", loaded, skipped, path)
