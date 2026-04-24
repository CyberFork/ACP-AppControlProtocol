"""一键训练脚本 — MoT QLoRA 三阶段训练 CLI。

用法：
    # Stage 1（V-L 对齐）
    python -m acp.training.train --stage 1 --data datasets/screenspot/ --output models/adapters/web/v1/

    # Stage 2（UI 融合，从 Stage 1 checkpoint 继续）
    python -m acp.training.train --stage 2 --data datasets/aitw/ \\
        --checkpoint models/adapters/web/v1/stage1/checkpoint.pt \\
        --output models/adapters/web/v1/

    # Stage 3（端到端）
    python -m acp.training.train --stage 3 --data datasets/custom/ \\
        --checkpoint models/adapters/web/v1/stage2/checkpoint.pt \\
        --output models/adapters/web/v1/

    # 全部 3 阶段连续执行
    python -m acp.training.train --stage all --data datasets/ --output models/adapters/web/v1/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m acp.training.train",
        description="MoT QLoRA 三阶段训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 必填 ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        required=True,
        help="训练阶段：1=V-L对齐, 2=UI融合, 3=端到端, all=连续三阶段",
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="DIR_OR_FILE",
        help="训练数据路径（目录或 JSONL 文件）",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="输出目录（adapter 权重、registry 更新）",
    )

    # ── 可选：继续训练 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint",
        default="",
        metavar="FILE",
        help="从此 checkpoint 继续训练（.pt 文件）",
    )

    # ── 超参覆盖 ─────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（覆盖 config 默认值）")
    parser.add_argument("--lr", type=float, default=None, help="学习率（覆盖 config 默认值）")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--qlora-rank", type=int, default=None, help="QLoRA rank")
    parser.add_argument("--no-4bit", action="store_true", help="禁用 4bit 量化（用更多显存）")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"], help="训练设备")

    # ── 模型 ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--base-model",
        default="showui-2b",
        help="基座模型名称或路径（默认：showui-2b）",
    )

    # ── 日志 ─────────────────────────────────────────────────────────────────
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    parser.add_argument("--dry-run", action="store_true", help="只打印配置，不实际训练")

    return parser


# ---------------------------------------------------------------------------
# 配置构建
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace):
    """根据命令行参数构建 TrainingConfig。"""
    from acp.training.config import TrainingConfig

    stage_int = int(args.stage) if args.stage != "all" else 1

    cfg = TrainingConfig(
        output_dir=args.output,
        stage=stage_int,
    )

    # 数据路径——同一份数据用于所有阶段（all 模式）
    data = args.data
    cfg.stage1_data = data
    cfg.stage2_data = data
    cfg.stage3_data = data

    # 超参覆盖
    if args.epochs is not None:
        cfg.stage1_epochs = args.epochs
        cfg.stage2_epochs = args.epochs
        cfg.stage3_epochs = args.epochs
    if args.lr is not None:
        cfg.stage1_lr = args.lr
        cfg.stage2_lr = args.lr
        cfg.stage3_lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.gradient_accumulation = args.grad_accum
    if args.qlora_rank is not None:
        cfg.qlora_rank = args.qlora_rank
    if args.no_4bit:
        cfg.use_4bit = False
    if args.device is not None:
        cfg.device = args.device

    return cfg


# ---------------------------------------------------------------------------
# 模型加载（stub，Phase 4 集成 ShowUI 时替换）
# ---------------------------------------------------------------------------

def _load_model(base_model: str, config, checkpoint: str = ""):
    """加载 / 构建模型。

    当前为 stub：返回一个最小 mock 模型，便于 --help 和 --dry-run 测试。
    Phase 4 实现时替换为实际的 ACPMoT 模型加载逻辑。
    """
    try:
        import torch.nn as nn

        class _MockModel(nn.Module):
            """占位模型，仅用于 CI 和 --dry-run 验证。"""
            def __init__(self):
                super().__init__()
                import torch
                self.perceiver = nn.Linear(4, 4)
                self.cross_attn = nn.Linear(4, 4)
                self.action_head = nn.Linear(4, 4)
                self.action_projector = nn.Linear(4, 4)

            def forward(self, batch):
                import torch
                return type("Output", (), {"loss": torch.tensor(0.0, requires_grad=True)})()

        model = _MockModel()
        logger.info("加载 mock 模型（base_model=%s）—— Phase 4 替换为真实 ACPMoT", base_model)
        return model

    except ImportError:
        # 无 torch 环境
        logger.info("torch 不可用，使用 None 模型（CPU mock 模式）")
        return None


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 构建配置
    config = _build_config(args)

    # 打印配置摘要
    print("=" * 60)
    print(f"  ACP MoT 训练")
    print(f"  阶段:      {args.stage}")
    print(f"  数据:      {args.data}")
    print(f"  输出目录:  {args.output}")
    print(f"  基座模型:  {args.base_model}")
    print(f"  QLoRA:     rank={config.qlora_rank}, alpha={config.qlora_alpha}, 4bit={config.use_4bit}")
    print(f"  批大小:    {config.batch_size} × 梯度累积 {config.gradient_accumulation} = {config.effective_batch_size}")
    print(f"  设备:      {config.device}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    if args.dry_run:
        print("[dry-run] 配置验证通过，不执行实际训练。")
        return 0

    # 加载模型
    model = _load_model(args.base_model, config)

    from acp.training.trainer import MoTTrainer
    trainer = MoTTrainer(config, model)

    # 加载 checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"错误：checkpoint 不存在：{args.checkpoint}", file=sys.stderr)
            return 1
        trainer.load_checkpoint(str(checkpoint_path))

    # 确保输出目录存在
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 执行训练
    try:
        if args.stage == "all":
            trainer.train_all()
            # 保存最终 checkpoint
            final_ckpt = Path(args.output) / "stage3" / "checkpoint.pt"
            trainer.save_checkpoint(str(final_ckpt))
        else:
            stage_int = int(args.stage)
            stage_methods = {
                1: trainer.train_stage1,
                2: trainer.train_stage2,
                3: trainer.train_stage3,
            }
            stage_methods[stage_int]()

            ckpt_path = Path(args.output) / f"stage{stage_int}" / "checkpoint.pt"
            trainer.save_checkpoint(str(ckpt_path))
            print(f"Checkpoint 已保存：{ckpt_path}")

    except KeyboardInterrupt:
        print("\n训练被用户中断。")
        return 130
    except Exception as e:
        logger.exception("训练失败：%s", e)
        return 1

    print("训练完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
