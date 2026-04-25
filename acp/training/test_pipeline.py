"""训练管道 CPU mock 验证。

验证流程：构造 5 条假数据 → UITrajectoryDataset 加载 → MoTTrainer 跑 1 个 batch
（CPU，dummy 模型）→ 梯度流通 → loss 下降。

用法：
    python -m acp.training.test_pipeline
"""

from __future__ import annotations

import json
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from acp.training.config import TrainingConfig
from acp.training.data_loader import UITrajectoryDataset
from acp.training.trainer import MoTTrainer


# ---------------------------------------------------------------------------
# Dummy 模型
# ---------------------------------------------------------------------------

class _DummyModel(nn.Module):
    """只含一个可训练标量的 dummy 模型。

    forward 忽略 batch 内容，返回 w^2（标量 loss），
    优化器将 w 推向 0，loss 单调下降——方便验证梯度流通。
    """

    def __init__(self) -> None:
        super().__init__()
        # 初始 w=3.0，loss=9.0，梯度 dL/dw=2w=6，一步后 loss 必然下降
        self.w = nn.Parameter(torch.tensor(3.0))

    def forward(self, batch: Any) -> torch.Tensor:  # noqa: ARG002
        return self.w ** 2


# ---------------------------------------------------------------------------
# 辅助：生成最小合法 PNG（不依赖 PIL/Pillow）
# ---------------------------------------------------------------------------

def _write_minimal_png(path: Path) -> None:
    """写入 1×1 白色像素 PNG，作为 mock 截图。"""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + _chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# 假数据构造
# ---------------------------------------------------------------------------

def _make_fake_jsonl(out_dir: Path, n: int = 5) -> Path:
    """构造 n 条统一格式样本，复用同一张 mock PNG 作为截图。"""
    img = out_dir / "screen.png"
    _write_minimal_png(img)

    jsonl = out_dir / "traces.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for i in range(n):
            sample = {
                "screenshot": str(img),
                "instruction": f"点击按钮 {i}",
                "elements": [
                    {
                        "id": f"e{i}",
                        "type": "button",
                        "bbox": [0.05 * i, 0.1, 0.05 * i + 0.4, 0.3],
                        "label": f"btn{i}",
                    }
                ],
                "action": {"type": "click", "coord": [0.3, 0.2]},
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return jsonl


# ---------------------------------------------------------------------------
# 主验证流程
# ---------------------------------------------------------------------------

def run_pipeline_test() -> None:
    """端到端 CPU mock 验证，失败时抛出 AssertionError。"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Step 1: 构造 5 条假数据 ───────────────────────────────────────────
        jsonl_path = _make_fake_jsonl(tmp, n=5)
        print(f"[1] 假数据 JSONL: {jsonl_path}  ({jsonl_path.stat().st_size} bytes)")

        # ── Step 2: UITrajectoryDataset 加载 ─────────────────────────────────
        dataset = UITrajectoryDataset.from_custom(str(jsonl_path))
        assert len(dataset) == 5, f"期望 5 条样本，实际 {len(dataset)}"

        sample0 = dataset[0]
        required_keys = {"screenshot", "instruction", "elements", "action"}
        assert required_keys <= set(sample0.keys()), \
            f"样本缺少字段：{required_keys - set(sample0.keys())}"

        # __getitem__ 取第 4 条（边界检查）
        sample4 = dataset[4]
        assert sample4["instruction"] == "点击按钮 4"

        print(f"[2] 数据集加载：{len(dataset)} 条，样本键={sorted(sample0.keys())}")

        # ── Step 3: MoTTrainer 跑 1 个 batch ─────────────────────────────────
        model = _DummyModel()
        config = TrainingConfig(
            stage1_data=str(jsonl_path),
            stage1_epochs=1,
            batch_size=2,
            device="cpu",
            fp16=False,
            gradient_checkpointing=False,
        )
        trainer = MoTTrainer(config, model)

        # 取 2 条样本模拟 batch（DataLoader collate 之前的原始列表）
        batch = [dataset[i] for i in range(2)]

        # 前向 #1：记录初始 loss
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
        optimizer.zero_grad()
        loss1 = trainer._forward_step(batch)

        print(f"[3] 初始 loss: {loss1.item():.6f}")

        # ── Step 4: 验证梯度流通 ──────────────────────────────────────────────
        assert loss1.requires_grad, "loss 必须支持自动微分（requires_grad=True）"

        loss1.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 '{name}' 梯度为 None"
                assert param.grad.abs().item() > 0, f"参数 '{name}' 梯度全零"

        print(f"[4] 梯度验证通过（dL/dw = {model.w.grad.item():.4f}）")

        # 参数更新
        optimizer.step()
        optimizer.zero_grad()

        # ── Step 5: 验证 loss 下降 ────────────────────────────────────────────
        with torch.no_grad():
            loss2 = trainer._forward_step(batch)

        drop = loss1.item() - loss2.item()
        print(f"[5] 更新后 loss: {loss2.item():.6f}（下降 {drop:.6f}）")
        assert loss2.item() < loss1.item(), \
            f"loss 应下降：{loss1.item():.6f} → {loss2.item():.6f}"

        print("\n[✓] 训练管道 CPU mock 验证全部通过\n")


if __name__ == "__main__":
    run_pipeline_test()
