"""数据加载 — UITrajectoryDataset。

统一格式（每条样本）：
{
    "screenshot": "path/to/img.png",
    "instruction": "点击登录按钮",
    "elements": [
        {"id": "e0001", "type": "button", "bbox": [x1, y1, x2, y2], "label": "登录"}
    ],
    "action": {"type": "click", "element_id": "e0001", "coord": [x, y]}
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    # CPU mock：允许在无 GPU 环境下导入和测试
    _TORCH_AVAILABLE = False

    class _TorchDataset:  # type: ignore[no-redef]
        """torch.utils.data.Dataset 的 stub（用于无 GPU 环境测试）。"""
        pass


class UITrajectoryDataset(_TorchDataset):
    """UI 操作轨迹数据集。

    统一格式：
        {
            "screenshot": "path/to/img.png",
            "instruction": "点击登录按钮",
            "elements": [{"id": "...", "type": "button",
                          "bbox": [x1,y1,x2,y2], "label": "登录"}],
            "action": {"type": "click", "element_id": "...", "coord": [x, y]}
        }

    用法：
        ds = UITrajectoryDataset.from_custom("datasets/custom/traces.jsonl")
        sample = ds[0]
    """

    def __init__(self, samples: list[dict[str, Any]]):
        self._samples = samples

    # ── 标准 Dataset 接口 ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]

    # ── 工厂方法：各数据集格式 ───────────────────────────────────────────────

    @classmethod
    def from_aitw(cls, data_dir: str) -> "UITrajectoryDataset":
        """加载 AITW（Android In The Wild）格式数据集。

        AITW 原始格式为 TFRecord；此处期望已预处理为 JSONL，每行字段：
          episode_id, step_id, goal, screenshot_path, action_type,
          touch_point, lift_point, typed_text, screen_elements
        """
        data_path = Path(data_dir)
        samples: list[dict] = []

        for jsonl_file in sorted(data_path.glob("**/*.jsonl")):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    sample = cls._convert_aitw(raw, base_dir=data_path)
                    if sample:
                        samples.append(sample)

        logger.info("AITW: 加载 %d 条样本（目录：%s）", len(samples), data_dir)
        return cls(samples)

    @classmethod
    def from_mind2web(cls, data_dir: str) -> "UITrajectoryDataset":
        """加载 Mind2Web 格式数据集。

        Mind2Web 原始格式为 JSON 文件，每个文件包含一条任务轨迹：
          task, website, domain, subdomain, actions（列表）
        每条 action 含 action_type, element（xpath/text），raw_html（可选）。
        """
        data_path = Path(data_dir)
        samples: list[dict] = []

        for json_file in sorted(data_path.glob("**/*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("跳过损坏文件 %s: %s", json_file, e)
                continue

            # raw 可能是单条轨迹，也可能是列表
            trajectories = raw if isinstance(raw, list) else [raw]
            for traj in trajectories:
                for step_sample in cls._convert_mind2web(traj, base_dir=data_path):
                    samples.append(step_sample)

        logger.info("Mind2Web: 加载 %d 条样本（目录：%s）", len(samples), data_dir)
        return cls(samples)

    @classmethod
    def from_screenspot(cls, data_dir: str) -> "UITrajectoryDataset":
        """加载 ScreenSpot 格式数据集（用于 grounding 训练）。

        ScreenSpot 每条样本：
          instruction, screenshot_path, bbox（[x1,y1,x2,y2]，归一化 0-1）
        """
        data_path = Path(data_dir)
        samples: list[dict] = []

        for json_file in sorted(data_path.glob("**/*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    items = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("跳过损坏文件 %s: %s", json_file, e)
                continue

            if not isinstance(items, list):
                items = [items]
            for item in items:
                sample = cls._convert_screenspot(item, base_dir=data_path)
                if sample:
                    samples.append(sample)

        logger.info("ScreenSpot: 加载 %d 条样本（目录：%s）", len(samples), data_dir)
        return cls(samples)

    @classmethod
    def from_custom(cls, jsonl_path: str) -> "UITrajectoryDataset":
        """加载自定义 JSONL 格式。

        支持两种格式：
        1. 统一格式（collect_traces.py 输出）：每行含 screenshot/instruction/elements/action
        2. testenv auto_label.js 导出格式：每行含 session_id/page/elements/actions
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL 文件不存在：{jsonl_path}")

        samples: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("第 %d 行解析失败：%s", lineno, e)
                    continue
                # auto_label.js 会话格式：含 session_id + actions 列表
                if "session_id" in record and "actions" in record:
                    for s in cls._convert_autolabel_session(record):
                        samples.append(s)
                else:
                    samples.append(record)

        logger.info("custom JSONL: 加载 %d 条样本（文件：%s）", len(samples), jsonl_path)
        return cls(samples)

    # ── 私有转换方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _convert_aitw(
        raw: dict, base_dir: Path
    ) -> Optional[dict[str, Any]]:
        """AITW 单步 → 统一格式。"""
        screenshot = raw.get("screenshot_path", "")
        instruction = raw.get("instruction", raw.get("goal", ""))
        # gesture_type 是 AITW 原始字段，action_type 为预处理后的别名
        action_type = raw.get("gesture_type", raw.get("action_type", ""))
        touch = raw.get("touch_point", [0.5, 0.5])
        typed = raw.get("typed_text", "")

        if not (screenshot and instruction and action_type):
            return None

        # 坐标为归一化（0-1），统一保留
        coord = [float(touch[0]), float(touch[1])]

        if action_type == "CLICK":
            action = {"type": "click", "coord": coord}
        elif action_type == "TYPE":
            action = {"type": "type", "text": typed, "coord": coord}
        elif action_type == "SCROLL":
            direction = raw.get("scroll_direction", "down")
            action = {"type": "scroll", "direction": direction, "coord": coord}
        else:
            action = {"type": action_type.lower(), "coord": coord}

        elements = []
        for el in raw.get("screen_elements", []):
            elements.append({
                "id": el.get("id", ""),
                "type": el.get("type", "unknown"),
                "bbox": el.get("bbox", [0, 0, 1, 1]),
                "label": el.get("text", ""),
            })

        return {
            "screenshot": str(base_dir / screenshot) if not Path(screenshot).is_absolute() else screenshot,
            "instruction": instruction,
            "elements": elements,
            "action": action,
        }

    @staticmethod
    def _convert_mind2web(
        traj: dict, base_dir: Path
    ) -> list[dict[str, Any]]:
        """Mind2Web 轨迹 → 统一格式列表（每步一条）。"""
        task = traj.get("task", "")
        actions = traj.get("actions", [])
        samples = []

        for step in actions:
            screenshot = step.get("screenshot_path", step.get("screenshot", ""))
            action_type = step.get("action_type", "click").lower()
            element = step.get("element", {})
            bbox = element.get("bbox", [0, 0, 1, 1]) if isinstance(element, dict) else [0, 0, 1, 1]
            label = element.get("text", "") if isinstance(element, dict) else str(element)

            coord = [
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            ]

            if action_type == "click":
                action = {"type": "click", "coord": coord}
            elif action_type in ("type", "input"):
                action = {"type": "type", "text": step.get("value", ""), "coord": coord}
            else:
                action = {"type": action_type, "coord": coord}

            samples.append({
                "screenshot": str(base_dir / screenshot) if screenshot and not Path(screenshot).is_absolute() else screenshot,
                "instruction": task,
                "elements": [{"id": "e0", "type": "interactive", "bbox": bbox, "label": label}],
                "action": action,
            })

        return samples

    @staticmethod
    def _convert_screenspot(
        item: dict, base_dir: Path
    ) -> Optional[dict[str, Any]]:
        """ScreenSpot 单条 → 统一格式（grounding：给出目标框）。"""
        # img_path 是 ScreenSpot 原始字段，img_filename 为常见别名
        screenshot = item.get("img_path", item.get("img_filename", item.get("screenshot", "")))
        instruction = item.get("instruction", "")
        bbox = item.get("bbox", None)

        if not (screenshot and instruction and bbox):
            return None

        coord = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        return {
            "screenshot": str(base_dir / screenshot) if not Path(screenshot).is_absolute() else screenshot,
            "instruction": instruction,
            "elements": [{"id": "target", "type": "target", "bbox": bbox, "label": instruction}],
            "action": {"type": "click", "coord": coord},
        }

    @staticmethod
    def _convert_autolabel_session(session: dict) -> list[dict[str, Any]]:
        """auto_label.js 会话 → 统一格式列表（每条 action 一条样本）。

        auto_label.js 导出格式：
          session_id, page, screenshot(base64 or null),
          elements([{id, type, bbox:[x,y,w,h], text, ...}]),
          actions([{type, coord, element, value, key, ...}])
        """
        screenshot = session.get("screenshot") or ""

        # 元素列表：bbox 格式 [x, y, w, h] → [x1, y1, x2, y2]
        unified_elements: list[dict[str, Any]] = []
        for el in session.get("elements", []):
            raw_bbox = el.get("bbox", [0, 0, 0, 0])
            if len(raw_bbox) == 4:
                x, y, w, h = raw_bbox
                bbox = [x, y, x + w, y + h]
            else:
                bbox = raw_bbox
            unified_elements.append({
                "id": el.get("id", ""),
                "type": el.get("type", "unknown"),
                "bbox": bbox,
                "label": el.get("text", ""),
            })

        samples: list[dict[str, Any]] = []
        for act in session.get("actions", []):
            action_type = act.get("type", "click").lower()
            coord = act.get("coord", [0, 0])
            element_info = act.get("element") or {}
            value = act.get("value", "")

            if action_type in ("click", "dblclick", "right_click"):
                action: dict[str, Any] = {"type": action_type, "coord": coord}
            elif action_type == "input":
                action = {"type": "type", "text": value, "coord": coord}
            elif action_type == "scroll":
                scroll_y = act.get("scrollY", 0)
                action = {
                    "type": "scroll",
                    "direction": "down" if scroll_y >= 0 else "up",
                    "coord": coord,
                }
            elif action_type == "keydown":
                action = {"type": "keydown", "key": act.get("key", "")}
            elif action_type in ("dragstart", "drop"):
                action = {"type": action_type, "coord": coord}
            else:
                action = {"type": action_type, "coord": coord}

            label = element_info.get("text", "") or element_info.get("type", action_type)
            instruction = f"{action_type} {label}".strip()

            samples.append({
                "screenshot": screenshot,
                "instruction": instruction,
                "elements": unified_elements,
                "action": action,
            })

        return samples
