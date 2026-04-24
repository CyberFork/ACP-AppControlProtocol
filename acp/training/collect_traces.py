"""将 FlowRunner 的执行记录转换为训练数据（统一 JSONL 格式）。

FlowRunner 执行时产生的 log 包含每步的 action/element/screenshot，
本脚本将其提取为模型可直接消费的训练三元组。

用法：
    python -m acp.training.collect_traces \\
        --flow-log flow_log.json \\
        --screenshots screenshots/ \\
        --output datasets/custom/traces.jsonl

flow_log.json 格式（FlowRunner.log）：
[
    {
        "step": 1,
        "action": "click",
        "target": "登录按钮",
        "element_id": "e0001",
        "coord": [0.5, 0.3],
        "screenshot": "step_001.png",
        "success": true
    },
    ...
]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 核心转换逻辑
# ---------------------------------------------------------------------------

def extract_training_samples(
    flow_log: list[dict[str, Any]],
    screenshots_dir: str,
    instruction: str = "",
    only_success: bool = True,
) -> list[dict[str, Any]]:
    """从 FlowRunner log 提取训练三元组。

    Args:
        flow_log:        FlowRunner.log（每条含 step/action/target/element_id/coord/screenshot/success）
        screenshots_dir: 截图目录，step log 中的 screenshot 相对于此目录
        instruction:     全局任务指令（如 "登录 ai6666.ai"）；若为空则用 target 作为 instruction
        only_success:    是否只保留成功步骤（默认 True）

    Returns:
        统一格式的样本列表
    """
    screenshots_path = Path(screenshots_dir)
    samples: list[dict] = []

    for entry in flow_log:
        if only_success and not entry.get("success", False):
            logger.debug("跳过失败步骤 step=%s", entry.get("step"))
            continue

        screenshot_file = entry.get("screenshot", "")
        if not screenshot_file:
            logger.warning("步骤 %s 缺少 screenshot，跳过", entry.get("step"))
            continue

        screenshot_path = screenshots_path / screenshot_file
        if not screenshot_path.exists():
            logger.warning("截图不存在：%s，跳过", screenshot_path)
            continue

        action_type = entry.get("action", "click").lower()
        element_id = entry.get("element_id", "")
        coord = entry.get("coord", [0.5, 0.5])
        target = entry.get("target", "")
        value = entry.get("value", "")
        bbox = entry.get("bbox", [0, 0, 1, 1])

        # 构建 action
        if action_type == "click":
            action: dict[str, Any] = {"type": "click", "coord": coord}
        elif action_type in ("type", "fill"):
            action = {"type": "type", "text": value, "coord": coord}
        elif action_type == "scroll":
            action = {"type": "scroll", "direction": entry.get("direction", "down"), "coord": coord}
        elif action_type == "navigate":
            action = {"type": "navigate", "url": entry.get("url", "")}
        else:
            action = {"type": action_type, "coord": coord}

        if element_id:
            action["element_id"] = element_id

        # 构建 elements（如果 log 提供了元素信息）
        elements: list[dict] = []
        if element_id or target:
            elements.append({
                "id": element_id or "e0",
                "type": entry.get("element_type", "interactive"),
                "bbox": bbox,
                "label": target,
            })

        # 若 log 含完整 elements 列表
        for el in entry.get("elements", []):
            # 避免重复
            if not any(e["id"] == el.get("id") for e in elements):
                elements.append({
                    "id": el.get("id", ""),
                    "type": el.get("type", "unknown"),
                    "bbox": el.get("bbox", [0, 0, 1, 1]),
                    "label": el.get("label", el.get("text", "")),
                })

        sample = {
            "screenshot": str(screenshot_path.resolve()),
            "instruction": instruction or target or action_type,
            "elements": elements,
            "action": action,
        }

        # 保留元数据（方便调试）
        if entry.get("step") is not None:
            sample["_meta"] = {
                "step": entry["step"],
                "flow_log_success": entry.get("success", True),
            }

        samples.append(sample)

    logger.info("提取 %d 条训练样本（共 %d 步骤）", len(samples), len(flow_log))
    return samples


def convert_flow_log_file(
    flow_log_path: str,
    screenshots_dir: str,
    output_path: str,
    instruction: str = "",
    only_success: bool = True,
    append: bool = False,
) -> int:
    """从文件读取 flow log，转换后写入 JSONL 文件。

    Returns:
        写入的样本数量
    """
    log_path = Path(flow_log_path)
    with open(log_path, encoding="utf-8") as f:
        flow_log = json.load(f)

    if not isinstance(flow_log, list):
        # 兼容 {"log": [...]} 格式
        flow_log = flow_log.get("log", [])

    samples = extract_training_samples(
        flow_log=flow_log,
        screenshots_dir=screenshots_dir,
        instruction=instruction,
        only_success=only_success,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    written = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    logger.info("已写入 %d 条样本 → %s", written, out_path)
    return written


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m acp.training.collect_traces",
        description="将 FlowRunner 执行记录转换为训练数据（统一 JSONL 格式）",
    )
    parser.add_argument(
        "--flow-log",
        required=True,
        metavar="FILE",
        help="FlowRunner log JSON 文件路径（列表格式）",
    )
    parser.add_argument(
        "--screenshots",
        required=True,
        metavar="DIR",
        help="截图目录，log 中 screenshot 字段相对于此目录",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--instruction",
        default="",
        metavar="TEXT",
        help="全局任务指令（可选，为空则从每步 target 推断）",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="也包含失败步骤（默认只保留成功步骤）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加到已有 JSONL 文件（默认覆盖）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细日志",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    try:
        written = convert_flow_log_file(
            flow_log_path=args.flow_log,
            screenshots_dir=args.screenshots,
            output_path=args.output,
            instruction=args.instruction,
            only_success=not args.include_failed,
            append=args.append,
        )
        print(f"完成：{written} 条样本 → {args.output}")
        return 0
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("转换失败：%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
