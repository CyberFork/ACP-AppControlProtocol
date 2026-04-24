"""
VLM 视觉理解接口（预留）

提供基于视觉语言模型（VLM）的 UI 理解能力：
  - ShowUI-2B：专为 GUI 操作设计的 VLM，ScreenSpot 基准优秀
  - UGround-V1-2B：通用 GUI 元素定位，ScreenSpot 81.5%
  - UI-TARS-2（7B）：综合最强，OSWorld 47.5%

接口设计：
  - describe(screenshot) → 页面描述文本
  - locate(screenshot, query) → 坐标 (x, y)
  - identify_elements(screenshot) → 元素列表

Phase 2 实现路线：
  1. vLLM 部署 UGround-V1-2B（推荐，平衡精度和成本）
  2. gRPC/HTTP 接口对接
  3. 结果转换为 ACPElement Schema
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseVLM(ABC):
    """VLM 视觉理解基础接口。

    所有具体 VLM 实现继承此类。
    """

    @abstractmethod
    async def describe(self, screenshot: bytes) -> str:
        """描述页面内容。

        Args:
            screenshot: PNG 截图字节

        Returns:
            页面描述文本
        """

    @abstractmethod
    async def locate(self, screenshot: bytes, query: str) -> Optional[tuple[float, float]]:
        """定位目标元素（文本描述 → 坐标）。

        Args:
            screenshot: PNG 截图字节
            query: 目标描述，如 "登录按钮"

        Returns:
            (x, y) 像素坐标，或 None（未找到）
        """

    @abstractmethod
    async def identify_elements(self, screenshot: bytes) -> list[dict]:
        """识别页面中的所有 UI 元素。

        Args:
            screenshot: PNG 截图字节

        Returns:
            元素列表，每项包含 {type, bbox, text, confidence}
        """


class ShowUI(BaseVLM):
    """ShowUI-2B VLM 接口（预留 - Phase 2 实现）。

    ShowUI-2B 专为 GUI 操作设计：
      - 基础：Qwen2-VL-2B-Instruct
      - 训练：GUI 感知 + GUI 导航数据
      - 推理：截图 → 操作目标坐标

    部署（Phase 2）：
      vllm serve Qwen/ShowUI-2B --host 0.0.0.0 --port 8002

    参考：https://huggingface.co/showlab/ShowUI-2B
    """

    def __init__(self, endpoint: str = "http://localhost:8002/v1") -> None:
        self._endpoint = endpoint
        logger.info("ShowUI 接口初始化（endpoint=%s）", endpoint)

    async def describe(self, screenshot: bytes) -> str:
        """预留 - Phase 2 实现"""
        raise NotImplementedError(
            "ShowUI.describe() 尚未实现，等待 Phase 2 vLLM 部署。"
        )

    async def locate(self, screenshot: bytes, query: str) -> Optional[tuple[float, float]]:
        """预留 - Phase 2 实现"""
        raise NotImplementedError(
            "ShowUI.locate() 尚未实现，等待 Phase 2 vLLM 部署。"
        )

    async def identify_elements(self, screenshot: bytes) -> list[dict]:
        """预留 - Phase 2 实现"""
        raise NotImplementedError(
            "ShowUI.identify_elements() 尚未实现，等待 Phase 2 vLLM 部署。"
        )


class UGround(BaseVLM):
    """UGround-V1-2B VLM 接口（预留 - Phase 2 实现）。

    UGround-V1-2B 通用 GUI 元素定位：
      - ScreenSpot 基准：81.5%（超越 GPT-4V）
      - 参数量：2B，可单卡 A10G 运行
      - 特点：坐标输出精确，适合"找到 X 并点击"

    部署（Phase 2）：
      vllm serve osunlp/UGround-V1-2B --host 0.0.0.0 --port 8003

    参考：https://huggingface.co/osunlp/UGround-V1-2B
    """

    def __init__(self, endpoint: str = "http://localhost:8003/v1") -> None:
        self._endpoint = endpoint
        logger.info("UGround 接口初始化（endpoint=%s）", endpoint)

    async def describe(self, screenshot: bytes) -> str:
        """预留 - Phase 2 实现"""
        raise NotImplementedError(
            "UGround.describe() 尚未实现，等待 Phase 2 vLLM 部署。"
        )

    async def locate(self, screenshot: bytes, query: str) -> Optional[tuple[float, float]]:
        """预留 - Phase 2 实现

        预期输入/输出格式：
          prompt: "在截图中找到：{query}，返回坐标 (x%, y%)"
          output: "(0.35, 0.42)"  # 相对坐标，需乘以图片尺寸
        """
        raise NotImplementedError(
            "UGround.locate() 尚未实现，等待 Phase 2 vLLM 部署。"
        )

    async def identify_elements(self, screenshot: bytes) -> list[dict]:
        """预留 - Phase 2 实现"""
        raise NotImplementedError(
            "UGround.identify_elements() 尚未实现，等待 Phase 2 vLLM 部署。"
        )
