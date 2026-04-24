"""
实例分割接口 - 基于 MobileSAM（预留）

用途：将 YOLO 的粗糙检测框精化为精确的像素级掩码，
     从而获得更准确的元素边界和点击坐标。

Phase 2 实现路线：
  1. 部署 MobileSAM（~10M 参数，ONNX 导出）
  2. 将 YOLO bbox 作为 prompt 输入 MobileSAM
  3. 输出精确分割掩码 → 更新 ACPElement.bounds

参考：
  - MobileSAM: https://github.com/ChaoningZhang/MobileSAM
  - 延迟目标：~10ms（GPU）
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MobileSAMSegmenter:
    """基于 MobileSAM 的实例分割器（预留 - Phase 2 实现）。

    使用方式：
        segmenter = MobileSAMSegmenter()
        await segmenter.load()
        mask = await segmenter.segment(screenshot, bbox=[x1, y1, x2, y2])
    """

    def __init__(
        self,
        model_path: str = "mobile_sam.pt",
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._model = None
        logger.info(
            "MobileSAMSegmenter 初始化（model=%s, device=%s）— 等待 Phase 2 实现",
            model_path,
            device,
        )

    async def load(self) -> None:
        """加载 MobileSAM 模型（Phase 2 实现）。"""
        raise NotImplementedError(
            "MobileSAMSegmenter.load() 尚未实现，等待 Phase 2。\n"
            "Phase 2 路线：\n"
            "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n"
            "  from mobile_sam import sam_model_registry, SamPredictor\n"
            "  model = sam_model_registry['vit_t'](checkpoint=model_path)"
        )

    async def segment(
        self,
        screenshot: bytes,
        bbox: list[float],
    ) -> Optional[dict]:
        """对指定区域进行精确分割（Phase 2 实现）。

        Args:
            screenshot: PNG 截图字节。
            bbox: YOLO 检测框 [x1, y1, x2, y2]。

        Returns:
            分割结果字典：
            {
                "mask": np.ndarray,          # 二值掩码
                "refined_bbox": [x1,y1,x2,y2],  # 精化后的边界框
                "center": [cx, cy],              # 精化后的中心点
                "confidence": float,
            }
            或 None（分割失败）。
        """
        raise NotImplementedError(
            "MobileSAMSegmenter.segment() 尚未实现，等待 Phase 2。"
        )

    async def segment_batch(
        self,
        screenshot: bytes,
        bboxes: list[list[float]],
    ) -> list[Optional[dict]]:
        """批量分割（Phase 2 实现）。

        Args:
            screenshot: PNG 截图字节。
            bboxes: YOLO 检测框列表，每项 [x1, y1, x2, y2]。

        Returns:
            与 bboxes 等长的分割结果列表（None 表示该框分割失败）。
        """
        raise NotImplementedError(
            "MobileSAMSegmenter.segment_batch() 尚未实现，等待 Phase 2。"
        )
