"""
视觉兜底模块（Tier-3）

模块结构：
  detector.py   — YOLOv8n UI 元素检测（支持 ultralytics / onnxruntime 双后端）
  segmenter.py  — MobileSAM 实例分割（预留 - Phase 2）
  vlm.py        — VLM 视觉理解（ShowUI-2B / UGround-2B，预留 - Phase 2）

使用场景：控件树失效时（Canvas 游戏、Closed Shadow DOM、验证码等）

快速开始：
    from acp.vision.detector import UIDetector

    # 测试模式（无需安装模型）
    detector = UIDetector(use_mock=True)
    detections = detector.detect(screenshot_bytes)
    elements = detector.detect_to_acp_elements(screenshot_bytes, page_url="https://example.com")

    # 生产模式（需要 pip install -r acp/vision/requirements.txt）
    detector = UIDetector(model_path="yolov8n.pt", confidence=0.3)
"""

from acp.vision.detector import YOLO_TO_ELEMENT_TYPE, UIDetector
from acp.vision.segmenter import MobileSAMSegmenter
from acp.vision.vlm import BaseVLM, ShowUI, UGround

__all__ = [
    "UIDetector",
    "YOLO_TO_ELEMENT_TYPE",
    "MobileSAMSegmenter",
    "BaseVLM",
    "ShowUI",
    "UGround",
]
