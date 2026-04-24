"""
UI 元素检测器 - 基于 YOLOv8n

支持：
  1. 预训练 YOLOv8n（通用物体检测，作为 baseline）
  2. UI fine-tuned 权重（后续加载 OmniParser 的权重或自训练权重）

后端优先级：ultralytics → onnxruntime（轻量替代）

延迟目标：< 10ms（GPU）/ < 100ms（CPU）
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YOLO 类别 → ElementType 映射（基于 VINS 数据集的 21 类）
# ---------------------------------------------------------------------------

YOLO_TO_ELEMENT_TYPE: dict[str, str] = {
    # VINS / Rico 常见类别
    "Button": "BUTTON",
    "button": "BUTTON",
    "btn": "BUTTON",
    "CheckBox": "CHECKBOX",
    "checkbox": "CHECKBOX",
    "radio": "CHECKBOX",
    "EditText": "TEXT_INPUT",
    "edit_text": "TEXT_INPUT",
    "TextInput": "TEXT_INPUT",
    "Input": "TEXT_INPUT",
    "input": "TEXT_INPUT",
    "textbox": "TEXT_INPUT",
    "ImageView": "IMAGE",
    "image_view": "IMAGE",
    "Image": "IMAGE",
    "image": "IMAGE",
    "img": "IMAGE",
    "Icon": "IMAGE",
    "icon": "IMAGE",
    "TextView": "TEXT",
    "text_view": "TEXT",
    "Text": "TEXT",
    "text": "TEXT",
    "label": "TEXT",
    "link": "BUTTON",
    "Switch": "SWITCH",
    "switch": "SWITCH",
    "Slider": "CONTAINER",
    "slider": "CONTAINER",
    "Toolbar": "NAV_BAR",
    "toolbar": "NAV_BAR",
    "NavigationBar": "NAV_BAR",
    "nav_bar": "NAV_BAR",
    "nav": "NAV_BAR",
    "Tab": "TAB",
    "tab": "TAB",
    "TabBar": "TAB",
    "List": "LIST",
    "list": "LIST",
    "ListItem": "LIST_ITEM",
    "list_item": "LIST_ITEM",
    "listitem": "LIST_ITEM",
    "ScrollView": "SCROLL_VIEW",
    "scroll_view": "SCROLL_VIEW",
    "Container": "CONTAINER",
    "container": "CONTAINER",
    # WebIcon 等额外标签
    "WebIcon": "IMAGE",
    "RadioButton": "CHECKBOX",
    "ProgressBar": "CONTAINER",
    "PageIndicator": "CONTAINER",
    "UpperTaskBar": "NAV_BAR",
    "BlockingLayer": "CONTAINER",
    "BackgroundImage": "IMAGE",
    "Advertisement": "IMAGE",
}


def _map_class_to_element_type(class_name: str):
    """将 YOLO 检测到的类别名映射到 ACP ElementType。"""
    from acp.schema.elements import ElementType

    type_name = YOLO_TO_ELEMENT_TYPE.get(class_name)
    if type_name is None:
        # 尝试大小写不敏感匹配
        lower = class_name.lower()
        type_name = YOLO_TO_ELEMENT_TYPE.get(lower)
    if type_name is None:
        return ElementType.UNKNOWN
    try:
        return ElementType[type_name]
    except KeyError:
        return ElementType.UNKNOWN


def _bbox_to_element_id(bbox: list[float], class_name: str) -> str:
    """根据 bbox + 类别生成唯一 ID（确定性哈希）。"""
    key = f"{class_name}_{bbox[0]:.1f}_{bbox[1]:.1f}_{bbox[2]:.1f}_{bbox[3]:.1f}"
    return "vis_" + hashlib.md5(key.encode()).hexdigest()[:12]


def _bbox_to_rect_and_center(bbox: list[float]):
    """将 [x1, y1, x2, y2] 格式转为 Rect + 中心 Point。"""
    from acp.schema.elements import Point, Rect

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    rect = Rect(x=x1, y=y1, width=width, height=height)
    center = Point(x=x1 + width / 2, y=y1 + height / 2)
    return rect, center


def _infer_actions(element_type) -> list[str]:
    """根据元素类型推断可执行的动作列表。"""
    from acp.schema.elements import ElementType

    clickable_types = {
        ElementType.BUTTON,
        ElementType.CHECKBOX,
        ElementType.SWITCH,
        ElementType.TAB,
        ElementType.LIST_ITEM,
        ElementType.IMAGE,
        ElementType.NAV_BAR,
    }
    typable_types = {ElementType.TEXT_INPUT}
    scrollable_types = {ElementType.SCROLL_VIEW, ElementType.LIST, ElementType.CONTAINER}

    actions: list[str] = []
    if element_type in clickable_types:
        actions.append("click")
    if element_type in typable_types:
        actions.extend(["click", "type"])
    if element_type in scrollable_types:
        actions.append("scroll")
    if not actions:
        actions.append("click")
    return actions


# ---------------------------------------------------------------------------
# 后端抽象
# ---------------------------------------------------------------------------


class _DetectorBackend:
    """检测器后端抽象基类。"""

    def predict(self, image_bytes: bytes) -> list[dict[str, Any]]:
        """
        返回检测结果列表，格式：
          [{"class": str, "bbox": [x1, y1, x2, y2], "confidence": float}, ...]
        """
        raise NotImplementedError


class _UltralyticsBackend(_DetectorBackend):
    """基于 ultralytics 库的 YOLOv8 后端。"""

    def __init__(self, model_path: str, confidence: float) -> None:
        from ultralytics import YOLO  # type: ignore[import]

        self._model = YOLO(model_path)
        self._confidence = confidence
        logger.info("UIDetector: 使用 ultralytics 后端，模型 = %s", model_path)

    def predict(self, image_bytes: bytes) -> list[dict[str, Any]]:
        from PIL import Image  # type: ignore[import]

        image = Image.open(io.BytesIO(image_bytes))
        results = self._model(image, conf=self._confidence, verbose=False)

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    }
                )
        return detections


class _ONNXBackend(_DetectorBackend):
    """基于 onnxruntime 的轻量 YOLOv8 后端。

    适用于 ultralytics 安装失败或需要轻量化的场景。
    模型文件需为 YOLOv8 ONNX 格式（带 NMS 后处理）。
    """

    def __init__(self, model_path: str, confidence: float) -> None:
        import onnxruntime as ort  # type: ignore[import]

        self._session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._confidence = confidence
        self._input_name = self._session.get_inputs()[0].name
        # 推断输入尺寸（通常 640x640）
        shape = self._session.get_inputs()[0].shape
        self._input_h = shape[2] if isinstance(shape[2], int) else 640
        self._input_w = shape[3] if isinstance(shape[3], int) else 640
        logger.info(
            "UIDetector: 使用 ONNX Runtime 后端，模型 = %s，输入尺寸 = %dx%d",
            model_path,
            self._input_w,
            self._input_h,
        )

    def predict(self, image_bytes: bytes) -> list[dict[str, Any]]:
        import numpy as np  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size

        # 缩放到模型输入尺寸
        resized = image.resize((self._input_w, self._input_h))
        img_array = np.array(resized, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

        outputs = self._session.run(None, {self._input_name: img_array})
        # 输出格式：[1, num_detections, 6] (x1,y1,x2,y2,conf,cls)
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]

        scale_x = orig_w / self._input_w
        scale_y = orig_h / self._input_h

        detections: list[dict[str, Any]] = []
        for det in raw:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            if float(conf) < self._confidence:
                continue
            detections.append(
                {
                    "class": str(int(cls_id)),
                    "bbox": [
                        float(x1) * scale_x,
                        float(y1) * scale_y,
                        float(x2) * scale_x,
                        float(y2) * scale_y,
                    ],
                    "confidence": float(conf),
                }
            )
        return detections


class _MockBackend(_DetectorBackend):
    """Mock 后端，用于测试（无需真实模型）。"""

    def __init__(self, detections: Optional[list[dict[str, Any]]] = None) -> None:
        # 注意：不用 `or` 以避免将空列表 [] 替换为默认值
        if detections is None:
            self._detections: list[dict[str, Any]] = [
                {"class": "Button", "bbox": [10.0, 20.0, 110.0, 60.0], "confidence": 0.92},
                {"class": "EditText", "bbox": [10.0, 80.0, 300.0, 120.0], "confidence": 0.87},
                {"class": "TextView", "bbox": [10.0, 140.0, 200.0, 170.0], "confidence": 0.75},
            ]
        else:
            self._detections = detections
        logger.info("UIDetector: 使用 Mock 后端（测试模式）")

    def predict(self, image_bytes: bytes) -> list[dict[str, Any]]:
        return list(self._detections)


# ---------------------------------------------------------------------------
# UIDetector
# ---------------------------------------------------------------------------


class UIDetector:
    """基于 YOLOv8 的 UI 元素检测器。

    自动选择后端：
      1. 优先使用 ultralytics（功能完整，包含 class names）
      2. ultralytics 不可用时降级到 onnxruntime（轻量）
      3. 两者均不可用时抛出 RuntimeError（可通过 use_mock=True 绕过）

    Args:
        model_path: 模型权重路径。
            - ultralytics 后端：.pt 文件（如 "yolov8n.pt"）
            - ONNX 后端：.onnx 文件（如 "yolov8n.onnx"）
        confidence: 最低置信度阈值，低于此值的检测结果丢弃。
        backend: 强制指定后端（"ultralytics" | "onnx" | "mock" | "auto"），默认 "auto"。
        use_mock: 为 True 时强制使用 Mock 后端（测试用）。
        mock_detections: 自定义 Mock 检测结果（仅 use_mock=True 时有效）。
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.3,
        backend: str = "auto",
        use_mock: bool = False,
        mock_detections: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.confidence = confidence
        self._backend = self._load_backend(
            model_path, confidence, backend, use_mock, mock_detections
        )

    @staticmethod
    def _load_backend(
        model_path: str,
        confidence: float,
        backend: str,
        use_mock: bool,
        mock_detections: Optional[list[dict[str, Any]]],
    ) -> _DetectorBackend:
        if use_mock or backend == "mock":
            return _MockBackend(mock_detections)

        if backend == "ultralytics":
            return _UltralyticsBackend(model_path, confidence)

        if backend == "onnx":
            return _ONNXBackend(model_path, confidence)

        # "auto"：尝试 ultralytics → ONNX → 报错
        try:
            return _UltralyticsBackend(model_path, confidence)
        except ImportError:
            logger.warning(
                "ultralytics 未安装，尝试 ONNX Runtime 后端。"
                "如需使用 ultralytics 请运行：pip install ultralytics"
            )
        onnx_path = model_path.replace(".pt", ".onnx")
        try:
            return _ONNXBackend(onnx_path, confidence)
        except ImportError:
            raise RuntimeError(
                "ultralytics 和 onnxruntime 均未安装，无法加载视觉检测后端。\n"
                "请运行：pip install -r acp/vision/requirements.txt\n"
                "或使用 use_mock=True 启动测试模式。"
            )

    # ---- 核心检测 ----

    def detect(self, screenshot: bytes) -> list[dict[str, Any]]:
        """输入截图字节，输出检测结果列表。

        Args:
            screenshot: 图像字节数据（PNG / JPEG）。

        Returns:
            检测结果列表，每项格式：
            {"class": "button", "bbox": [x1, y1, x2, y2], "confidence": 0.85}
            其中 bbox 为 [左上x, 左上y, 右下x, 右下y]（像素坐标）。
        """
        return self._backend.predict(screenshot)

    def detect_to_acp_elements(
        self,
        screenshot: bytes,
        page_url: str = "",
        app: str = "",
    ) -> list:
        """检测截图并将结果直接转为 ACP Element Schema。

        Args:
            screenshot: 图像字节数据（PNG / JPEG）。
            page_url: 当前页面 URL（用于丰富元素 selector 信息）。
            app: 应用标识（预留，用于 QLoRA 适配器选择）。

        Returns:
            ACPElement 列表，source 字段为 ElementSource.VISUAL_MODEL。
        """
        from acp.schema.elements import ACPElement, ElementSource, ElementStates, ElementType

        raw_detections = self.detect(screenshot)
        elements = []

        for det in raw_detections:
            class_name: str = det["class"]
            bbox: list[float] = det["bbox"]
            conf: float = det["confidence"]

            element_type = _map_class_to_element_type(class_name)
            bounds, center = _bbox_to_rect_and_center(bbox)
            element_id = _bbox_to_element_id(bbox, class_name)
            actions = _infer_actions(element_type)

            element = ACPElement(
                id=element_id,
                type=element_type,
                platform_class=class_name,
                bounds=bounds,
                center=center,
                source=ElementSource.VISUAL_MODEL,
                confidence=conf,
                states=ElementStates(
                    clickable=element_type
                    in {
                        ElementType.BUTTON,
                        ElementType.CHECKBOX,
                        ElementType.SWITCH,
                        ElementType.TAB,
                        ElementType.LIST_ITEM,
                    },
                    enabled=True,
                    visible=True,
                ),
                actions=actions,
                selector=f"visual://{class_name}@{bbox[0]:.0f},{bbox[1]:.0f}",
            )
            elements.append(element)

        logger.debug(
            "UIDetector.detect_to_acp_elements: 检测到 %d 个元素（url=%s）",
            len(elements),
            page_url,
        )
        return elements
