"""
OmniParser v2 感知层：截图 → 元素列表 (bbox, label, type, center)

架构：
  YOLO icon_detect  → 图标 bounding box
  Florence-2-base   → 图标 caption
  easyOCR           → 文字识别
所有模型一次加载、复用。MPS 后端（Apple Silicon）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# HuggingFace repo
_REPO_ID = "microsoft/OmniParser-v2.0"


@dataclass
class UIElement:
    idx: int
    label: str                    # text content or icon caption
    elem_type: str                # "text" | "icon"
    bbox: list[float]             # [x1, y1, x2, y2] normalized 0-1
    center_x: float               # pixel
    center_y: float               # pixel
    interactivity: bool = True
    source: str = ""


class OmniPerception:
    """OmniParser v2 封装，前置一次加载，复用于多次 detect()。"""

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self._loaded = False
        self.yolo = None
        self.processor = None
        self.caption_model = None
        self.ocr = None

    def load(self) -> None:
        """下载并加载所有模型（首次调用时执行，之后复用）。"""
        if self._loaded:
            return

        import easyocr
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoProcessor
        from ultralytics import YOLO

        logger.info("下载 OmniParser v2 模型（首次运行会较慢）...")
        t0 = time.time()
        model_dir = snapshot_download(
            repo_id=_REPO_ID,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        logger.info("模型下载完成，耗时 %.1fs，路径: %s", time.time() - t0, model_dir)

        # YOLO icon detector
        self.yolo = YOLO(f"{model_dir}/icon_detect/model.pt")

        # Florence-2 caption model（icon_caption 是 fine-tuned Florence-2-base）
        logger.info("加载 Florence-2 caption 模型...")
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        self.caption_model = AutoModelForCausalLM.from_pretrained(
            f"{model_dir}/icon_caption",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.caption_model.eval()

        # easyOCR：支持中英文
        logger.info("加载 easyOCR...")
        self.ocr = easyocr.Reader(["en", "ch_sim"], gpu=(self.device.type != "cpu"))

        self._loaded = True
        logger.info("OmniPerception 加载完成（device=%s）", self.device)

    def detect(
        self,
        image: Image.Image,
        bbox_threshold: float = 0.02,
        iou_threshold: float = 0.7,
        caption_batch_size: int = 32,
    ) -> list[UIElement]:
        """从截图提取 UI 元素列表。

        Returns:
            按从上到下、从左到右排序的 UIElement 列表。
        """
        if not self._loaded:
            self.load()

        import cv2
        from torchvision.ops import box_convert
        from torchvision.transforms import ToPILImage

        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        image_np = np.asarray(image)

        # ── 1. OCR ──────────────────────────────────────────────────────────
        ocr_result = self.ocr.readtext(image_np, text_threshold=0.5)
        ocr_texts = [str(item[1]) for item in ocr_result]
        ocr_bboxes_raw = [self._coords_to_xyxy(item[0]) for item in ocr_result]
        # 归一化
        ocr_bboxes_norm = [
            [b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in ocr_bboxes_raw
        ]
        ocr_items = [
            {
                "type": "text",
                "bbox": bbox,
                "interactivity": True,   # 文字元素在 Web 场景也可点击
                "content": text,
            }
            for bbox, text in zip(ocr_bboxes_norm, ocr_texts)
            if self._bbox_area(bbox, w, h) > 0
        ]

        # ── 2. YOLO 图标检测 ─────────────────────────────────────────────────
        yolo_out = self.yolo.predict(
            image,
            imgsz=[h, w],
            conf=bbox_threshold,
            iou=iou_threshold,
            verbose=False,
        )[0]
        if yolo_out.boxes is None:
            icon_items = []
        else:
            xyxy = yolo_out.boxes.xyxy
            xyxy_norm = (xyxy / torch.tensor([w, h, w, h], dtype=xyxy.dtype, device=xyxy.device)).tolist()
            icon_items = [
                {
                    "type": "icon",
                    "bbox": bbox,
                    "interactivity": True,
                    "content": None,
                }
                for bbox in xyxy_norm
                if self._bbox_area(bbox, w, h) > 0
            ]

        # ── 3. 去重合并 ──────────────────────────────────────────────────────
        all_items = self._remove_overlap(icon_items, ocr_items, iou_threshold)

        # ── 4. Florence-2 为图标生成 caption ────────────────────────────────
        icon_indices = [i for i, it in enumerate(all_items) if it["content"] is None]
        if icon_indices:
            bbox_images = []
            for idx in icon_indices:
                b = all_items[idx]["bbox"]
                xmin, xmax = int(b[0] * w), int(b[2] * w)
                ymin, ymax = int(b[1] * h), int(b[3] * h)
                crop = image_np[ymin:ymax, xmin:xmax]
                crop = cv2.resize(crop, (64, 64))
                bbox_images.append(ToPILImage()(crop))

            captions = []
            for start in range(0, len(bbox_images), caption_batch_size):
                batch = bbox_images[start: start + caption_batch_size]
                inputs = self.processor(
                    images=batch,
                    text=["<CAPTION>"] * len(batch),
                    return_tensors="pt",
                    do_resize=False,
                )
                if self.device.type in ("cuda", "mps"):
                    inputs = inputs.to(device=self.device, dtype=torch.float16)
                with torch.inference_mode():
                    generated_ids = self.caption_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=20,
                        num_beams=1,
                        do_sample=False,
                    )
                generated = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                captions.extend([t.strip() for t in generated])

            for idx, cap in zip(icon_indices, captions):
                all_items[idx]["content"] = cap

        # ── 5. 转换为 UIElement ──────────────────────────────────────────────
        elements: list[UIElement] = []
        for i, item in enumerate(all_items):
            b = item["bbox"]
            cx = (b[0] + b[2]) / 2 * w
            cy = (b[1] + b[3]) / 2 * h
            elements.append(
                UIElement(
                    idx=i,
                    label=item.get("content") or "",
                    elem_type=item["type"],
                    bbox=b,
                    center_x=cx,
                    center_y=cy,
                    interactivity=item.get("interactivity", True),
                    source=item.get("source", ""),
                )
            )

        # 从上到下、从左到右排序
        elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
        for i, e in enumerate(elements):
            e.idx = i

        return elements

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _coords_to_xyxy(coords: list) -> list[int]:
        return [
            int(coords[0][0]), int(coords[0][1]),
            int(coords[2][0]), int(coords[2][1]),
        ]

    @staticmethod
    def _bbox_area(bbox: list[float], w: int, h: int) -> float:
        return (bbox[2] - bbox[0]) * w * (bbox[3] - bbox[1]) * h

    @staticmethod
    def _iou(a: list, b: list) -> float:
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter + 1e-9
        r_a = inter / (area_a + 1e-9)
        r_b = inter / (area_b + 1e-9)
        return max(inter / union, r_a, r_b)

    @staticmethod
    def _overlap_ratio(inner: list, outer: list) -> float:
        ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
        iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
        inter = ix * iy
        area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1]) + 1e-9
        return inter / area_inner

    def _remove_overlap(
        self,
        icon_items: list,
        ocr_items: list,
        iou_threshold: float,
    ) -> list:
        result = list(ocr_items)
        for icon in icon_items:
            keep = True
            merged_labels: list[str] = []
            ocr_to_remove = []

            # 只有小图标（面积 < 5% 页面）才吸收 OCR 文字；大容器框不吸收
            b = icon["bbox"]
            icon_area = (b[2] - b[0]) * (b[3] - b[1])
            is_small_icon = icon_area < 0.05

            for ocr in result:
                if is_small_icon:
                    ratio = self._overlap_ratio(ocr["bbox"], icon["bbox"])
                    if ratio > 0.80:
                        merged_labels.append(ocr["content"])
                        ocr_to_remove.append(ocr)
                        continue
                if self._iou(icon["bbox"], ocr["bbox"]) > iou_threshold:
                    keep = False
                    break

            if not keep:
                continue
            for r in ocr_to_remove:
                result.remove(r)
            icon = dict(icon)
            if merged_labels:
                icon["content"] = " ".join(s for s in merged_labels if s)
            result.append(icon)
        return result
