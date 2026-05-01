"""
StateDescriber：页面状态文本化。

截图只在本地使用（OmniParser 模式），绝不传给云端 LLM。
Phase 1 默认 DOM 模式（读 data-acp-id + 标准 HTML 元素），对 testenv 精度高且无 GPU 开销。

DOM 模式 vs OmniParser 模式：
  - DOM 模式（phase 1）：要求页面有 data-acp-id 标注或标准 HTML 结构，适合 testenv
  - OmniParser 模式（phase 2，通用）：任意页面，latency +1-2s

README 中已标注边界：
  Phase 1 DOM 描述（要求 data-acp-id），phase 2 切 OmniParser（通用但 latency 更高）
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# DOM 模式的 JS：提取 data-acp-id 元素 + 常用可交互标签
_DOM_EXTRACT_JS = """
() => {
    const elements = [];

    // 1. data-acp-id 标注的元素（testenv 专用）
    document.querySelectorAll('[data-acp-id]').forEach(el => {
        const id = el.getAttribute('data-acp-id');
        const tag = el.tagName.toLowerCase();
        const text = (el.value || el.textContent || '').trim().slice(0, 80);
        const style = window.getComputedStyle(el);
        const visible = style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return;

        elements.push({
            id,
            tag,
            text,
            visible,
            x: Math.round(rect.x + rect.width / 2),
            y: Math.round(rect.y + rect.height / 2),
        });
    });

    // 2. 无 data-acp-id 的 input/button/a/select（补充可交互元素）
    document.querySelectorAll('input, button, a, select, textarea').forEach(el => {
        if (el.getAttribute('data-acp-id')) return;  // 已被上面收集
        const style = window.getComputedStyle(el);
        const visible = style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
        if (!visible) return;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return;
        const text = (el.value || el.textContent || el.placeholder || el.getAttribute('aria-label') || '').trim().slice(0, 60);
        elements.push({
            id: el.id || el.name || null,
            tag: el.tagName.toLowerCase(),
            text,
            visible,
            x: Math.round(rect.x + rect.width / 2),
            y: Math.round(rect.y + rect.height / 2),
        });
    });

    return elements;
}
"""


def _position_label(x: int, y: int, vw: int = 1280, vh: int = 800) -> str:
    xr = x / vw
    yr = y / vh
    x_label = "左侧" if xr < 0.33 else ("右侧" if xr > 0.67 else "中部")
    y_label = "顶部" if yr < 0.25 else ("上半部" if yr < 0.5 else ("下半部" if yr < 0.75 else "底部"))
    return f"{y_label}{x_label}"


class StateDescriber:
    """将当前页面状态描述成自然语言文本，供云端规划 LLM 使用。"""

    async def describe(self, page) -> str:
        """从 Playwright page 对象提取页面状态文本描述。

        返回的字符串只包含文字信息，截图绝不出现在返回值中。
        """
        try:
            elements = await page.evaluate(_DOM_EXTRACT_JS)
        except Exception as exc:
            logger.warning("DOM 提取失败: %s", exc)
            return "（页面状态提取失败）"

        if not elements:
            return "（页面无可识别元素）"

        # 按 y 坐标从上到下排序
        elements.sort(key=lambda e: (e.get("y", 0), e.get("x", 0)))

        lines = ["页面元素（从上到下）："]
        for i, el in enumerate(elements):
            vis = "" if el.get("visible", True) else "（不可见）"
            text = el.get("text", "").strip()
            tag = el.get("tag", "")
            elem_id = el.get("id") or ""
            pos = _position_label(el.get("x", 640), el.get("y", 400))

            # 构造描述
            if text:
                desc = f'[{i}] {tag}"{text}"'
            elif elem_id:
                desc = f"[{i}] {tag}#{elem_id}"
            else:
                desc = f"[{i}] {tag}"

            if elem_id and text:
                desc = f'[{i}] {tag}#{elem_id} "{text}"'

            lines.append(f"  {desc}  位置:{pos}{vis}")

        return "\n".join(lines)
