"""
Fallback 触发规则检测器

检测是否需要从 Tier-2（控件树/DOM）降级到 Tier-3（视觉）。

7 条降级规则：
  1. 控件树基本为空（elements < 5）
  2. Closed Shadow DOM 检测
  3. 验证码检测（页面含验证码关键词/图片）
  4. Canvas 为主内容
  5. 跨域 iframe
  6. 同一 selector 连续失败 3 次（3-Strike）
  7. 专用 MCP 认证失败/限流（外部触发）
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acp.adapters.web_adapter import WebAdapter

logger = logging.getLogger(__name__)


# 验证码关键词（多语言）
_CAPTCHA_KEYWORDS = [
    "captcha", "recaptcha", "hcaptcha", "验证码", "人机验证",
    "robot", "verify you are human", "cf-challenge", "cloudflare",
    "turnstile", "prove you're not a robot",
]

# Closed Shadow DOM 检测脚本
_CLOSED_SHADOW_DOM_JS = """
() => {
    // 尝试检测 closed shadow root（在 customElements 中绑定 attachShadow）
    let closedCount = 0;
    const allElements = document.querySelectorAll('*');
    for (const el of allElements) {
        // 若 shadowRoot 为 null 但有 shadow 宿主标志，可能是 closed
        if (el.shadowRoot === null && el.tagName.includes('-')) {
            // 自定义元素通常有 shadow DOM
            closedCount++;
        }
    }
    return closedCount;
}
"""

# Canvas 占比检测脚本
_CANVAS_DOMINANT_JS = """
() => {
    const canvases = document.querySelectorAll('canvas');
    if (canvases.length === 0) return false;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const viewportArea = vw * vh;
    let canvasArea = 0;
    for (const c of canvases) {
        const r = c.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {
            canvasArea += r.width * r.height;
        }
    }
    // canvas 占视口面积 > 50% 认为是主内容
    return canvasArea / viewportArea > 0.5;
}
"""

# 跨域 iframe 检测脚本
_CROSS_ORIGIN_IFRAME_JS = """
() => {
    const iframes = document.querySelectorAll('iframe');
    for (const f of iframes) {
        try {
            // 若能访问则同源，跨域会抛异常
            const _ = f.contentDocument;
        } catch (e) {
            return true;  // 跨域 iframe 存在
        }
        // 检查 src 域名是否和当前页面不同
        const src = f.getAttribute('src') || '';
        if (src.startsWith('http') && !src.startsWith(location.origin)) {
            return true;
        }
    }
    return false;
}
"""

# 验证码图片/关键词检测脚本
_CAPTCHA_JS = """
(keywords) => {
    const bodyText = document.body ? document.body.innerText.toLowerCase() : '';
    for (const kw of keywords) {
        if (bodyText.includes(kw.toLowerCase())) return {found: true, keyword: kw};
    }
    // 检查图片 alt/src
    const imgs = document.querySelectorAll('img');
    for (const img of imgs) {
        const alt = (img.alt || '').toLowerCase();
        const src = (img.src || '').toLowerCase();
        for (const kw of keywords) {
            const kwl = kw.toLowerCase();
            if (alt.includes(kwl) || src.includes(kwl)) return {found: true, keyword: kw};
        }
    }
    return {found: false, keyword: ''};
}
"""


class FallbackDetector:
    """检测是否需要从 Tier-2（控件树）降级到 Tier-3（视觉）。

    使用方法：
        detector = FallbackDetector()
        should_fb, reason = await detector.should_fallback(adapter)
        if should_fb:
            # 切换到视觉 MCP
            ...
    """

    def __init__(self) -> None:
        # selector → 连续失败次数（规则 6）
        self._selector_fail_count: dict[str, int] = defaultdict(int)

    # ── 规则 6 外部接口 ──────────────────────────────────────────────────────

    def record_selector_failure(self, selector: str) -> None:
        """记录 selector 定位失败（用于规则 6：连续失败 3 次）。"""
        self._selector_fail_count[selector] += 1
        logger.debug("selector_fail_count[%r]=%d", selector, self._selector_fail_count[selector])

    def record_selector_success(self, selector: str) -> None:
        """记录 selector 定位成功（重置失败计数）。"""
        self._selector_fail_count[selector] = 0

    def get_selector_fail_count(self, selector: str) -> int:
        """获取指定 selector 的连续失败次数。"""
        return self._selector_fail_count.get(selector, 0)

    # ── 主检测方法 ────────────────────────────────────────────────────────────

    async def should_fallback(self, adapter: "WebAdapter") -> tuple[bool, str]:
        """检测当前页面是否需要降级到视觉兜底。

        Args:
            adapter: 已启动的 WebAdapter 实例

        Returns:
            (需要降级, 原因描述)
        """
        page = adapter._page
        if page is None:
            return False, ""

        try:
            # 规则 1: 控件树基本为空
            elements = await adapter.get_elements()
            if len(elements) < 5:
                reason = f"规则1: 控件树过少（{len(elements)} 个元素，阈值 5）"
                logger.info("Fallback 触发: %s", reason)
                return True, reason

            # 规则 4: Canvas 为主内容（在规则 2 之前，因为 canvas 页面通常无 DOM）
            canvas_dominant = await page.evaluate(_CANVAS_DOMINANT_JS)
            if canvas_dominant:
                reason = "规则4: Canvas 占据主内容区域（> 50% 视口）"
                logger.info("Fallback 触发: %s", reason)
                return True, reason

            # 规则 2: Closed Shadow DOM 检测
            closed_count = await page.evaluate(_CLOSED_SHADOW_DOM_JS)
            # 若自定义元素多但提取元素少，可能是 closed shadow DOM
            if closed_count > 3 and len(elements) < 20:
                reason = f"规则2: 疑似 Closed Shadow DOM（自定义元素 {closed_count} 个，可提取元素 {len(elements)} 个）"
                logger.info("Fallback 触发: %s", reason)
                return True, reason

            # 规则 3: 验证码检测
            captcha_result = await page.evaluate(_CAPTCHA_JS, _CAPTCHA_KEYWORDS)
            if captcha_result.get("found"):
                reason = f"规则3: 检测到验证码（关键词: {captcha_result.get('keyword', '?')}）"
                logger.info("Fallback 触发: %s", reason)
                return True, reason

            # 规则 5: 跨域 iframe
            has_cross_origin_iframe = await page.evaluate(_CROSS_ORIGIN_IFRAME_JS)
            if has_cross_origin_iframe:
                reason = "规则5: 存在跨域 iframe（可能包含不可读内容）"
                logger.info("Fallback 触发: %s（非致命，仅记录）", reason)
                # 跨域 iframe 本身不一定需要降级，只有当主内容在 iframe 内才需要
                # 此处保守处理：只在元素也不足时才降级
                # return True, reason  # 保守：不直接触发

            # 规则 6: 同一 selector 连续失败 3 次
            for selector, count in self._selector_fail_count.items():
                if count >= 3:
                    reason = f"规则6: selector {repr(selector)} 连续失败 {count} 次"
                    logger.info("Fallback 触发: %s", reason)
                    return True, reason

            return False, ""

        except Exception as exc:
            logger.warning("FallbackDetector 检测失败: %s", exc)
            return False, ""

    # ── 规则 7：专用 MCP 认证失败（外部调用）────────────────────────────────

    def mark_mcp_auth_failure(self) -> tuple[bool, str]:
        """规则 7：专用 MCP 认证失败 → 降级到视觉。"""
        reason = "规则7: 专用 MCP 认证失败或限流"
        logger.info("Fallback 触发: %s", reason)
        return True, reason

    # ── 快速检查接口（不需要 adapter）─────────────────────────────────────────

    @staticmethod
    def check_elements_count(element_count: int, threshold: int = 5) -> tuple[bool, str]:
        """快速检查：元素数量是否不足（规则 1，不需要 adapter）。"""
        if element_count < threshold:
            reason = f"规则1: 控件树过少（{element_count} 个元素，阈值 {threshold}）"
            return True, reason
        return False, ""

    # ── MCP Registry 自动选择（Tier-2 → Tier-3 切换）──────────────────────────

    @staticmethod
    async def get_fallback_mcp(adapter) -> "VisionMCP":
        """当 should_fallback() 返回 True 时，自动创建并返回视觉兜底 MCP。

        Args:
            adapter: WebAdapter 实例（视觉 MCP 需要用来截图和点击）

        Returns:
            已绑定 adapter 的 VisionMCP 实例
        """
        from acp.mcp.tools.vision_mcp import VisionMCP
        vision_mcp = VisionMCP(adapter=adapter)
        logger.info("Fallback 触发：已切换到 VisionMCP（Tier-3）")
        return vision_mcp
