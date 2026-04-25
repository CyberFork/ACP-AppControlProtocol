"""
Web/H5 平台适配器
底层工具：Playwright (CDP 直连)
感知方式：DOM 解析 → ACP Element Schema
操作方式：Playwright API

使用方式（异步上下文管理器）：
    async with WebAdapter() as adapter:
        await adapter.navigate("https://example.com")
        elements = await adapter.get_elements()
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from patchright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from acp.adapters.base import BaseAdapter
from acp.schema.elements import (
    ACPElement,
    ElementSource,
    ElementStates,
    ElementType,
    PageSnapshot,
    PageState,
    Point,
    Rect,
)
from acp.schema.plan import ActionResult


# ---------------------------------------------------------------------------
# HTML tag / role → ElementType 映射
# ---------------------------------------------------------------------------

# (selector_hint, ElementType) — 顺序决定优先级
_TAG_TYPE_MAP: list[tuple[str, ElementType]] = [
    # 按钮类
    ("button", ElementType.BUTTON),
    ("input[type=submit]", ElementType.BUTTON),
    ("input[type=button]", ElementType.BUTTON),
    ("input[type=reset]", ElementType.BUTTON),
    ("[role=button]", ElementType.BUTTON),
    # 文本输入类
    ("input[type=text]", ElementType.TEXT_INPUT),
    ("input[type=search]", ElementType.TEXT_INPUT),
    ("input[type=email]", ElementType.TEXT_INPUT),
    ("input[type=password]", ElementType.TEXT_INPUT),
    ("input[type=number]", ElementType.TEXT_INPUT),
    ("input[type=tel]", ElementType.TEXT_INPUT),
    ("input[type=url]", ElementType.TEXT_INPUT),
    ("textarea", ElementType.TEXT_INPUT),
    # 链接
    ("a", ElementType.BUTTON),          # 链接当按钮处理（可点击）
    # 图片
    ("img", ElementType.IMAGE),
    # 下拉框
    ("select", ElementType.UNKNOWN),    # 先占位，后面 JS 判断会覆盖
    # 复选框
    ("input[type=checkbox]", ElementType.CHECKBOX),
    # 单选框 —— 复用 CHECKBOX 语义
    ("input[type=radio]", ElementType.CHECKBOX),
    # 导航栏
    ("nav", ElementType.NAV_BAR),
    # 列表
    ("ul", ElementType.LIST),
    ("ol", ElementType.LIST),
    ("[role=list]", ElementType.LIST),
    # 列表项
    ("li", ElementType.LIST_ITEM),
    ("[role=listitem]", ElementType.LIST_ITEM),
]

# 简单 tag → ElementType 字典（用于 JS 返回的 tagName 快速查找）
_SIMPLE_TAG_MAP: dict[str, ElementType] = {
    "button": ElementType.BUTTON,
    "textarea": ElementType.TEXT_INPUT,
    "a": ElementType.BUTTON,
    "img": ElementType.IMAGE,
    "select": ElementType.UNKNOWN,
    "nav": ElementType.NAV_BAR,
    "ul": ElementType.LIST,
    "ol": ElementType.LIST,
    "li": ElementType.LIST_ITEM,
}


def _infer_type(tag: str, input_type: str, role: str) -> ElementType:
    """根据 tag、input type、role 推断 ElementType。"""
    tag = (tag or "").lower()
    input_type = (input_type or "").lower()
    role = (role or "").lower()

    # role 优先
    if role == "button":
        return ElementType.BUTTON
    if role in ("list", "listbox"):
        return ElementType.LIST
    if role == "listitem":
        return ElementType.LIST_ITEM
    if role == "checkbox":
        return ElementType.CHECKBOX
    if role == "tab":
        return ElementType.TAB  # M-3: tab 应映射为 TAB，不是 SWITCH
    if role == "navigation":
        return ElementType.NAV_BAR

    # input 细分
    if tag == "input":
        if input_type in ("submit", "button", "reset"):
            return ElementType.BUTTON
        if input_type in ("checkbox",):
            return ElementType.CHECKBOX
        if input_type in ("radio",):
            return ElementType.CHECKBOX
        # 其他 input 默认文本输入
        return ElementType.TEXT_INPUT

    return _SIMPLE_TAG_MAP.get(tag, ElementType.CONTAINER)


def _infer_actions(element_type: ElementType, is_interactive: bool) -> list[str]:
    """根据元素类型推断可执行操作列表。"""
    if element_type == ElementType.BUTTON:
        return ["click"]
    if element_type == ElementType.TEXT_INPUT:
        return ["click", "type", "clear"]
    if element_type == ElementType.CHECKBOX:
        return ["click"]
    if element_type == ElementType.IMAGE:
        return ["click"]
    if element_type in (ElementType.LIST, ElementType.LIST_ITEM):
        return ["click", "scroll"]
    return ["click"] if is_interactive else []


# ---------------------------------------------------------------------------
# 用于在页面内运行的 JS 脚本
# ---------------------------------------------------------------------------

_COLLECT_ELEMENTS_JS = """
() => {
    const INTERACTIVE_TAGS = new Set([
        'a','button','input','select','textarea','label',
        'nav','ul','ol','li'
    ]);
    const VISIBLE_TEXT_TAGS = new Set([
        'h1','h2','h3','h4','h5','h6','p','span','div','section',
        'article','main','header','footer','aside','td','th'
    ]);

    function isVisible(el) {
        const st = window.getComputedStyle(el);
        if (st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity) < 0.01)
            return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    }

    function getLabel(el) {
        // aria-label / aria-labelledby / title / placeholder / alt
        if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
        const lby = el.getAttribute('aria-labelledby');
        if (lby) {
            const lel = document.getElementById(lby);
            if (lel) return lel.innerText.trim();
        }
        if (el.title) return el.title;
        if (el.placeholder) return null;   // placeholder 单独字段
        if (el.alt) return el.alt;
        // 查找关联 label（使用 CSS.escape 避免特殊字符注入）
        if (el.id) {
            const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
            if (lbl) return lbl.innerText.trim();
        }
        return null;
    }

    function getSelector(el) {
        // 尝试 id
        if (el.id) return '#' + CSS.escape(el.id);
        // 尝试 data-testid
        const tid = el.getAttribute('data-testid') || el.getAttribute('data-test-id');
        if (tid) return '[data-testid="' + tid + '"]';
        // 尝试 name 属性（表单元素常用）
        // 对 button 类型，加入所属 form 的 action URL 作为额外指纹（多 Form 消歧义）
        if (el.tagName.match(/^(INPUT|SELECT|TEXTAREA|BUTTON)$/i)) {
            const form = el.form || el.closest('form');
            const formAction = form ? (form.action || form.getAttribute('action') || '') : '';
            if (el.name) {
                const base = el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                // 有 form action 时加入限定，防止同名字段跨表单混淆
                if (formAction) {
                    // 取 action 路径部分（去掉 origin）
                    try {
                        const actionPath = new URL(formAction, location.href).pathname;
                        return 'form[action="' + actionPath + '"] ' + base;
                    } catch(e) { /* ignore */ }
                }
                return base;
            }
            // 无 name 的 button，用 form + type 定位
            if ((el.tagName === 'BUTTON' || el.type === 'submit') && formAction) {
                try {
                    const actionPath = new URL(formAction, location.href).pathname;
                    return 'form[action="' + actionPath + '"] ' + el.tagName.toLowerCase() + '[type="' + (el.type || 'submit') + '"]';
                } catch(e) { /* ignore */ }
            }
        }
        // 生成从下往上的路径
        const parts = [];
        let cur = el;
        let depth = 0;
        while (cur && cur !== document.body && depth < 4) {
            let seg = cur.tagName.toLowerCase();
            const parent = cur.parentElement;
            if (parent) {
                const siblings = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
                if (siblings.length > 1) {
                    const idx = siblings.indexOf(cur) + 1;
                    seg += ':nth-of-type(' + idx + ')';
                }
            }
            parts.unshift(seg);
            cur = parent;
            depth++;
        }
        return parts.join(' > ');
    }

    const results = [];
    const seen = new Set();

    // 收集可交互元素
    const interactiveEls = document.querySelectorAll(
        'a,button,input,select,textarea,[role=button],[role=checkbox],[role=tab],[tabindex]'
    );

    interactiveEls.forEach(el => {
        if (seen.has(el)) return;
        if (!isVisible(el)) return;
        seen.add(el);
        const r = el.getBoundingClientRect();
        results.push({
            tag: el.tagName.toLowerCase(),
            inputType: el.type || '',
            role: el.getAttribute('role') || '',
            text: (el.innerText || el.value || '').trim().slice(0, 200),
            label: getLabel(el),
            placeholder: el.placeholder || null,
            selector: getSelector(el),
            bounds: { x: r.left, y: r.top, width: r.width, height: r.height },
            clickable: true,
            enabled: !el.disabled,
            checked: !!el.checked,
            focused: el === document.activeElement,
            href: el.href || null,
            isInteractive: true
        });
    });

    // 收集有文本的可见元素（静态文本）
    VISIBLE_TEXT_TAGS.forEach(tagName => {
        document.querySelectorAll(tagName).forEach(el => {
            if (seen.has(el)) return;
            if (!isVisible(el)) return;
            const txt = (el.innerText || '').trim();
            if (!txt || txt.length < 2) return;
            seen.add(el);
            const r = el.getBoundingClientRect();
            results.push({
                tag: el.tagName.toLowerCase(),
                inputType: '',
                role: el.getAttribute('role') || '',
                text: txt.slice(0, 200),
                label: el.getAttribute('aria-label') || null,
                placeholder: null,
                selector: getSelector(el),
                bounds: { x: r.left, y: r.top, width: r.width, height: r.height },
                clickable: false,
                enabled: true,
                checked: false,
                focused: false,
                href: null,
                isInteractive: false
            });
        });
    });

    return results;
}
"""


def _make_element_id(page_url: str, index: int, selector: str) -> str:
    """生成稳定但唯一的元素 ID。"""
    raw = f"{page_url}::{index}::{selector}"
    short = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"e{index:04d}_{short}"


# ---------------------------------------------------------------------------
# WebAdapter
# ---------------------------------------------------------------------------


class WebAdapter(BaseAdapter):
    """Web/H5 平台适配器（Playwright 实现）

    推荐使用异步上下文管理器：
        async with WebAdapter() as adapter:
            await adapter.navigate("https://example.com")
    """

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        slow_mo: int = 0,
        cookie_file: str = None,
    ) -> None:
        self._headless = headless
        self._browser_type = browser_type
        self._slow_mo = slow_mo
        self._cookie_file = cookie_file

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        # element_id → selector 缓存
        self._element_cache: dict[str, str] = {}
        # element_id → text 缓存（兼容）
        self._element_text_cache: dict[str, str] = {}
        # element_id → 语义信息缓存（text, placeholder, role, tag）
        self._element_semantic_cache: dict[str, dict] = {}

    # ---- 生命周期 ----

    async def start(self) -> None:
        """启动 Playwright 浏览器实例（伪装为真实浏览器）。"""
        self._playwright = await async_playwright().start()
        browser_launcher = getattr(self._playwright, self._browser_type)
        self._browser = await browser_launcher.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",  # 隐藏 webdriver 标记
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="zh-CN",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            extra_http_headers={
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.1",
            },
        )
        # 在每个页面加载前注入反检测脚本
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            window.chrome = { runtime: {} };
        """)
        self._page = await self._context.new_page()

        # 加载持久化 cookies
        if self._cookie_file:
            cookie_path = Path(self._cookie_file)
            if cookie_path.exists():
                with open(cookie_path) as f:
                    cookies = json.load(f)
                await self._context.add_cookies(cookies)
                logger.info("已加载 cookies: %s", self._cookie_file)

    async def save_cookies(self) -> None:
        """将当前 context 的 cookies 保存到 cookie_file。"""
        if not self._cookie_file or not self._context:
            return
        cookies = await self._context.cookies()
        with open(self._cookie_file, "w") as f:
            json.dump(cookies, f)
        logger.info("已保存 cookies: %s", self._cookie_file)

    async def close(self) -> None:
        """关闭浏览器并释放 Playwright 资源。"""
        if self._cookie_file and self._context:
            await self.save_cookies()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None

    async def __aenter__(self) -> "WebAdapter":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ---- 内部工具 ----

    def _assert_ready(self) -> None:
        if self._page is None:
            raise RuntimeError("WebAdapter 未初始化，请先调用 start() 或使用 async with。")

    def _app_from_url(self, url: str) -> str:
        """从 URL 提取 app 标识（域名主体部分）。"""
        try:
            host = urlparse(url).hostname or ""
            parts = host.split(".")
            return parts[-2] if len(parts) >= 2 else host
        except Exception:
            return "unknown"

    # ---- 平台标识 ----

    @property
    def platform(self) -> str:
        return "web"

    # ---- 感知接口 ----

    async def get_elements(self) -> list[ACPElement]:
        """解析当前页面 DOM，转换为 ACPElement 列表。

        等待策略：
        1. 尝试 networkidle（5s 超时）以等待 SPA 异步渲染
        2. 注入 requestAnimationFrame 等待确保渲染帧完成
        3. 提取 DOM 元素
        """
        self._assert_ready()
        # 等待网络空闲（SPA 异步渲染），超时不报错
        try:
            await self._page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass  # 超时忽略，继续提取

        # 等待一帧渲染完成，确保 SPA 动态内容已更新到 DOM
        try:
            await self._page.evaluate(
                "() => new Promise(resolve => requestAnimationFrame(resolve))"
            )
        except Exception:
            pass

        raw_items: list[dict] = await self._page.evaluate(_COLLECT_ELEMENTS_JS)  # type: ignore[arg-type]

        current_url = self._page.url
        elements: list[ACPElement] = []
        self._element_cache.clear()
        self._element_text_cache.clear()
        self._element_semantic_cache.clear()

        for idx, item in enumerate(raw_items):
            bounds_dict = item.get("bounds", {})
            bounds = Rect(
                x=bounds_dict.get("x", 0),
                y=bounds_dict.get("y", 0),
                width=bounds_dict.get("width", 0),
                height=bounds_dict.get("height", 0),
            )
            center = Point(
                x=bounds.x + bounds.width / 2,
                y=bounds.y + bounds.height / 2,
            )

            elem_type = _infer_type(
                item.get("tag", ""),
                item.get("inputType", ""),
                item.get("role", ""),
            )
            # 若元素无文本、非交互，且 type 仍是 CONTAINER，跳过纯容器
            is_interactive = item.get("isInteractive", False)
            text = item.get("text") or None
            if elem_type == ElementType.CONTAINER and not text and not is_interactive:
                continue

            # 纯文本静态元素
            if not is_interactive and elem_type == ElementType.CONTAINER:
                elem_type = ElementType.TEXT

            selector = item.get("selector", "")
            eid = _make_element_id(current_url, idx, selector)

            states = ElementStates(
                clickable=item.get("clickable", False) or is_interactive,
                enabled=item.get("enabled", True),
                visible=True,
                checked=item.get("checked", False),
                focused=item.get("focused", False),
            )

            actions = _infer_actions(elem_type, is_interactive)

            element = ACPElement(
                id=eid,
                type=elem_type,
                platform_class=item.get("tag", ""),
                text=text,
                label=item.get("label") or None,
                placeholder=item.get("placeholder") or None,
                bounds=bounds,
                center=center,
                states=states,
                selector=selector,
                actions=actions,
                source=ElementSource.DOM,
                confidence=1.0,
            )
            elements.append(element)
            if selector:
                self._element_cache[eid] = selector
            # 缓存语义信息，用于人类化定位
            first_line = (text or "").split("\n")[0].strip()
            if first_line:
                self._element_text_cache[eid] = first_line
            self._element_semantic_cache[eid] = {
                "text": first_line,
                "placeholder": item.get("placeholder") or "",
                "role": item.get("role") or "",
                "tag": item.get("tag") or "",
            }

        return elements

    async def get_page_state(self) -> PageState:
        """获取当前页面状态（URL、标题等）。"""
        self._assert_ready()
        url = self._page.url
        title = await self._page.title()
        return PageState(
            platform="web",
            app=self._app_from_url(url),
            title=title,
            url=url,
        )

    async def screenshot(self) -> bytes:
        """截取当前页面截图，返回 PNG 字节。"""
        self._assert_ready()
        return await self._page.screenshot(type="png", full_page=False)  # type: ignore[return-value]

    async def get_snapshot(self) -> PageSnapshot:
        """获取完整页面快照（状态 + 元素列表）。"""
        page_state = await self.get_page_state()
        elements = await self.get_elements()
        return PageSnapshot(page=page_state, elements=elements)

    # ---- 操作接口 ----

    async def navigate(self, url: str) -> ActionResult:
        """导航到指定 URL，等待页面加载完成。"""
        self._assert_ready()
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page_state = await self.get_page_state()
            return ActionResult(
                success=True,
                data={"url": self._page.url, "title": page_state.title},
                page_state=page_state,
            )
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    async def _locate(self, element_id: str):
        """语义优先的元素定位。

        定位策略（按优先级）：
        1. 有 placeholder → get_by_placeholder
        2. 有 role + text → get_by_role(role, name=text)
        3. 有 text → get_by_text(text, exact=True)
        4. 有 name 属性 selector → CSS selector
        5. fallback → CSS selector + text 二次筛选
        """
        info = self._element_semantic_cache.get(element_id)
        if not info:
            selector = self._element_cache.get(element_id)
            if not selector:
                logger.debug("_locate(%s): 无语义缓存且无 selector，返回 None", element_id)
                return None
            logger.debug("_locate(%s): 无语义缓存，直接用 CSS selector=%r", element_id, selector)
            return self._page.locator(selector).first

        role = info.get("role", "")
        text = info.get("text", "")
        placeholder = info.get("placeholder", "")
        tag = info.get("tag", "")
        selector = self._element_cache.get(element_id, "")

        # 输入框：用 placeholder 定位（最像人类）
        if placeholder and tag in ("input", "textarea"):
            logger.debug("_locate(%s): 策略1 get_by_placeholder(%r)", element_id, placeholder)
            loc = self._page.get_by_placeholder(placeholder, exact=True)
            if await loc.count() == 1:
                return loc
            # placeholder 不唯一，加 CSS 范围限定
            if await loc.count() > 1 and selector:
                logger.debug("_locate(%s): 策略1b placeholder 不唯一，回退 CSS selector=%r", element_id, selector)
                return self._page.locator(selector).first

        # 按钮/链接：用 role + name 定位（最语义化）
        # H-5: 扩展映射表，添加 input/textarea → textbox（ARIA role 标准）
        playwright_role = {
            "button": "button", "link": "link", "a": "link",
            "checkbox": "checkbox", "tab": "tab",
            "textbox": "textbox", "input": "textbox", "textarea": "textbox",
        }.get(role or tag, "")
        if playwright_role and text:
            logger.debug("_locate(%s): 策略2 get_by_role(%r, name=%r)", element_id, playwright_role, text)
            loc = self._page.get_by_role(playwright_role, name=text, exact=True)
            if await loc.count() == 1:
                return loc
            # 不唯一时加 CSS 范围
            if await loc.count() > 1 and selector:
                parent_sel = " > ".join(selector.split(" > ")[:-1]) if " > " in selector else ""
                if parent_sel:
                    logger.debug("_locate(%s): 策略2b role+text 不唯一，缩小范围 parent=%r", element_id, parent_sel)
                    loc = self._page.locator(parent_sel).get_by_role(playwright_role, name=text, exact=True)
                    if await loc.count() >= 1:
                        return loc.first

        # 纯文本匹配
        if text and len(text) < 50:
            logger.debug("_locate(%s): 策略3 get_by_text(%r)", element_id, text)
            loc = self._page.get_by_text(text, exact=True)
            if await loc.count() == 1:
                return loc

        # 策略4: CSS selector + text 筛选
        if selector:
            logger.debug("_locate(%s): 策略4 CSS selector=%r", element_id, selector)
            loc = self._page.locator(selector)
            cnt = await loc.count()
            if cnt == 1:
                return loc
            if cnt > 1 and text:
                filtered = loc.filter(has_text=text)
                if await filtered.count() >= 1:
                    return filtered.first
            if cnt > 1:
                # 多匹配时选 z-index 最高且可见的元素（多 Form 消歧义）
                logger.debug("_locate(%s): 策略4b 多匹配（%d），按 z-index 筛选可见元素", element_id, cnt)
                best = await self._pick_visible_top(loc, cnt)
                return best

        logger.debug("_locate(%s): 所有策略均失败，返回 None", element_id)
        return None

    async def _pick_visible_top(self, loc, count: int):
        """在多个匹配的 locator 中，选择可见且 z-index 最高的那个。

        用于多 Form 消歧义：当同一 selector 匹配多个元素时，
        优先选择可见（display != none, visibility != hidden, aria-hidden != true）
        且 z-index 最高的元素。
        """
        best_idx = 0
        best_zindex = -999999
        for i in range(count):
            item = loc.nth(i)
            try:
                # 检查可见性和 z-index
                is_visible = await item.is_visible()
                if not is_visible:
                    continue
                aria_hidden = await item.get_attribute("aria-hidden")
                if aria_hidden == "true":
                    continue
                z = await item.evaluate(
                    "(el) => { const s = window.getComputedStyle(el); return parseInt(s.zIndex) || 0; }"
                )
                if z > best_zindex:
                    best_zindex = z
                    best_idx = i
            except Exception:
                pass
        return loc.nth(best_idx)

    async def click(self, element_id: str) -> ActionResult:
        """点击指定元素（语义优先定位）。"""
        self._assert_ready()
        locator = await self._locate(element_id)
        if not locator:
            return ActionResult(
                success=False,
                error=f"未找到元素 {element_id}，请先调用 get_elements() 刷新缓存。",
            )
        try:
            await locator.click(timeout=10_000)
            page_state = await self.get_page_state()
            return ActionResult(
                success=True,
                data={"clicked": element_id},
                page_state=page_state,
            )
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    async def type(self, element_id: str, text: str) -> ActionResult:
        """向指定元素输入文本（双重保障）。

        策略：
        1. 点击聚焦
        2. fill() 设置 value（确保表单提交时值正确）
        3. 触发 input/change 事件（满足 JS 事件监听）
        4. 如果 fill() 失败，fallback 到逐字按键输入
        """
        self._assert_ready()
        locator = await self._locate(element_id)
        if not locator:
            return ActionResult(
                success=False,
                error=f"未找到元素 {element_id}，请先调用 get_elements() 刷新缓存。",
            )
        try:
            # 1. 点击聚焦
            await locator.click(timeout=10_000)

            # 2. 尝试 fill()（最可靠地设置 form value）
            try:
                await locator.fill(text, timeout=5_000)
            except Exception:
                # fill 失败（某些框架阻止），fallback 到逐字输入
                await self._page.keyboard.press("Control+a")
                await self._page.keyboard.press("Backspace")
                await locator.press_sequentially(text, delay=50)

            # 3. 手动触发事件（确保 JS 监听器收到通知）
            await locator.dispatch_event("input")
            await locator.dispatch_event("change")

            page_state = await self.get_page_state()
            return ActionResult(
                success=True,
                data={"typed_into": element_id, "text_length": len(text)},
                page_state=page_state,
            )
        except Exception as exc:
            # H-1: 只记录元素 id 和异常类型，不包含实际输入文本（防止密码泄露）
            return ActionResult(
                success=False,
                error=f"type() 失败 element_id={element_id} exc={type(exc).__name__}",
            )

    async def scroll(
        self,
        direction: str,
        element_id: str | None = None,
        amount: int = 300,
    ) -> ActionResult:
        """滚动页面或指定元素。direction: 'up'|'down'|'left'|'right'"""
        self._assert_ready()
        direction = direction.lower()
        delta_map = {
            "up":    (0, -amount),
            "down":  (0,  amount),
            "left":  (-amount, 0),
            "right": ( amount, 0),
        }
        if direction not in delta_map:
            return ActionResult(
                success=False,
                error=f"不支持的滚动方向：{direction}，应为 up/down/left/right",
            )
        dx, dy = delta_map[direction]
        try:
            if element_id:
                locator = await self._locate(element_id)
                if locator:
                    # C-2: 使用参数传递方式，避免 JS 字符串拼接注入风险
                    await locator.evaluate(
                        "(el, args) => el.scrollBy(args.dx, args.dy)",
                        {"dx": dx, "dy": dy},
                    )
                else:
                    await self._page.evaluate(
                        "([dx, dy]) => window.scrollBy(dx, dy)",
                        [dx, dy],
                    )
            else:
                await self._page.evaluate(
                    "([dx, dy]) => window.scrollBy(dx, dy)",
                    [dx, dy],
                )
            page_state = await self.get_page_state()
            return ActionResult(
                success=True,
                data={"direction": direction, "amount": amount},
                page_state=page_state,
            )
        except Exception as exc:
            return ActionResult(success=False, error=str(exc))

    # ---- 等待接口 ----

    async def wait_for_element(
        self,
        selector: str,
        timeout: int = 10,
    ) -> ACPElement | None:
        """等待指定 CSS 选择器元素出现，返回对应 ACPElement（若已在缓存）。"""
        self._assert_ready()
        try:
            locator = self._page.locator(selector)
            await locator.wait_for(state="visible", timeout=timeout * 1000)
            # 从缓存中查找匹配的元素
            for eid, sel in self._element_cache.items():
                if sel == selector:
                    # 找到了，重新获取元素信息
                    elements = await self.get_elements()
                    for el in elements:
                        if el.selector == selector:
                            return el
            # 未在缓存中，刷新并重找
            elements = await self.get_elements()
            for el in elements:
                if el.selector == selector:
                    return el
            return None
        except Exception:
            return None

    async def wait_for_navigation(self, timeout: int = 30) -> PageState:
        """等待页面导航/加载完成（networkidle）。"""
        self._assert_ready()
        try:
            await self._page.wait_for_load_state("networkidle", timeout=timeout * 1000)
        except Exception:
            pass  # 超时也尽量返回当前状态
        return await self.get_page_state()

    # ---- 扩展工具方法 ----

    async def find_element_by_text(self, text: str) -> ACPElement | None:
        """通过文本内容查找元素（精确 or 包含匹配）。"""
        elements = await self.get_elements()
        # 精确匹配
        for el in elements:
            if el.text == text or el.label == text:
                return el
        # 包含匹配
        for el in elements:
            if (el.text and text in el.text) or (el.label and text in el.label):
                return el
        return None

    async def clear_and_type(self, element_id: str, text: str) -> ActionResult:
        """先清空再输入文本（使用语义定位 _locate()，与 type() 保持一致）。"""
        self._assert_ready()
        locator = await self._locate(element_id)
        if not locator:
            return ActionResult(
                success=False,
                error=f"未找到元素 {element_id}，请先调用 get_elements() 刷新缓存。",
            )
        try:
            await locator.click(timeout=10_000)
            await locator.fill("", timeout=5_000)
            await locator.fill(text, timeout=10_000)
            await locator.dispatch_event("input")
            await locator.dispatch_event("change")
            page_state = await self.get_page_state()
            return ActionResult(
                success=True,
                data={"typed_into": element_id, "text_length": len(text)},
                page_state=page_state,
            )
        except Exception as exc:
            # H-1: 只记录元素 id 和异常类型，不包含实际输入文本
            return ActionResult(
                success=False,
                error=f"clear_and_type() 失败 element_id={element_id} exc={type(exc).__name__}",
            )
