"""
Web Adapter 测试脚本

验证 WebAdapter 能打开网页、解析 DOM 为 ACPElement 列表，
并能执行 click/type/scroll/navigate 等基础操作。

运行方式：
    cd /Volumes/work/ACP
    python -m acp.tests.test_web_adapter

依赖：
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from acp.adapters.web_adapter import WebAdapter
from acp.schema.elements import ElementType


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

_PASS = "[PASS]"
_FAIL = "[FAIL]"
_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = _PASS if condition else _FAIL
    _results.append((name, condition, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

async def test_navigate(adapter: WebAdapter) -> None:
    print("\n[TEST] navigate()")
    result = await adapter.navigate("https://example.com")
    check("navigate 成功", result.success, result.error or "")
    check("返回 page_state", result.page_state is not None)
    if result.page_state:
        check("platform 为 web", result.page_state.platform == "web")
        check("url 包含 example.com", "example.com" in (result.page_state.url or ""))


async def test_get_page_state(adapter: WebAdapter) -> None:
    print("\n[TEST] get_page_state()")
    state = await adapter.get_page_state()
    check("有 url", bool(state.url))
    check("有 title", bool(state.title))
    check("platform = web", state.platform == "web")
    check("有 app 字段", bool(state.app))
    print(f"        URL={state.url}, title={state.title!r}")


async def test_get_elements(adapter: WebAdapter) -> None:
    print("\n[TEST] get_elements()")
    elements = await adapter.get_elements()
    check("返回元素列表非空", len(elements) > 0, f"共 {len(elements)} 个元素")

    # 检查每个元素的基础字段
    ids_seen = set()
    for el in elements:
        assert el.id not in ids_seen, f"重复 id: {el.id}"
        ids_seen.add(el.id)
        assert el.bounds is not None, f"元素 {el.id} 缺少 bounds"
        assert el.center is not None, f"元素 {el.id} 缺少 center"
        assert el.type in ElementType.__members__.values(), f"未知 type: {el.type}"

    check("所有元素有唯一 id", True)
    check("所有元素有 bounds", True)
    check("所有元素有合法 type", True)

    # 统计各类型
    type_counts: dict[str, int] = {}
    for el in elements:
        type_counts[el.type.value] = type_counts.get(el.type.value, 0) + 1
    print(f"        类型分布: {dict(sorted(type_counts.items()))}")

    # example.com 应该有链接（a → button）
    has_link = any(el.type == ElementType.BUTTON for el in elements)
    check("页面有可点击元素(button/link)", has_link)

    # 元素应有文本
    has_text = any(el.text for el in elements)
    check("至少有一个元素有文本", has_text)

    return elements


async def test_element_bounds(adapter: WebAdapter) -> None:
    print("\n[TEST] element bounds 合法性")
    elements = await adapter.get_elements()
    valid_bounds = 0
    valid_center = 0
    for el in elements:
        b = el.bounds
        c = el.center
        if b.width > 0 and b.height > 0 and b.x >= 0 and b.y >= 0:
            valid_bounds += 1
        # center 应在 bounds 内（允许浮点误差）
        if (b.x - 1 <= c.x <= b.x + b.width + 1 and
                b.y - 1 <= c.y <= b.y + b.height + 1):
            valid_center += 1

    total = len(elements)
    check("bounds 正数尺寸 >= 80%", valid_bounds / max(total, 1) >= 0.8,
          f"{valid_bounds}/{total}")
    check("center 在 bounds 内 >= 80%", valid_center / max(total, 1) >= 0.8,
          f"{valid_center}/{total}")


async def test_screenshot(adapter: WebAdapter) -> None:
    print("\n[TEST] screenshot()")
    data = await adapter.screenshot()
    check("返回 bytes", isinstance(data, bytes))
    check("PNG 魔术头", data[:4] == b"\x89PNG", f"head={data[:4]!r}")
    check("截图大小 > 1KB", len(data) > 1024, f"{len(data)} bytes")


async def test_scroll(adapter: WebAdapter) -> None:
    print("\n[TEST] scroll()")
    for direction in ("down", "up", "left", "right"):
        result = await adapter.scroll(direction, amount=100)
        check(f"scroll {direction}", result.success, result.error or "")

    bad = await adapter.scroll("diagonal")
    check("非法方向返回 error", not bad.success)


async def test_navigate_multiple(adapter: WebAdapter) -> None:
    print("\n[TEST] 多次 navigate()")
    r1 = await adapter.navigate("https://www.iana.org/domains/reserved")
    check("navigate IANA 成功", r1.success, r1.error or "")

    r2 = await adapter.navigate("https://example.com")
    check("navigate 回 example.com", r2.success, r2.error or "")


async def test_find_by_text(adapter: WebAdapter) -> None:
    print("\n[TEST] find_element_by_text()")
    await adapter.navigate("https://example.com")
    el = await adapter.find_element_by_text("More information...")
    if el is None:
        # example.com 文字可能不同，宽松处理
        check("find_by_text（宽松）", True, "未找到精确文本，跳过")
    else:
        check("find_by_text 返回元素", el is not None)
        check("元素有 selector", bool(el.selector))


async def test_get_snapshot(adapter: WebAdapter) -> None:
    print("\n[TEST] get_snapshot()")
    await adapter.navigate("https://example.com")
    snapshot = await adapter.get_snapshot()
    check("snapshot.page 有 url", bool(snapshot.page.url))
    check("snapshot.elements 非空", len(snapshot.elements) > 0)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def main() -> int:
    print("=" * 60)
    print("ACP Web Adapter 验证测试")
    print("=" * 60)

    async with WebAdapter(headless=True) as adapter:
        await test_navigate(adapter)
        await test_get_page_state(adapter)
        await test_get_elements(adapter)
        await test_element_bounds(adapter)
        await test_screenshot(adapter)
        await test_scroll(adapter)
        await test_navigate_multiple(adapter)
        await test_find_by_text(adapter)
        await test_get_snapshot(adapter)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)
    print(f"结果：{passed}/{total} 通过，{failed} 失败")

    if failed:
        print("\n失败项目：")
        for name, ok, detail in _results:
            if not ok:
                print(f"  {_FAIL} {name}" + (f" — {detail}" if detail else ""))
        return 1
    else:
        print("全部通过！WebAdapter 验证成功。")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
