"""
验证 testenv T9/T10/T11 页面的 data-acp-id 和交互流程。

用法：
    python scripts/verify_testenv.py
    python scripts/verify_testenv.py --base-url http://localhost:8765
    python scripts/verify_testenv.py --file   # 直接用 file:// 协议（不需要 server）

测试场景：
    1. T9 弹窗关闭 → 登录
    2. T10+T11 跨 App 复制粘贴
"""

import asyncio
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


async def verify_popup_login(page, base_url: str) -> bool:
    url = f"{base_url}/pages/popup-login.html"
    print(f"\n[T9] 测试弹窗登录页: {url}")
    await page.goto(url, wait_until="domcontentloaded")

    # 验证弹窗出现
    overlay = page.locator('[data-acp-id="modal-overlay"]')
    assert await overlay.is_visible(), "弹窗未出现"
    print("  [OK] 弹窗可见")

    # 点击关闭按钮
    await page.locator('[data-acp-id="modal-close"]').click()
    await page.wait_for_timeout(300)
    assert not await overlay.is_visible(), "弹窗未关闭"
    print("  [OK] 弹窗已关闭")

    # 填写登录
    await page.locator('[data-acp-id="login-username"]').fill("demo")
    await page.locator('[data-acp-id="login-password"]').fill("123456")
    print("  [OK] 表单填写完成")

    # 提交
    await page.locator('[data-acp-id="login-submit"]').click()
    await page.wait_for_timeout(300)

    # 验证成功
    success = page.locator('[data-acp-id="login-success"]')
    assert await success.is_visible(), "登录成功提示未出现"
    print("  [OK] 登录成功提示可见")

    print("[T9] PASS")
    return True


async def verify_cross_app(context, base_url: str) -> bool:
    print(f"\n[T10+T11] 测试跨 App 复制粘贴")

    # 打开笔记页
    notes_page = await context.new_page()
    await notes_page.goto(f"{base_url}/pages/cross-app/notes-app.html", wait_until="domcontentloaded")

    # 验证 data-acp-id 存在
    for acp_id in ["note-1-copy", "note-2-copy", "note-3-copy", "note-1-content", "copy-status"]:
        el = notes_page.locator(f'[data-acp-id="{acp_id}"]')
        assert await el.count() > 0, f"元素 {acp_id} 不存在"
    print("  [OK] notes-app data-acp-id 全部存在")

    # 点击第一条复制按钮
    note_text = await notes_page.locator('[data-acp-id="note-1-content"]').inner_text()
    note_text = note_text.strip()
    await notes_page.locator('[data-acp-id="note-1-copy"]').click()
    await notes_page.wait_for_timeout(500)

    # 验证按钮状态变化
    btn_text = await notes_page.locator('[data-acp-id="note-1-copy"]').inner_text()
    assert "已复制" in btn_text, f"复制按钮未变为已复制，实际: {btn_text}"
    print("  [OK] 复制按钮状态变化正确")

    # 读取剪贴板
    clipboard_text = await notes_page.evaluate("() => navigator.clipboard.readText()")
    assert note_text[:20] in clipboard_text, "剪贴板内容不匹配"
    print(f"  [OK] 剪贴板内容正确（前 20 字符匹配）")

    # 打开聊天页
    chat_page = await context.new_page()
    await chat_page.goto(f"{base_url}/pages/cross-app/chat-app.html", wait_until="domcontentloaded")

    # 验证 data-acp-id 存在
    for acp_id in ["message-list", "chat-input", "paste-btn", "send-btn"]:
        el = chat_page.locator(f'[data-acp-id="{acp_id}"]')
        assert await el.count() > 0, f"元素 {acp_id} 不存在"
    print("  [OK] chat-app data-acp-id 全部存在")

    # 初始消息数量
    initial_count = await chat_page.locator('[data-acp-type="message"]').count()

    # 点击粘贴按钮
    await chat_page.locator('[data-acp-id="paste-btn"]').click()
    await chat_page.wait_for_timeout(500)

    # 验证输入框有内容
    input_val = await chat_page.locator('[data-acp-id="chat-input"]').input_value()
    assert input_val.strip(), "粘贴后输入框为空"
    print("  [OK] 粘贴成功，输入框有内容")

    # 发送
    await chat_page.locator('[data-acp-id="send-btn"]').click()
    await chat_page.wait_for_timeout(800)

    # 验证消息增加
    new_count = await chat_page.locator('[data-acp-type="message"]').count()
    assert new_count > initial_count, f"发送后消息数未增加（{initial_count} → {new_count}）"
    print(f"  [OK] 消息已发送（{initial_count} → {new_count} 条）")

    await notes_page.close()
    await chat_page.close()
    print("[T10+T11] PASS")
    return True


async def main(base_url: str):
    try:
        from patchright.async_api import async_playwright
    except ImportError:
        from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            permissions=["clipboard-read", "clipboard-write"],
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        results = []
        try:
            results.append(await verify_popup_login(page, base_url))
        except Exception as e:
            print(f"[T9] FAIL: {e}")
            results.append(False)

        try:
            results.append(await verify_cross_app(context, base_url))
        except Exception as e:
            print(f"[T10+T11] FAIL: {e}")
            results.append(False)

        await browser.close()

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"结果: {passed}/{total} 通过")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8765", help="testenv base URL")
    parser.add_argument("--file", action="store_true", help="使用 file:// 协议（无需 server）")
    args = parser.parse_args()

    if args.file:
        testenv_dir = ROOT / "acp" / "testenv"
        base_url = testenv_dir.as_uri()
    else:
        base_url = args.base_url.rstrip("/")

    asyncio.run(main(base_url))
