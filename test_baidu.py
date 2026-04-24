import asyncio
from acp.adapters.web_adapter import WebAdapter

async def test_baidu():
    adapter = WebAdapter(headless=False)
    await adapter.start()

    # 1. 打开百度
    print("=== 步骤 1: 打开百度 ===")
    result = await adapter.navigate("https://www.baidu.com")
    print("导航结果: success=" + str(result.success))

    # 2. 感知页面元素
    print("\n=== 步骤 2: 感知页面元素 ===")
    state = await adapter.get_page_state()
    elements = await adapter.get_elements()
    print("页面: " + str(state.title))
    print("元素数量: " + str(len(elements)))

    # 找搜索框和按钮
    search_input = None
    search_btn = None
    for e in elements:
        text = e.text or ""
        placeholder = e.placeholder or ""
        selector = e.selector or ""
        print(f"  [{e.type}] id={e.id} text={repr(text[:30])} placeholder={repr(placeholder[:30])}")
        if e.type == "text_input":
            search_input = e
        if e.type == "button" and ("百度" in text or "submit" in selector.lower()):
            search_btn = e

    print("\n搜索框: " + (search_input.id if search_input else "未找到"))
    print("搜索按钮: " + (search_btn.id if search_btn else "未找到"))

    # 3. 输入搜索词
    if search_input:
        print('\n=== 步骤 3: 输入 "狐狸" ===')
        result = await adapter.type(search_input.id, "狐狸")
        print("输入结果: success=" + str(result.success))

    # 4. 按回车搜索（百度搜索框用回车比点按钮更可靠）
    if search_input:
        print("\n=== 步骤 4: 按回车搜索 ===")
        await adapter._page.keyboard.press("Enter")
        await asyncio.sleep(3)

        # 5. 获取搜索结果
        print("\n=== 步骤 5: 搜索结果页 ===")
        state = await adapter.get_page_state()
        print("页面标题: " + str(state.title))
        print("URL: " + str(state.url))

        elements2 = await adapter.get_elements()
        print("结果页元素数量: " + str(len(elements2)))
        text_elements = [e for e in elements2 if e.text and len(e.text) > 5]
        print("有文本的元素: " + str(len(text_elements)) + " 个，前 15 个:")
        for e in text_elements[:15]:
            print(f"  [{e.type}] {e.text[:80]}")
    else:
        print("\n未找到搜索按钮，尝试回车提交")
        if search_input:
            # 用 Playwright 直接按回车
            await adapter._page.keyboard.press("Enter")
            await asyncio.sleep(3)

            state = await adapter.get_page_state()
            print("\n=== 搜索结果页 ===")
            print("页面标题: " + str(state.title))
            print("URL: " + str(state.url))

            elements2 = await adapter.get_elements()
            text_elements = [e for e in elements2 if e.text and len(e.text) > 5]
            print("有文本的元素: " + str(len(text_elements)) + " 个，前 15 个:")
            for e in text_elements[:15]:
                print(f"  [{e.type}] {e.text[:80]}")

    await adapter.close()
    print("\n=== 测试完成 ===")

asyncio.run(test_baidu())
