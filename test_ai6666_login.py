"""
ACP 测试：ai6666.ai 完整流程
  1. 打开网站
  2. 点击登录
  3. 输入账号密码并登录
  4. 回到首页，找一个碳基圈帖子
  5. 进入帖子，发表评论 "ACP测试"

运行方式：
  cd /Volumes/work/ACP
  python3 test_ai6666_login.py
"""
import asyncio
import yaml
from acp.adapters.web_adapter import WebAdapter


def load_credentials():
    with open("acp/config/sites/ai6666/credentials.yaml") as f:
        creds = yaml.safe_load(f)
    return creds["auth"]["email"], creds["auth"]["password"]


def find_element(elements, **criteria):
    """在元素列表中查找匹配的元素。

    criteria 支持:
      type=    精确匹配元素类型（字符串包含）
      text=    文本包含
      placeholder=  placeholder 包含
      selector=     selector 包含
    """
    for e in elements:
        etype = str(e.type)
        text = (e.text or "").strip()
        placeholder = (e.placeholder or "")
        selector = (e.selector or "")

        match = True
        if "type" in criteria and criteria["type"] not in etype.lower():
            match = False
        if "text" in criteria and criteria["text"] not in text:
            match = False
        if "placeholder" in criteria and criteria["placeholder"] not in placeholder.lower():
            match = False
        if "selector" in criteria and criteria["selector"] not in selector.lower():
            match = False
        if match:
            return e
    return None


def find_elements(elements, **criteria):
    """查找所有匹配的元素。"""
    results = []
    for e in elements:
        etype = str(e.type)
        text = (e.text or "").strip()
        placeholder = (e.placeholder or "")
        selector = (e.selector or "")

        match = True
        if "type" in criteria and criteria["type"] not in etype.lower():
            match = False
        if "text" in criteria and criteria["text"] not in text:
            match = False
        if "placeholder" in criteria and criteria["placeholder"] not in placeholder.lower():
            match = False
        if "selector" in criteria and criteria["selector"] not in selector.lower():
            match = False
        if match:
            results.append(e)
    return results


def print_step(n, title):
    print(f"\n{'='*60}")
    print(f"  步骤 {n}: {title}")
    print(f"{'='*60}")


def print_elements_summary(elements, label="元素"):
    """打印元素摘要，只显示可交互的。"""
    interactive = [e for e in elements if "button" in str(e.type).lower()
                   or "input" in str(e.type).lower()
                   or "text_input" in str(e.type).lower()]
    print(f"  {label}: 共 {len(elements)} 个，其中可交互 {len(interactive)} 个")
    for e in interactive[:20]:
        text = (e.text or "")[:40]
        print(f"    [{str(e.type).split('.')[-1]:12}] id={e.id} text={repr(text)}")


async def main():
    email, password = load_credentials()
    adapter = WebAdapter(headless=False)
    await adapter.start()

    try:
        # ============================================================
        # 步骤 1: 打开 ai6666.ai
        # ============================================================
        print_step(1, "打开 ai6666.ai")
        result = await adapter.navigate("https://ai6666.ai")
        await asyncio.sleep(2)
        state = await adapter.get_page_state()
        print(f"  页面: {state.title}")
        print(f"  URL: {state.url}")

        # ============================================================
        # 步骤 2: 点击登录按钮（导航栏）
        # ============================================================
        print_step(2, "点击登录按钮")
        elements = await adapter.get_elements()
        print_elements_summary(elements, "首页元素")

        login_link = find_element(elements, text="登录", type="button")
        if not login_link:
            print("  ❌ 未找到登录按钮！")
            return
        print(f"  ✅ 找到登录按钮: {login_link.id}")

        result = await adapter.click(login_link.id)
        print(f"  点击结果: {'成功' if result.success else '失败: ' + str(result.error)}")
        await asyncio.sleep(2)

        # ============================================================
        # 步骤 3: 登录页 - 输入账号密码
        # ============================================================
        print_step(3, "输入账号密码")
        state = await adapter.get_page_state()
        print(f"  页面: {state.title}")
        print(f"  URL: {state.url}")

        elements = await adapter.get_elements()
        print_elements_summary(elements, "登录页元素")

        # 找输入框
        text_inputs = find_elements(elements, type="text_input")
        print(f"  发现 {len(text_inputs)} 个输入框:")
        for inp in text_inputs:
            print(f"    id={inp.id} placeholder={repr((inp.placeholder or '')[:40])} selector={inp.selector}")

        # 通常第一个是邮箱/用户名，第二个是密码
        email_input = None
        password_input = None
        for inp in text_inputs:
            p = (inp.placeholder or "").lower()
            s = (inp.selector or "").lower()
            if any(k in p + s for k in ["email", "mail", "邮箱", "用户", "账号", "user", "phone", "手机"]):
                email_input = inp
            elif any(k in p + s for k in ["password", "密码", "pass"]):
                password_input = inp

        # fallback: 按顺序分配
        if not email_input and len(text_inputs) >= 1:
            email_input = text_inputs[0]
        if not password_input and len(text_inputs) >= 2:
            password_input = text_inputs[1]

        if email_input:
            print(f"  ✅ 邮箱框: {email_input.id}")
            result = await adapter.type(email_input.id, email)
            print(f"  输入邮箱: {'成功' if result.success else '失败: ' + str(result.error)}")
        else:
            print("  ❌ 未找到邮箱输入框！")

        if password_input:
            print(f"  ✅ 密码框: {password_input.id}")
            result = await adapter.type(password_input.id, password)
            print(f"  输入密码: {'成功' if result.success else '失败: ' + str(result.error)}")
        else:
            print("  ❌ 未找到密码输入框！")

        # ============================================================
        # 步骤 4: 点击登录提交
        # ============================================================
        print_step(4, "点击登录提交")
        # 重新获取元素（输入后 DOM 可能变化）
        elements = await adapter.get_elements()

        # 找登录/提交按钮
        submit_btn = find_element(elements, text="登录", type="button")
        # 如果有多个"登录"，找不是导航栏那个的
        submit_candidates = find_elements(elements, text="登录", type="button")
        if len(submit_candidates) > 1:
            # 选 selector 里有 submit/form/btn 的，或者选最后一个
            for c in submit_candidates:
                s = (c.selector or "").lower()
                if "submit" in s or "form" in s:
                    submit_btn = c
                    break
            else:
                submit_btn = submit_candidates[-1]  # 通常表单里的在后面

        if submit_btn:
            print(f"  ✅ 提交按钮: {submit_btn.id} text={repr(submit_btn.text)}")
            result = await adapter.click(submit_btn.id)
            print(f"  点击结果: {'成功' if result.success else '失败: ' + str(result.error)}")
        else:
            print("  未找到提交按钮，尝试回车提交")
            await adapter._page.keyboard.press("Enter")

        await asyncio.sleep(3)

        # 检查登录结果
        state = await adapter.get_page_state()
        print(f"  登录后页面: {state.title}")
        print(f"  URL: {state.url}")

        # ============================================================
        # 步骤 5: 回到首页，找碳基圈帖子
        # ============================================================
        print_step(5, "找碳基圈帖子")

        # 如果不在首页，导航回去
        if "ai6666" not in (state.url or ""):
            await adapter.navigate("https://ai6666.ai")
            await asyncio.sleep(2)

        elements = await adapter.get_elements()
        print_elements_summary(elements, "首页元素")

        # 找帖子链接 - 通常是包含用户名和内容的可点击元素
        # 找第一个看起来像帖子的元素（有内容文本的链接/按钮）
        post_link = None
        for e in elements:
            text = (e.text or "")
            etype = str(e.type).lower()
            # 跳过导航栏元素、纯数字、太短的
            if "button" in etype and len(text) > 10 and "换一换" not in text \
                and "APP" not in text and "邀请" not in text and "登录" not in text \
                and "注册" not in text and "文心" not in text and "更多" not in text \
                and "首页" not in text and "音乐" not in text and "🎵 我也来" not in text \
                and "ICP" not in text and "Web 5.0" not in text:
                # 看起来像帖子内容或用户名
                post_link = e
                break

        if post_link:
            print(f"  ✅ 找到帖子: {post_link.id} text={repr((post_link.text or '')[:60])}")
            result = await adapter.click(post_link.id)
            print(f"  点击结果: {'成功' if result.success else '失败: ' + str(result.error)}")
            await asyncio.sleep(2)
        else:
            print("  ❌ 未找到帖子，尝试点击第一个内容区域")
            # 滚动一下看看
            await adapter.scroll("down")
            await asyncio.sleep(1)
            elements = await adapter.get_elements()
            # 再试一次
            for e in elements:
                text = (e.text or "")
                if len(text) > 15 and "button" in str(e.type).lower():
                    post_link = e
                    break
            if post_link:
                result = await adapter.click(post_link.id)
                await asyncio.sleep(2)

        # ============================================================
        # 步骤 6: 在帖子页发表评论
        # ============================================================
        print_step(6, "发表评论 'ACP测试'")
        state = await adapter.get_page_state()
        print(f"  页面: {state.title}")
        print(f"  URL: {state.url}")

        elements = await adapter.get_elements()
        print_elements_summary(elements, "帖子页元素")

        # 找评论输入框
        comment_input = None
        text_inputs = find_elements(elements, type="text_input")
        for inp in text_inputs:
            p = (inp.placeholder or "").lower()
            s = (inp.selector or "").lower()
            if any(k in p for k in ["评论", "comment", "回复", "说点", "写"]):
                comment_input = inp
                break

        # fallback: 任何 textarea 或 text_input
        if not comment_input and text_inputs:
            comment_input = text_inputs[0]

        if comment_input:
            print(f"  ✅ 评论框: {comment_input.id} placeholder={repr((comment_input.placeholder or '')[:40])}")

            # 先点击聚焦
            await adapter.click(comment_input.id)
            await asyncio.sleep(0.5)

            result = await adapter.type(comment_input.id, "ACP测试")
            print(f"  输入评论: {'成功' if result.success else '失败: ' + str(result.error)}")

            # 找发送/提交按钮
            await asyncio.sleep(1)
            elements = await adapter.get_elements()

            send_btn = None
            for e in elements:
                text = (e.text or "")
                if "button" in str(e.type).lower() and any(k in text for k in ["发送", "提交", "发表", "评论", "回复"]):
                    send_btn = e
                    break

            if send_btn:
                print(f"  ✅ 发送按钮: {send_btn.id} text={repr(send_btn.text)}")
                result = await adapter.click(send_btn.id)
                print(f"  发送结果: {'成功' if result.success else '失败: ' + str(result.error)}")
                await asyncio.sleep(2)
            else:
                print("  未找到发送按钮，尝试 Ctrl+Enter 提交")
                await adapter._page.keyboard.press("Control+Enter")
                await asyncio.sleep(2)

            # 验证
            state = await adapter.get_page_state()
            print(f"  当前页面: {state.title}")
        else:
            print("  ❌ 未找到评论输入框")
            print("  列出所有输入框:")
            for inp in text_inputs:
                print(f"    id={inp.id} placeholder={repr(inp.placeholder)} selector={inp.selector}")
            print("  可能需要先滚动到评论区域")

        # ============================================================
        # 完成
        # ============================================================
        print(f"\n{'='*60}")
        print(f"  流程完成！浏览器保持 15 秒供查看")
        print(f"{'='*60}")
        await asyncio.sleep(15)

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
