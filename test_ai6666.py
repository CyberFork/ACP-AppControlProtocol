import asyncio
import yaml
from acp.adapters.web_adapter import WebAdapter

async def test_ai6666():
    # 读取凭证
    with open("acp/config/credentials.yaml") as f:
        creds = yaml.safe_load(f)
    site_creds = creds["sites"]["ai6666"]
    email = site_creds["email"]
    password = site_creds["password"]

    adapter = WebAdapter(headless=False)
    await adapter.start()

    # 1. 打开网站
    print("=== 步骤 1: 打开 ai6666.ai ===")
    result = await adapter.navigate("https://ai6666.ai")
    print("导航结果: success=" + str(result.success))
    await asyncio.sleep(2)

    # 2. 感知页面
    print("\n=== 步骤 2: 感知页面元素 ===")
    state = await adapter.get_page_state()
    elements = await adapter.get_elements()
    print("页面标题: " + str(state.title))
    print("URL: " + str(state.url))
    print("元素数量: " + str(len(elements)))

    # 列出所有可交互元素
    print("\n--- 可交互元素 ---")
    for e in elements:
        text = e.text or ""
        placeholder = e.placeholder or ""
        if e.type in ("text_input", "button", "link", "checkbox",
                       "ElementType.TEXT_INPUT", "ElementType.BUTTON",
                       "ElementType.LINK", "ElementType.CHECKBOX"):
            print(f"  [{e.type}] id={e.id} text={repr(text[:50])} placeholder={repr(placeholder[:50])} selector={e.selector}")
        elif text and len(text) < 50:
            print(f"  [{e.type}] id={e.id} text={repr(text[:50])}")

    # 3. 尝试找登录相关元素
    print("\n--- 查找登录元素 ---")
    email_input = None
    password_input = None
    login_btn = None

    for e in elements:
        etype = str(e.type)
        text = (e.text or "").lower()
        placeholder = (e.placeholder or "").lower()
        selector = (e.selector or "").lower()

        if "text_input" in etype:
            if any(k in placeholder + selector for k in ["email", "mail", "用户", "账号", "user", "login"]):
                email_input = e
                print(f"  找到邮箱输入框: {e.id} placeholder={repr(e.placeholder)}")
            elif any(k in placeholder + selector + (e.text or "").lower() for k in ["password", "密码", "pass"]):
                password_input = e
                print(f"  找到密码输入框: {e.id} placeholder={repr(e.placeholder)}")
            elif not email_input:
                email_input = e
                print(f"  可能的邮箱输入框: {e.id} placeholder={repr(e.placeholder)}")
            elif not password_input:
                password_input = e
                print(f"  可能的密码输入框: {e.id}")

        if "button" in etype and any(k in text for k in ["登录", "login", "sign in", "log in"]):
            login_btn = e
            print(f"  找到登录按钮: {e.id} text={repr(e.text)}")

    # 4. 尝试登录
    if email_input:
        print(f"\n=== 步骤 3: 输入邮箱 ===")
        result = await adapter.type(email_input.id, email)
        print("输入邮箱: success=" + str(result.success))

    if password_input:
        print(f"\n=== 步骤 4: 输入密码 ===")
        result = await adapter.type(password_input.id, password)
        print("输入密码: success=" + str(result.success))

    if login_btn:
        print(f"\n=== 步骤 5: 点击登录 ===")
        result = await adapter.click(login_btn.id)
        print("点击登录: success=" + str(result.success))
        await asyncio.sleep(3)

        state = await adapter.get_page_state()
        print("登录后页面: " + str(state.title))
        print("URL: " + str(state.url))

    if not login_btn and not email_input:
        print("\n未找到登录表单，可能需要先点击登录按钮/链接")
        # 找任何包含"登录"的可点击元素
        for e in elements:
            text = (e.text or "")
            if "登录" in text or "Login" in text or "Sign" in text:
                print(f"  发现: [{e.type}] id={e.id} text={repr(text[:50])}")

    # 保持浏览器打开一会让你看
    print("\n浏览器将保持 10 秒供查看...")
    await asyncio.sleep(10)

    await adapter.close()
    print("\n=== 测试完成 ===")

asyncio.run(test_ai6666())
