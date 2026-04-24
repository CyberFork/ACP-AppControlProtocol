"""
ACP 智能测试：ai6666.ai 完整流程（LLM 辅助决策）
  1. 打开网站
  2. 登录
  3. 找碳基圈帖子
  4. 评论 "ACP测试"

LLM 参与：每步操作前让 DeepSeek 分析页面元素，决定操作目标。
失败时 LLM 分析原因并给出替代方案。

运行方式：
  cd /Volumes/work/ACP
  python3 test_ai6666_smart.py
"""
import asyncio
import json
import os
import yaml
from urllib.request import Request, urlopen

from acp.adapters.web_adapter import WebAdapter


# ── 配置加载 ──────────────────────────────────────────────

def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)

def load_credentials():
    with open("acp/config/sites/ai6666/credentials.yaml") as f:
        creds = yaml.safe_load(f)
    return creds["auth"]["email"], creds["auth"]["password"]


# ── LLM 调用 ─────────────────────────────────────────────

def ask_llm(prompt: str, max_tokens: int = 500) -> str:
    """调用 DeepSeek LLM，返回文本回复。"""
    api_key = os.environ.get("ACP_LLM_API_KEY", "")
    base_url = os.environ.get("ACP_LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("ACP_LLM_MODEL", "deepseek-chat")

    req = Request(
        base_url + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    resp = json.loads(urlopen(req, timeout=30).read())
    return resp["choices"][0]["message"]["content"].strip()


def ask_llm_json(prompt: str) -> dict:
    """调用 LLM 并解析 JSON 回复。"""
    raw = ask_llm(prompt, max_tokens=800)
    # 提取 JSON（可能被 ```json ``` 包裹）
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return json.loads(raw.strip())


# ── 页面元素摘要（给 LLM 看的） ──────────────────────────

def elements_for_llm(elements, max_count=60) -> str:
    """将元素列表转成 LLM 可读的文本，只保留关键信息。"""
    lines = []
    for e in elements[:max_count]:
        etype = str(e.type).split(".")[-1]
        text = (e.text or "").replace("\n", " ").strip()[:60]
        placeholder = (e.placeholder or "").strip()[:40]
        selector = (e.selector or "")[:60]
        line = f"  id={e.id} type={etype} text={repr(text)}"
        if placeholder:
            line += f" placeholder={repr(placeholder)}"
        lines.append(line)
    result = "\n".join(lines)
    if len(elements) > max_count:
        result += f"\n  ...（还有 {len(elements) - max_count} 个元素未列出）"
    return result


# ── 智能操作：LLM 选择 + 执行 + 重试 ────────────────────

async def smart_action(adapter, task_description: str, elements=None, max_retries=2) -> dict:
    """让 LLM 分析页面元素并决定操作。失败时重试。

    返回: {"success": bool, "action": str, "element_id": str, "detail": str}
    """
    if elements is None:
        elements = await adapter.get_elements()

    elem_text = elements_for_llm(elements)
    state = await adapter.get_page_state()

    last_error = ""
    for attempt in range(max_retries + 1):
        prompt = f"""你是一个 Web 自动化助手。当前页面信息：
- 标题: {state.title}
- URL: {state.url}

任务: {task_description}

当前页面的可用元素：
{elem_text}

请分析页面元素，决定执行什么操作来完成任务。返回 JSON 格式：
{{
  "action": "click" 或 "type" 或 "scroll" 或 "navigate" 或 "press_key",
  "element_id": "目标元素 id（click/type 时必填）",
  "text": "要输入的文本（type 时必填）",
  "key": "要按的键（press_key 时必填，如 Enter）",
  "url": "要导航的 URL（navigate 时必填）",
  "reasoning": "为什么选择这个操作（简短说明）"
}}

注意：
- 只返回 JSON，不要其他内容
- element_id 必须从上面的列表中选择，不要编造
- 如果找不到合适的元素，action 设为 "scroll" 向下滚动查找"""

        if attempt > 0:
            prompt += f"\n\n上一次尝试失败了：{last_error}\n请选择其他元素或方法。"

        try:
            decision = ask_llm_json(prompt)
        except Exception as exc:
            print(f"  ⚠️ LLM 调用失败: {exc}")
            return {"success": False, "detail": f"LLM 调用失败: {exc}"}

        action = decision.get("action", "")
        element_id = decision.get("element_id", "")
        reasoning = decision.get("reasoning", "")
        print(f"  🤖 LLM 决策 (尝试 {attempt+1}): {action} → {element_id}")
        print(f"     理由: {reasoning}")

        result = None
        try:
            if action == "click" and element_id:
                result = await adapter.click(element_id)
            elif action == "type" and element_id:
                text = decision.get("text", "")
                result = await adapter.type(element_id, text)
            elif action == "scroll":
                result = await adapter.scroll("down")
                await asyncio.sleep(1)
                # 滚动后刷新元素
                elements = await adapter.get_elements()
                elem_text = elements_for_llm(elements)
                continue  # 滚动后重新让 LLM 决策
            elif action == "press_key":
                key = decision.get("key", "Enter")
                await adapter._page.keyboard.press(key)
                result = type("R", (), {"success": True, "error": None})()
            elif action == "navigate":
                url = decision.get("url", "")
                result = await adapter.navigate(url)
            else:
                last_error = f"未知操作: {action}"
                continue

            if result and result.success:
                return {
                    "success": True,
                    "action": action,
                    "element_id": element_id,
                    "detail": reasoning,
                }
            else:
                last_error = result.error if result else "操作返回失败"
                print(f"  ⚠️ 操作失败: {last_error}")

        except Exception as exc:
            last_error = str(exc)
            print(f"  ⚠️ 异常: {last_error}")

    return {"success": False, "detail": f"重试 {max_retries+1} 次后仍失败: {last_error}"}


# ── 主流程 ────────────────────────────────────────────────

def print_step(n, title):
    print(f"\n{'='*60}")
    print(f"  步骤 {n}: {title}")
    print(f"{'='*60}")


async def main():
    load_env()
    email, password = load_credentials()

    adapter = WebAdapter(headless=False)
    await adapter.start()

    try:
        # ── 步骤 1: 打开网站 ──
        print_step(1, "打开 ai6666.ai")
        await adapter.navigate("https://ai6666.ai")
        await asyncio.sleep(2)
        state = await adapter.get_page_state()
        print(f"  页面: {state.title}")
        print(f"  URL: {state.url}")

        # ── 步骤 2: 点击登录 ──
        print_step(2, "点击登录按钮")
        result = await smart_action(
            adapter,
            "点击页面上的'登录'按钮或链接，进入登录页面。不要点'注册'。"
        )
        if not result["success"]:
            # fallback: 直接导航到登录页
            print("  尝试直接导航到登录页...")
            await adapter.navigate("https://ai6666.com/login")
        await asyncio.sleep(2)

        state = await adapter.get_page_state()
        print(f"  当前页面: {state.title} | URL: {state.url}")

        # ── 步骤 3: 输入邮箱 ──
        print_step(3, "输入邮箱")
        elements = await adapter.get_elements()
        result = await smart_action(
            adapter,
            f"在登录表单中找到邮箱/用户名输入框，输入: {email}",
            elements=elements,
        )
        print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败: ' + result['detail']}")

        # ── 步骤 4: 输入密码 ──
        print_step(4, "输入密码")
        elements = await adapter.get_elements()
        result = await smart_action(
            adapter,
            f"在登录表单中找到密码输入框（type=password），输入: {password}",
            elements=elements,
        )
        print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败: ' + result['detail']}")

        # ── 步骤 4.5: 点击显示密码 ──
        print_step("4.5", "点击显示密码")
        elements = await adapter.get_elements()
        result = await smart_action(
            adapter,
            "点击密码输入框旁边的'显示'或'Show'按钮，让密码可见。",
            elements=elements,
        )
        print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败（非关键）'}")

        # ── 步骤 5: 提交登录 ──
        print_step(5, "提交登录")
        elements = await adapter.get_elements()
        result = await smart_action(
            adapter,
            "点击登录表单的提交按钮（如'登录'、'Login'、'Sign In'按钮）。注意不是导航栏的登录链接，而是表单内的提交按钮。",
            elements=elements,
        )
        print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败: ' + result['detail']}")
        await asyncio.sleep(3)

        state = await adapter.get_page_state()
        print(f"  登录后: {state.title} | URL: {state.url}")

        # ── 步骤 6: 回首页找帖子 ──
        print_step(6, "找碳基圈帖子")

        # 确保在首页
        if "/login" in (state.url or ""):
            print("  登录可能失败，仍在登录页")

        # 导航到碳基圈
        await adapter.navigate("https://ai6666.com/")
        await asyncio.sleep(2)

        elements = await adapter.get_elements()
        result = await smart_action(
            adapter,
            "在碳基圈的帖子列表中，找一个具体的帖子（不是导航栏按钮、不是'AIHireHumans'logo、不是分类标签），点击进入帖子详情页。应该是某个用户发的内容帖子。",
            elements=elements,
        )
        print(f"  结果: {'✅ 成功' if result['success'] else '❌ 失败: ' + result['detail']}")
        await asyncio.sleep(2)

        state = await adapter.get_page_state()
        print(f"  帖子页: {state.title} | URL: {state.url}")

        # ── 步骤 7: 发表评论 ──
        print_step(7, "发表评论 'ACP测试'")
        elements = await adapter.get_elements()

        # 先找评论框
        result = await smart_action(
            adapter,
            "找到评论输入框（通常在帖子下方，placeholder 可能包含'评论'、'回复'、'说点什么'等），点击它使其获得焦点。",
            elements=elements,
        )

        if result["success"]:
            await asyncio.sleep(0.5)
            # 输入评论
            elements = await adapter.get_elements()
            result = await smart_action(
                adapter,
                "在评论输入框中输入文字: ACP测试",
                elements=elements,
            )
            print(f"  输入评论: {'✅ 成功' if result['success'] else '❌ 失败'}")

            if result["success"]:
                await asyncio.sleep(1)
                # 发送评论
                elements = await adapter.get_elements()
                result = await smart_action(
                    adapter,
                    "点击发送/提交/发表评论的按钮。",
                    elements=elements,
                )
                print(f"  发送评论: {'✅ 成功' if result['success'] else '❌ 失败'}")
                await asyncio.sleep(2)
        else:
            print("  ❌ 未找到评论框，可能需要滚动页面")

        # ── 完成 ──
        state = await adapter.get_page_state()
        print(f"\n{'='*60}")
        print(f"  流程完成！")
        print(f"  最终页面: {state.title}")
        print(f"  URL: {state.url}")
        print(f"  浏览器保持 15 秒供查看")
        print(f"{'='*60}")
        await asyncio.sleep(15)

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
