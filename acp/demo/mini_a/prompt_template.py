"""
Prompt 模板：将 UI 元素列表渲染为 Qwen2.5-3B 可理解的文本 prompt。

格式设计原则：
  - 元素列表用编号列出，每行一个，格式 [ID] type "label" @(cx,cy)
  - 支持历史动作，避免重复操作
  - 输出格式严格 JSON，便于 parse_action() 解析
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acp.demo.mini_a.perception import UIElement

SYSTEM_PROMPT = """\
你是一个 GUI 自动化助手。

步骤：
1. 阅读元素列表，判断当前屏幕状态
2. 对照指令，判断下一步该做什么
3. 输出 JSON

输出格式（只输出 JSON，不要其他任何内容）：
{"action":"click","element_id":数字,"reason":"一句话"}
{"action":"type","element_id":数字,"text":"内容","reason":"一句话"}
{"action":"done","reason":"一句话"}
{"action":"fail","reason":"一句话"}

判断规则：
- 弹窗特征：元素列表中有"今日特惠"或"立即领取"等促销文字 → 需要关闭弹窗
- 关闭弹窗：找 label 为空的小图标（在弹窗右上角的 X 按钮），坐标通常在 x>700
- 登录表单特征：元素中有"请输入用户名"或"请输入密码" → 需要填写表单
- 填写顺序：先 type 用户名，再 type 密码，再 click 登录按钮
- done 条件：元素列表中出现"欢迎登录"文字才算完成，否则不能判 done
- 如果上一步执行的 elem label 和本步相同，说明卡住了，换一个元素
"""

FEW_SHOT = """\
---步骤1 有弹窗---
元素：[0]icon"欢迎回来"@(518,109) [4]icon""@(799,287) [5]text"今日特惠"@(641,381) [9]icon"立即领取"@(639,489)
历史：[可见元素=10]
屏幕：有弹窗（今日特惠）。找 x>700 空 label 图标=X 按钮，elem=4。
{"action":"click","element_id":4,"reason":"关闭弹窗"}

---步骤2 弹窗已消失 填用户名---
元素：[0]icon"欢迎回来"@(518,109) [2]icon"请输入用户名"@(641,216) [4]icon"请输入密码"@(641,290) [5]icon"登录"@(640,346)
历史：[step1]click 4""→ok; [可见元素=6]
屏幕：无弹窗，有登录表单。elem=2 含"请输入用户名"=用户名输入框。
{"action":"type","element_id":2,"text":"demo","reason":"填用户名"}

---步骤3 填密码---
元素：[2]icon"demo"@(641,216) [4]icon"请输入密码"@(641,290) [5]icon"登录"@(640,346)
历史：[step1]click 4→ok; [step2]type 2"请输入用户名"→ok
屏幕：用户名已填(elem=2 变为"demo")，elem=4 含"请输入密码"=密码框。
{"action":"type","element_id":4,"text":"123456","reason":"填密码"}

---步骤4 点登录---
元素：[2]icon"demo"@(641,216) [4]icon"123456"@(641,290) [5]icon"登录"@(640,346)
历史：[step1]click 4→ok; [step2]type 2→ok; [step3]type 4→ok
屏幕：表单已填，elem=5"登录"=登录按钮。
{"action":"click","element_id":5,"reason":"点击登录按钮"}

---步骤5 完成---
元素：[0]text"欢迎登录"@(440,300) [1]text"用户：demo"@(440,340)
历史：[step1-4]全ok
屏幕：出现"欢迎登录"=成功。
{"action":"done","reason":"欢迎登录已出现"}
---
"""


def _infer_screen_state(elements: list) -> str:
    """根据元素 label 推断当前屏幕状态（辅助提示 LLM）。"""
    labels = " ".join((e.label or "") for e in elements).lower()
    if "今日特惠" in labels or "立即领取" in labels or "新用户" in labels:
        return "有弹窗（促销弹窗，需先关闭）"
    if "欢迎登录" in labels:
        return "登录成功（任务完成）"
    if "请输入用户名" in labels or "请输入密码" in labels:
        return "登录表单（需填写用户名和密码后点登录）"
    # 检查是否有类似"用户名已填"的状态
    has_username_filled = any(
        e.label and "请输入用户名" not in e.label and len(e.label) < 30
        and e.center_x > 580 and e.center_x < 700
        and e.center_y > 200 and e.center_y < 240
        for e in elements
    )
    if has_username_filled:
        return "登录表单（用户名已填）"
    return "未知状态"


NAIVE_SYSTEM_PROMPT = """\
你是一个 GUI 自动化助手。给定屏幕上的 UI 元素列表和用户指令，输出下一步操作。

输出格式（只输出 JSON，不要其他内容）：
{"action":"click","element_id":数字,"reason":"一句话"}
{"action":"type","element_id":数字,"text":"内容","reason":"一句话"}
{"action":"done","reason":"一句话"}
{"action":"fail","reason":"一句话"}
"""


def render_naive_prompt(
    instruction: str,
    elements: list["UIElement"],
    history: list[str],
) -> tuple[str, str]:
    """极简 prompt：无 few-shot，无屏幕状态推断，无弹窗提示。
    用于 A2 naive 对照实验，完全依赖 LLM 自身推理。
    """
    elem_lines = []
    for e in elements:
        label = (e.label or "").strip().replace("\n", " ")[:60]
        elem_lines.append(
            f"[{e.idx}] {e.elem_type} \"{label}\" @({e.center_x:.0f},{e.center_y:.0f})"
        )

    elem_block = "\n".join(elem_lines) if elem_lines else "（未检测到元素）"
    history_block = "; ".join(history) if history else "（无）"

    user_content = (
        f"元素列表：\n{elem_block}\n\n"
        f"指令：{instruction}\n"
        f"历史：{history_block}\n"
        f"输出："
    )
    return NAIVE_SYSTEM_PROMPT, user_content


def render_prompt(
    instruction: str,
    elements: list["UIElement"],
    history: list[str],
) -> str:
    """渲染完整 prompt（system + few-shot + 当前状态）。"""
    elem_lines = []
    for e in elements:
        label = (e.label or "").strip().replace("\n", " ")[:60]
        elem_lines.append(
            f"[{e.idx}] {e.elem_type} \"{label}\" @({e.center_x:.0f},{e.center_y:.0f})"
        )

    elem_block = "\n".join(elem_lines) if elem_lines else "（未检测到元素）"
    history_block = "; ".join(history) if history else "（无）"
    screen_state = _infer_screen_state(elements)

    user_content = (
        f"当前屏幕状态：{screen_state}\n\n"
        f"元素列表：\n{elem_block}\n\n"
        f"指令：{instruction}\n"
        f"历史：{history_block}\n"
        f"输出："
    )

    return SYSTEM_PROMPT, FEW_SHOT + "\n" + user_content
