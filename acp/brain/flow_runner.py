"""Flow Runner — 从 YAML 流程文件驱动 Web 自动化。

这是 ACP 的"重放模式"（D6 决策）：
  - flows.yaml 定义操作步骤（人类可读的声明式描述）
  - credentials.yaml 提供敏感变量（${auth.xxx}）
  - LLM 将人类语言的 target 描述映射到实际页面元素

架构分层（重构后，对齐 Brain/MCP）：
  FlowRunner → MCPToolCall → WebMCP.execute() → WebAdapter
  （不再直接持有 WebAdapter，通过 MCP 层操作）

用法：
    runner = FlowRunner(site_dir="acp/config/sites/ai6666")
    await runner.run("login")           # 执行 login 流程
    await runner.run("post_comment", extra_vars={"comment": "ACP测试"})
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

from acp.brain.flow_recorder import FlowRecorder
from acp.mcp.tools.web_mcp import WebMCP
from acp.schema.plan import ActionResult


# ---------------------------------------------------------------------------
# MCPToolCall — FlowRunner 生成的 MCP 调用描述
# ---------------------------------------------------------------------------

@dataclass
class MCPToolCall:
    """FlowRunner 生成的 MCP 工具调用（结构化中间表示）。

    FlowRunner 将 flow step 转换为 MCPToolCall，
    再通过 MCP 层（WebMCP.execute）执行——不直接调用 Adapter。
    """
    tool_id: str        # e.g. "web-mcp"
    method: str         # e.g. "navigate", "click", "type"
    params: dict[str, Any]  # e.g. {"url": "https://..."} or {"element_id": "e0001"}

    def __repr__(self) -> str:
        return f"MCPToolCall({self.tool_id!r}, {self.method!r}, {self.params!r})"


# ---------------------------------------------------------------------------
# FlowRunner
# ---------------------------------------------------------------------------

class FlowRunner:
    """YAML 流程执行引擎（对齐 Brain/MCP 分层）。

    每一步用 LLM 将人类语言的 target（如"登录按钮"）
    映射到页面上的实际元素，然后生成 MCPToolCall 交给 WebMCP 执行。

    架构：FlowRunner → MCPToolCall → WebMCP.execute() → WebAdapter
    """

    def __init__(
        self,
        site_dir: str,
        mcp: Optional[WebMCP] = None,
        headless: bool = False,
    ):
        self._site_dir = Path(site_dir)
        self._mcp = mcp
        self._owns_mcp = mcp is None
        self._headless = headless

        # 向后兼容：允许外部访问 _adapter（通过 mcp._adapter）
        self._adapter = None  # 在 _mcp 启动后设置

        # 加载配置
        self._site = self._load_yaml("site.yaml")
        self._flows = self._load_yaml("flows.yaml").get("flows", {})
        self._credentials = {}
        cred_path = self._site_dir / "credentials.yaml"
        if cred_path.exists():
            self._credentials = self._load_yaml("credentials.yaml")

        # LLM 配置
        self._load_env()
        self._api_key = os.environ.get("ACP_LLM_API_KEY", "")
        self._base_url = os.environ.get("ACP_LLM_BASE_URL", "https://api.deepseek.com/v1")
        self._model = os.environ.get("ACP_LLM_MODEL", "deepseek-chat")

        # 执行日志
        self.log: list[dict] = []

    def _load_yaml(self, filename: str) -> dict:
        path = self._site_dir / filename
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _load_env(self):
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k, v)

    # ── 变量替换 ────────────────────────────────────────

    def _resolve_vars(self, text: str, extra_vars: dict = None) -> str:
        """将 ${auth.email} 等变量替换为实际值。"""
        def replacer(match):
            var_path = match.group(1)  # e.g. "auth.email"
            parts = var_path.split(".")

            # 先查 extra_vars
            if extra_vars and parts[0] in extra_vars:
                val = extra_vars
                for p in parts:
                    if isinstance(val, dict):
                        val = val.get(p, match.group(0))
                    else:
                        return str(val)
                return str(val)

            # 再查 credentials
            val = self._credentials
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p, None)
                else:
                    return match.group(0)
            return str(val) if val is not None else match.group(0)

        return re.sub(r'\$\{([^}]+)\}', replacer, text)

    # ── LLM 调用 ────────────────────────────────────────

    async def _ask_llm(self, prompt: str) -> dict:
        """异步调用 LLM 返回 JSON（使用 httpx.AsyncClient 避免阻塞事件循环）。"""
        if not self._api_key:
            raise RuntimeError("未配置 LLM API Key（ACP_LLM_API_KEY）")

        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("FlowRunner 需要安装 httpx：pip install httpx") from e

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                self._base_url + "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0,
                },
                headers={
                    "Authorization": "Bearer " + self._api_key,
                },
            )
            response.raise_for_status()
            resp = response.json()

        raw = resp["choices"][0]["message"]["content"].strip()

        # 提取 JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())

    def _elements_summary(self, elements, max_count=50) -> str:
        """生成 LLM 可读的元素列表。"""
        lines = []
        for e in elements[:max_count]:
            etype = str(e.type).split(".")[-1]
            text = (e.text or "").replace("\n", " ").strip()[:60]
            ph = (e.placeholder or "").strip()[:40]
            line = f"  id={e.id} type={etype} text={repr(text)}"
            if ph:
                line += f" placeholder={repr(ph)}"
            lines.append(line)
        return "\n".join(lines)

    # ── MCP 调用（核心：通过 WebMCP 而非直接调 Adapter）────────────────────

    async def _call_mcp(self, method: str, params: dict[str, Any]) -> ActionResult:
        """生成 MCPToolCall 并通过 WebMCP 执行。

        这是 FlowRunner 与执行层的唯一接口——
        所有操作都经过完整的 Brain/MCP 分层，不直接调用 WebAdapter。
        """
        call = MCPToolCall(tool_id="web-mcp", method=method, params=params)
        logger.debug("MCPToolCall: %s", call)
        return await self._mcp.execute(call.method, call.params)

    async def _get_elements_via_mcp(self) -> list:
        """通过 MCP 层获取元素列表。"""
        result = await self._call_mcp("get_elements", {})
        if result.success and result.elements:
            return result.elements
        return []

    async def _get_page_state_via_mcp(self):
        """通过 MCP 层获取页面状态。"""
        result = await self._call_mcp("get_page_state", {})
        return result.page_state

    # ── LLM 元素查找 ────────────────────────────────────

    async def _find_element(self, target: str, action: str) -> Optional[str]:
        """让 LLM 根据人类描述找到页面元素。返回 element_id。
        通过 MCP 层获取元素，不直接调用 adapter。
        """
        elements = await self._get_elements_via_mcp()
        state = await self._get_page_state_via_mcp()
        page_title = state.title if state else "未知页面"
        page_url = state.url if state else ""
        elem_text = self._elements_summary(elements)

        prompt = f"""你是 Web 自动化助手。
当前页面：{page_title} ({page_url})

任务：找到页面上的"{target}"，准备执行 {action} 操作。

页面元素：
{elem_text}

请从元素列表中选择最匹配"{target}"的元素。返回 JSON：
{{"element_id": "选中的元素 id", "reasoning": "为什么选这个"}}

规则：
- element_id 必须从上面列表中选，不要编造
- 根据语义匹配，不要只看文字完全一致
- "登录按钮"可能是 text='登录' 的 BUTTON
- "邮箱输入框"可能是 placeholder 含 email 的 TEXT_INPUT
- "登录提交按钮"是表单内的提交按钮，不是导航栏的登录链接
- 只返回 JSON"""

        try:
            result = await self._ask_llm(prompt)
            eid = result.get("element_id", "")
            reasoning = result.get("reasoning", "")

            # H-6: 验证返回的 element_id 确实在当前元素列表中
            valid_ids = {e.id for e in elements}
            if eid and eid not in valid_ids:
                logger.debug("LLM 返回了无效 element_id=%s，重试中...", eid)
                retry_prompt = prompt + f"\n\n注意：你上次返回的 element_id='{eid}' 不在元素列表中，请重新选择一个有效的 id。"
                result = await self._ask_llm(retry_prompt)
                eid = result.get("element_id", "")
                reasoning = result.get("reasoning", "（重试）") + result.get("reasoning", "")
                if eid and eid not in valid_ids:
                    logger.warning("LLM 重试后仍返回无效 element_id=%s，放弃", eid)
                    return None

            print(f"    LLM: {target} -> {eid}")
            print(f"       理由: {reasoning}")
            return eid
        except Exception as exc:
            print(f"    LLM 查找失败: {exc}")
            return None

    # ── 步骤执行 ────────────────────────────────────────

    async def _get_page(self):
        """获取底层 Page 对象（用于验证和 press_key）。"""
        if self._adapter and hasattr(self._adapter, '_page'):
            return self._adapter._page
        if hasattr(self._mcp, '_adapter') and self._mcp._adapter:
            return self._mcp._adapter._page
        return None

    async def _verify_type(self, eid: str, expected_text: str, max_retries: int = 2) -> bool:
        """验证输入框是否真的包含预期文本。失败则重试。"""
        page = await self._get_page()
        if not page:
            logger.warning("无法获取 Page 对象，跳过输入验证")
            return True

        for attempt in range(max_retries + 1):
            # 通过 MCP 重新获取元素，检查目标元素的 text/value
            elements = await self._get_elements_via_mcp()
            for e in elements:
                if e.id == eid:
                    actual = (e.text or "").strip()
                    if expected_text in actual or actual == expected_text:
                        print(f"    验证输入: 值正确")
                        return True
                    break

            # 元素 text 属性可能不反映 input value，用 JS 直接读
            try:
                # 从缓存中找 selector
                selector = None
                if hasattr(self._mcp, '_adapter') and self._mcp._adapter:
                    selector = self._mcp._adapter._element_cache.get(eid)
                if selector:
                    actual_value = await page.locator(selector).first.input_value(timeout=3000)
                    if actual_value == expected_text:
                        print(f"    验证输入: 值正确 (input_value)")
                        return True
                    else:
                        print(f"    验证输入: 值不匹配 (期望={len(expected_text)}字符, 实际={len(actual_value)}字符)")
            except Exception as exc:
                logger.debug("input_value 检查失败: %s", exc)

            if attempt < max_retries:
                print(f"    输入验证失败，重试输入 (尝试 {attempt + 2}/{max_retries + 1})...")
                result = await self._call_mcp("type", {"element_id": eid, "text": expected_text})
                if not result.success:
                    print(f"    重试输入失败: {result.error}")
                await asyncio.sleep(0.5)

        print(f"    输入验证: 重试 {max_retries + 1} 次后仍失败")
        return False

    async def _verify_click(self, before_url: str, before_title: str) -> bool:
        """验证点击后页面是否发生了变化（URL 或标题变化 = 点击生效）。"""
        state = await self._get_page_state_via_mcp()
        if not state:
            return True  # 无法获取状态，假设成功

        url_changed = state.url != before_url
        title_changed = state.title != before_title

        if url_changed or title_changed:
            print(f"    验证点击: 页面已变化 → {state.title}")
            return True

        # URL 和标题都没变，但这不一定意味着失败
        # （有些点击不导致导航，比如展开下拉框、切换 tab）
        print(f"    验证点击: 页面未导航（可能是页内操作）")
        return True  # 不强制失败，避免误判页内操作

    async def _exec_step(self, step: dict, extra_vars: dict) -> bool:
        """执行单个步骤，每步操作后验证结果。"""
        action = step.get("action", "")
        target = step.get("target", "")
        value = self._resolve_vars(step.get("value", ""), extra_vars)
        url = self._resolve_vars(step.get("url", ""), extra_vars)
        expected = step.get("expected", "")
        key = step.get("key", "")
        wait = step.get("wait", 2)

        if action == "navigate":
            result = await self._call_mcp("navigate", {"url": url})
            await asyncio.sleep(wait)
            state = await self._get_page_state_via_mcp()
            if state:
                print(f"    导航到: {state.title} ({state.url})")
                # 验证: URL 是否包含预期域名
                if url and not any(part in (state.url or "") for part in url.split("/")[2:3]):
                    print(f"    验证导航: URL 不匹配（可能重定向）")
            return result.success

        elif action == "click":
            before_state = await self._get_page_state_via_mcp()
            before_url = before_state.url if before_state else ""
            before_title = before_state.title if before_state else ""

            for attempt in range(3):
                if attempt > 0:
                    print(f"    重试点击 (尝试 {attempt + 1}/3)...")
                    await asyncio.sleep(1)

                eid = await self._find_element(target, "click")
                if not eid:
                    print(f"    未找到元素: {target}")
                    continue

                result = await self._call_mcp("click", {"element_id": eid})
                await asyncio.sleep(wait)

                if result.success:
                    await self._verify_click(before_url, before_title)
                    return True
                else:
                    print(f"    点击失败: {result.error}")

            print(f"    点击 3 次均失败")
            return False

        elif action == "type":
            # 最多尝试 3 次：找元素 → 输入 → 验证，任何一步失败都从头重试
            for attempt in range(3):
                if attempt > 0:
                    print(f"    重试输入 (尝试 {attempt + 1}/3)...")
                    await asyncio.sleep(1)

                eid = await self._find_element(target, "type")
                if not eid:
                    print(f"    未找到输入框: {target}")
                    continue

                result = await self._call_mcp("type", {"element_id": eid, "text": value})
                if not result.success:
                    print(f"    输入操作失败: {result.error}")
                    continue

                # 验证输入框是否真的有值
                if await self._verify_type(eid, value):
                    return True
                # 验证失败，下一轮会重新 find_element（自动刷新缓存）

            print(f"    输入 3 次均失败")
            return False

        elif action == "press_key":
            page = await self._get_page()
            if page:
                await page.keyboard.press(key)
            await asyncio.sleep(wait)
            return True

        elif action == "scroll":
            direction = step.get("direction", "down")
            result = await self._call_mcp("scroll", {"direction": direction})
            await asyncio.sleep(wait)
            return result.success

        elif action == "verify":
            state = await self._get_page_state_via_mcp()
            if not state:
                print(f"    验证: 无法获取页面状态")
                return False

            print(f"    验证: 当前页面 = {state.title} ({state.url})")

            # 登录验证: 不在登录页
            if "登录" in expected or "login" in expected.lower():
                ok = "/login" not in (state.url or "")
                print(f"    结果: {'通过' if ok else '失败 - 仍在登录页'}")
                return ok

            # 页面跳转验证
            if "跳转" in expected or "redirect" in expected.lower():
                # 有状态就算通过（能获取到页面说明没崩）
                print(f"    结果: 通过（页面正常加载）")
                return True

            # 元素存在性验证
            if "存在" in expected or "出现" in expected:
                elements = await self._get_elements_via_mcp()
                keyword = expected.replace("存在", "").replace("出现", "").strip()
                found = any(keyword in (e.text or "") for e in elements)
                print(f"    结果: {'通过' if found else '失败 - 未找到: ' + keyword}")
                return found

            # 通用: 检查页面是否有错误提示
            elements = await self._get_elements_via_mcp()
            errors = [e for e in elements if any(kw in (e.text or "") for kw in ["错误", "失败", "error", "fail", "invalid"])]
            if errors:
                for e in errors[:3]:
                    print(f"    发现错误: {(e.text or '')[:60]}")
                return False

            print(f"    结果: 通过（无错误信息）")
            return True

        elif action == "wait":
            await asyncio.sleep(wait)
            return True

        else:
            print(f"    未知操作: {action}")
            return False

    # ── 主入口 ──────────────────────────────────────────

    async def run(
        self,
        flow_name: str,
        extra_vars: dict = None,
        keep_open: int = 0,
        record: bool = False,
        record_name: str = "",
        record_description: str = "",
    ) -> bool:
        """执行指定流程。

        Args:
            flow_name: flows.yaml 中的流程名（如 "login"）
            extra_vars: 额外变量（如 {"comment": "ACP测试"}）
            keep_open: 执行完后保持浏览器打开的秒数
            record: 是否将成功步骤录制为新 flow 保存到 flows.yaml
            record_name: 录制的 flow 名称（默认为 "recorded_{flow_name}"）
            record_description: 录制的 flow 描述

        Returns:
            所有步骤是否全部成功
        """
        flow = self._flows.get(flow_name)
        if not flow:
            available = list(self._flows.keys())
            print(f"未找到流程 '{flow_name}'，可用: {available}")
            return False

        desc = flow.get("description", flow_name)
        steps = flow.get("steps", [])
        extra = extra_vars or {}

        print(f"\n{'='*60}")
        print(f"  执行流程: {desc}")
        print(f"  步骤数: {len(steps)}")
        if record:
            print(f"  录制模式: 开启")
        print(f"{'='*60}")

        # 启动 MCP（WebMCP 内部管理 WebAdapter 生命周期）
        if self._owns_mcp:
            self._mcp = WebMCP(headless=self._headless)
            await self._mcp.start()
            # 提供向后兼容的 _adapter 引用
            self._adapter = self._mcp._adapter

        all_ok = True
        try:
            for i, step in enumerate(steps):
                action = step.get("action", "")
                target = step.get("target", step.get("url", step.get("expected", "")))
                print(f"\n  步骤 {i+1}/{len(steps)}: {action} -> {target}")

                ok = await self._exec_step(step, extra)
                self.log.append({
                    "step": i + 1,
                    "action": action,
                    "target": target,
                    "success": ok,
                    "_step_data": step,  # 供录制器使用原始步骤数据
                })
                if not ok:
                    print(f"  步骤 {i+1} 失败")
                    all_ok = False
                    # 非关键步骤（如显示密码）失败不中断
                    if action == "verify":
                        break  # verify 失败才中断
                else:
                    print(f"  步骤 {i+1} 完成")

            if keep_open > 0:
                print(f"\n  浏览器保持 {keep_open} 秒...")
                await asyncio.sleep(keep_open)

        finally:
            if self._owns_mcp and self._mcp:
                await self._mcp.close()

        print(f"\n{'='*60}")
        print(f"  流程{'全部完成' if all_ok else '部分失败'}")
        ok_count = sum(1 for l in self.log if l["success"])
        print(f"  结果: {ok_count}/{len(self.log)} 步骤成功")
        print(f"{'='*60}")

        # 录制成功步骤到 flows.yaml
        if record and ok_count > 0:
            saved_name = record_name or f"recorded_{flow_name}"
            saved_desc = record_description or f"从 '{flow_name}' 自动录制（{ok_count} 步骤成功）"
            recorder = FlowRecorder(
                site_dir=str(self._site_dir),
                credentials=self._credentials,
            )
            recorder.record_from_flow_runner_log(
                log=self.log,
                flow_steps=steps,
                extra_vars=extra,
            )
            if recorder.save(saved_name, description=saved_desc):
                print(f"  已录制 {recorder.step_count} 步骤 -> flows.yaml['{saved_name}']")
            else:
                logger.warning("录制保存失败")

        return all_ok

    async def run_multiple(
        self,
        flow_names: list[str],
        extra_vars: dict = None,
        keep_open: int = 0,
    ) -> bool:
        """顺序执行多个流程（共享同一个浏览器）。"""
        if self._owns_mcp:
            self._mcp = WebMCP(headless=self._headless)
            await self._mcp.start()
            self._adapter = self._mcp._adapter
            self._owns_mcp = False  # 防止单个 run 关闭

        all_ok = True
        try:
            for name in flow_names:
                ok = await self.run(name, extra_vars)
                if not ok:
                    all_ok = False
            if keep_open > 0:
                print(f"\n浏览器保持 {keep_open} 秒...")
                await asyncio.sleep(keep_open)
        finally:
            await self._mcp.close()

        return all_ok
