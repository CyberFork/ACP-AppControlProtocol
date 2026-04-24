"""Flow Recorder — 将 FlowRunner 成功执行的操作序列录制为 YAML flow 文件。

当 FlowRunner 成功执行一系列操作后，FlowRecorder 自动将操作序列录制为人类可读的
YAML flow，追加到站点的 flows.yaml 中，供后续重放。

核心功能：
  1. 记录每步操作（action / target 描述 / value）
  2. 敏感信息检测：密码等自动替换为 ${auth.xxx} 变量（防止明文写入 YAML）
  3. 录制结果追加到站点的 flows.yaml（合并，不覆盖已有流程）

用法：
    recorder = FlowRecorder(site_dir="acp/config/sites/ai6666")
    recorder.record_step("navigate", url="https://example.com")
    recorder.record_step("click", target="登录按钮")
    recorder.record_step("type", target="邮箱输入框", value="user@example.com")
    recorder.record_step("type", target="密码输入框", value="s3cr3t")
    recorder.save("my_flow", description="我的录制流程")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 敏感信息检测
# ---------------------------------------------------------------------------

# 敏感字段名关键词（大小写不敏感）
_SENSITIVE_KEYWORDS = [
    "password", "passwd", "pwd", "密码",
    "secret", "token", "api_key", "apikey",
    "credential",
]

# target 描述中提示是密码字段的关键词
_PASSWORD_TARGET_KEYWORDS = ["密码", "password", "passwd", "pwd"]

# 用于检测 value 是否像密码（非明文可识别内容）
_MIN_SECRET_LENGTH = 6


def _is_sensitive_target(target: str) -> bool:
    """判断 target 描述是否指向敏感字段（如密码输入框）。"""
    target_lower = target.lower()
    return any(kw in target_lower for kw in _PASSWORD_TARGET_KEYWORDS)


def _sanitize_value(value: str, target: str, credentials: dict) -> str:
    """将敏感值替换为 ${auth.xxx} 变量。

    检测逻辑：
    1. 先反查 credentials，若 value 与某个字段值完全匹配 → 替换为对应变量名
    2. 若 target 描述包含密码关键词，且 value 长度 >= 6 → 替换为 ${auth.password}
    3. 其余情况保留原值
    """
    if not value:
        return value

    # 反查 credentials 字典（支持嵌套，如 auth.email）
    var_name = _find_in_credentials(value, credentials, prefix="")
    if var_name:
        return f"${{{var_name}}}"

    # 基于 target 语义推断
    if _is_sensitive_target(target) and len(value) >= _MIN_SECRET_LENGTH:
        logger.debug("检测到密码字段 target=%r，将 value 替换为变量", target)
        return "${auth.password}"

    return value


def _find_in_credentials(value: str, obj: Any, prefix: str) -> Optional[str]:
    """递归在 credentials 字典中查找与 value 完全匹配的项，返回变量路径。

    例如 credentials = {"auth": {"email": "user@example.com"}}
    _find_in_credentials("user@example.com", ...) → "auth.email"
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            result = _find_in_credentials(value, v, full_key)
            if result:
                return result
    elif isinstance(obj, str) and obj == value and value:
        return prefix
    return None


# ---------------------------------------------------------------------------
# 录制步骤数据结构
# ---------------------------------------------------------------------------


class RecordedStep:
    """单条录制的操作步骤。"""

    def __init__(
        self,
        action: str,
        target: str = "",
        value: str = "",
        url: str = "",
        direction: str = "",
        key: str = "",
        expected: str = "",
        wait: int = 1,
    ):
        self.action = action
        self.target = target
        self.value = value
        self.url = url
        self.direction = direction
        self.key = key
        self.expected = expected
        self.wait = wait

    def to_dict(self) -> dict:
        """转为 YAML 可序列化的字典，只包含非空字段。"""
        d: dict = {"action": self.action}
        if self.target:
            d["target"] = self.target
        if self.value:
            d["value"] = self.value
        if self.url:
            d["url"] = self.url
        if self.direction:
            d["direction"] = self.direction
        if self.key:
            d["key"] = self.key
        if self.expected:
            d["expected"] = self.expected
        if self.wait != 1:
            d["wait"] = self.wait
        return d


# ---------------------------------------------------------------------------
# FlowRecorder
# ---------------------------------------------------------------------------


class FlowRecorder:
    """将操作序列录制为 YAML flow，追加到站点的 flows.yaml。

    典型用法：
        recorder = FlowRecorder(site_dir="acp/config/sites/xxx")
        recorder.record_step("navigate", url="https://xxx.com")
        recorder.record_step("click", target="登录按钮")
        recorder.record_step("type", target="邮箱输入框", value="me@example.com")
        recorder.save("auto_login", description="自动录制的登录流程")
    """

    def __init__(self, site_dir: str, credentials: Optional[dict] = None):
        """
        Args:
            site_dir: 站点目录路径（含 flows.yaml）
            credentials: 凭据字典（用于敏感值检测）；若为 None 则尝试从 credentials.yaml 加载
        """
        self._site_dir = Path(site_dir)
        self._flows_path = self._site_dir / "flows.yaml"
        self._steps: list[RecordedStep] = []

        # 加载 credentials（用于敏感值反查）
        if credentials is not None:
            self._credentials = credentials
        else:
            cred_path = self._site_dir / "credentials.yaml"
            if cred_path.exists():
                with open(cred_path) as f:
                    self._credentials = yaml.safe_load(f) or {}
            else:
                self._credentials = {}

    # ---- 录制接口 ----

    def record_step(
        self,
        action: str,
        target: str = "",
        value: str = "",
        url: str = "",
        direction: str = "",
        key: str = "",
        expected: str = "",
        wait: int = 1,
    ) -> None:
        """录制一条操作步骤。

        Args:
            action: 操作类型（navigate / click / type / scroll / press_key / verify / wait）
            target: 人类语言的目标描述（如"登录按钮"）
            value: 输入文本（自动脱敏）
            url: navigate 操作的目标 URL
            direction: scroll 的方向（up/down/left/right）
            key: press_key 的按键名
            expected: verify 的期望描述
            wait: 操作后等待秒数
        """
        # 敏感值替换
        sanitized_value = _sanitize_value(value, target, self._credentials)
        if sanitized_value != value:
            logger.debug("record_step: value 已脱敏 target=%r", target)

        step = RecordedStep(
            action=action,
            target=target,
            value=sanitized_value,
            url=url,
            direction=direction,
            key=key,
            expected=expected,
            wait=wait,
        )
        self._steps.append(step)
        logger.debug("录制步骤 #%d: action=%s target=%r", len(self._steps), action, target)

    def record_from_flow_runner_log(
        self,
        log: list[dict],
        flow_steps: list[dict],
        extra_vars: Optional[dict] = None,
    ) -> None:
        """从 FlowRunner 的执行日志和原始步骤中批量录制成功的步骤。

        Args:
            log: FlowRunner.log（每条含 step/action/target/success）
            flow_steps: 原始 flow steps 列表（含完整字段）
            extra_vars: 额外变量（用于脱敏时反查）
        """
        merged_credentials = {**self._credentials}
        if extra_vars:
            merged_credentials.update(extra_vars)

        for log_entry in log:
            if not log_entry.get("success"):
                logger.debug("跳过失败步骤: %s", log_entry)
                continue
            step_idx = log_entry.get("step", 0) - 1
            if step_idx < 0 or step_idx >= len(flow_steps):
                continue
            step = flow_steps[step_idx]
            action = step.get("action", "")
            target = step.get("target", "")
            raw_value = step.get("value", "")
            sanitized_value = _sanitize_value(raw_value, target, merged_credentials)

            self._steps.append(RecordedStep(
                action=action,
                target=target,
                value=sanitized_value,
                url=step.get("url", ""),
                direction=step.get("direction", ""),
                key=step.get("key", ""),
                expected=step.get("expected", ""),
                wait=step.get("wait", 1),
            ))

    def clear(self) -> None:
        """清空当前录制缓冲区。"""
        self._steps.clear()

    @property
    def step_count(self) -> int:
        """当前已录制的步骤数。"""
        return len(self._steps)

    # ---- 保存接口 ----

    def save(self, flow_name: str, description: str = "") -> bool:
        """将录制结果追加到 flows.yaml。

        若同名流程已存在，会覆盖（提示日志）。不影响其他流程。

        Args:
            flow_name: 流程名称（如 "auto_login"）
            description: 流程描述（如 "自动录制的登录流程"）

        Returns:
            True 表示保存成功
        """
        if not self._steps:
            logger.warning("FlowRecorder.save(): 无已录制步骤，跳过保存")
            return False

        # 加载现有 flows.yaml
        existing: dict = {}
        if self._flows_path.exists():
            with open(self._flows_path) as f:
                existing = yaml.safe_load(f) or {}

        if "flows" not in existing or existing["flows"] is None:
            existing["flows"] = {}

        if flow_name in existing["flows"]:
            logger.info("FlowRecorder: 流程 '%s' 已存在，将被覆盖", flow_name)

        # 构建新流程
        new_flow = {
            "description": description or f"录制的流程：{flow_name}",
            "steps": [step.to_dict() for step in self._steps],
        }
        existing["flows"][flow_name] = new_flow

        # 写回（保留注释头部困难，用 yaml.dump 重写）
        self._site_dir.mkdir(parents=True, exist_ok=True)
        with open(self._flows_path, "w", encoding="utf-8") as f:
            yaml.dump(
                existing,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )

        logger.info(
            "FlowRecorder: 已保存流程 '%s'（%d 步骤）到 %s",
            flow_name,
            len(self._steps),
            self._flows_path,
        )
        return True
